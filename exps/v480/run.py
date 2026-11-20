"""冻结初态困难度的真实标签重加权对照。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.covariance import OAS
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits
from torch import nn

Array = NDArray[Any]
LOG = logging.getLogger(__name__)


def eer(y: Array, score: Array) -> float:
    """线性插值 ROC 交点，返回百分数。"""
    fp, tp, _ = roc_curve(y, score, drop_intermediate=False)
    d = fp + tp - 1
    i = int(np.flatnonzero(d >= 0)[0])
    if i == 0:
        return float(100 * fp[0])
    return float(100 * (fp[i - 1] - d[i - 1] * (fp[i] - fp[i - 1]) / (d[i] - d[i - 1])))


@torch.inference_mode()
def embed(x: Array, state: dict[str, Any], side: str, cfg: dict[str, Any]) -> Array:
    """按冻结 Linear、BN、ReLU、L2 前向，Dropout 在推断关闭。"""
    prefix = side + ".net."
    p = {
        k[len(prefix) :]: v.to(cfg["device"])
        for k, v in state.items()
        if k.startswith(prefix)
    }
    chunks = []
    for start in range(0, len(x), cfg["batch_size"]):
        a = torch.from_numpy(x[start : start + cfg["batch_size"]]).to(cfg["device"])
        a = F.linear(a, p["0.weight"], p["0.bias"])
        a = F.batch_norm(
            a,
            p["1.running_mean"],
            p["1.running_var"],
            p["1.weight"],
            p["1.bias"],
            training=False,
            eps=cfg["bn_epsilon"],
        )
        # [媒体数, 128]：每条媒体独立归一化。
        chunks.append(F.normalize(F.relu(a), dim=1).cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def fit(
    face: Array, voice: Array, groups: list[tuple[Array, Array]], cfg: dict[str, Any]
) -> dict[str, Array]:
    """同一身份的人脸、声音分别统计，避免假设媒体行一一配对。"""
    dim = face.shape[1]
    wf = np.zeros((dim, dim))
    wv = np.zeros_like(wf)
    means = []
    for fi, vi in groups:
        assert len(fi) >= 2 and len(vi) >= 2
        f, v = face[fi], voice[vi]
        mf, mv = f.mean(0), v.mean(0)
        rf, rv = f - mf, v - mv
        wf += rf.T @ rf / len(fi)
        wv += rv.T @ rv / len(vi)
        means.append(np.concatenate([mf, mv]))
    within = np.zeros((2 * dim, 2 * dim))
    within[:dim, :dim] = wf / len(groups)
    within[dim:, dim:] = wv / len(groups)
    model = OAS(store_precision=False).fit(np.stack(means))
    same = model.covariance_ + within
    same += np.eye(len(same)) * np.trace(same) / len(same) * cfg["jitter_relative"]
    different = same.copy()
    different[:dim, dim:] = 0
    different[dim:, :dim] = 0
    ls, ld = np.linalg.cholesky(same), np.linalg.cholesky(different)
    identity = np.eye(len(same))
    ps = np.linalg.solve(ls.T, np.linalg.solve(ls, identity))
    pd = np.linalg.solve(ld.T, np.linalg.solve(ld, identity))
    delta = ps - pd
    return {
        "mean": model.location_,
        "same": same,
        "different": different,
        "precision_delta": (delta + delta.T) / 2,
        "constant": np.asarray(np.log(np.diag(ld)).sum() - np.log(np.diag(ls)).sum()),
        "shrinkage": np.asarray(model.shrinkage_),
    }


def score(face: Array, voice: Array, pairs: Array, stats: dict[str, Array]) -> Array:
    """逐配对 LLR，同时用独立 Cholesky 二次型验证。"""
    x = np.concatenate([face[pairs[:, 0]], voice[pairs[:, 1]]], axis=1) - stats["mean"]
    result = (
        -0.5 * np.sum((x @ stats["precision_delta"]) * x, axis=1) + stats["constant"]
    )
    a = np.linalg.solve(np.linalg.cholesky(stats["same"]), x.T)
    b = np.linalg.solve(np.linalg.cholesky(stats["different"]), x.T)
    direct = -0.5 * (np.sum(a * a, axis=0) - np.sum(b * b, axis=0)) + stats["constant"]
    assert np.max(abs(result - direct)) < 1e-8
    return result


class Tower(nn.Module):
    """保留原投影，额外返回未施加Dropout的同一次激活。"""

    def __init__(self, dimension: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, cfg["model_dim"]),
            nn.BatchNorm1d(cfg["model_dim"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
        )
        self.skip = nn.Linear(dimension, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clean = self.net[2](self.net[1](self.net[0](x)))
        return F.normalize(self.net[3](clean), dim=-1), F.normalize(clean, dim=-1)


class Matcher(nn.Module):
    """参数注册顺序与原头相同；参数从已保存初态精确加载。"""

    def __init__(self, df: int, dv: int, classes: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.face, self.voice = Tower(df, cfg), Tower(dv, cfg)
        self.face.skip = nn.Linear(cfg["face_base_dim"], 1, bias=False)
        self.proto = nn.Parameter(torch.zeros(classes, cfg["model_dim"]))


class Objective(nn.Module):
    """原正负平衡 BCE，可学习尺度与中心。"""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.center = nn.Parameter(torch.zeros(()))
        self.bounds = cfg["scale_bounds"]

    def forward(
        self, f: torch.Tensor, v: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        logits = self.log_scale.exp().clamp(*self.bounds) * (f @ v.T - self.center)
        positive = labels[:, None] == labels[None, :]
        return 0.5 * (
            F.softplus(-logits[positive]).mean() + F.softplus(logits[~positive]).mean()
        )


def load_checkpoint(path: Path) -> dict[str, Any]:
    """仅允许原本地权重必要的NumPy字符串元数据。"""
    with torch.serialization.safe_globals(
        [np._core.multiarray.scalar, np.dtype, type(np.dtype("U1"))]
    ):
        return torch.load(path, map_location="cpu", weights_only=True)


def main() -> None:
    """保持初始化和采样一致，对照真实标签、冻结困难度与打乱权重。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    cfg = json.loads((here / "config.json").read_text())
    cfg["mode"] = args.mode
    assert sys.prefix == cfg["python_prefix"]
    assert not cfg["uses_evaluation_information"]
    torch.set_num_threads(cfg["cpu_threads"])
    threadpool_limits(cfg["cpu_threads"])
    torch.cuda.set_per_process_memory_fraction(cfg["gpu_fraction"])
    torch.backends.cuda.matmul.allow_tf32 = False
    run_id = (
        datetime.now().strftime("%m-%d-%H%M%S")
        + "-"
        + args.mode
        + "-"
        + uuid4().hex[:6]
    )
    out = root / cfg["output_root"] / cfg["experiment_id"] / run_id
    out.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")],
    )
    (out / "禁止正式提交.txt").write_text(
        "本运行仅使用训练身份真实标签与冻结困难度。外部来源资格仍需完整核验，未自动生成正式提交。\n"
    )
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    (out / "run.py").write_text(Path(__file__).read_text())
    (out / "environment.json").write_text(
        json.dumps(
            {
                "executable": sys.executable,
                "prefix": sys.prefix,
                "torch": torch.__version__,
                "argv": sys.argv,
            },
            indent=2,
        )
    )
    start = time.monotonic()
    hashes: dict[str, str] = {}

    def record(path: Path) -> Path:
        rel = str(path.relative_to(root))
        assert not any(token in rel for token in cfg["forbidden_input_tokens"]), rel
        if rel not in hashes:
            hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read(path: Path) -> Any:
        return json.loads(record(path).read_text())

    def array(spec: list[str]) -> Array:
        with np.load(record(root / spec[0]), allow_pickle=False) as z:
            return z[spec[1]].astype(np.float32)

    blocks = cfg["feature_blocks"]
    face = np.concatenate(
        [
            np.concatenate([array(s) for s in blocks[k]])
            for k in ("face", "face_additional")
        ],
        axis=1,
    )
    voice = np.concatenate(
        [
            np.concatenate([array(s) for s in blocks[k]], axis=1)
            for k in ("voice_train", "voice_extra")
        ]
    )
    ids = {
        side: np.asarray(
            [
                cfg["canonical_alias"].get(r["id"], r["id"])
                for path in cfg["metadata"][side]
                for r in read(root / path)
            ]
        )
        for side in ("face", "voice")
    }
    xf, xv = (
        torch.from_numpy(face).to(cfg["device"]),
        torch.from_numpy(voice).to(cfg["device"]),
    )
    tracker = None
    if args.mode == "full":
        try:
            import wandb

            tracker = wandb.init(
                project=cfg["wandb_project"],
                mode="offline",
                dir=str(out),
                id=run_id,
                group=cfg["experiment_id"],
                job_type="objective-transfer",
                config=cfg,
            )
        except Exception:
            LOG.exception("W&B初始化失败，保留本地数据")
    history: list[dict[str, Any]] = []

    def log(row: dict[str, Any]) -> None:
        history.append(row)
        assert time.monotonic() - start < cfg["max_seconds"]
        if row["step"] % cfg["log_interval"] == 0 or row["stage"] == "validation":
            LOG.info("阶段=%s 指标=%s", row["stage"], row)
            (out / "history.json").write_text(json.dumps(history, ensure_ascii=False))
        if tracker is not None:
            try:
                tracker.log(row)
            except Exception:
                LOG.exception("W&B记录失败")

    replay_checks = []
    parent = root / cfg["parent_run"]
    seeds = cfg["seeds"] if args.mode == "full" else cfg["seeds"][:1]
    folds = cfg["folds"] if args.mode == "full" else cfg["folds"][:1]
    steps = cfg["student_steps"] if args.mode == "full" else cfg["smoke_steps"]
    predictions: dict[str, Array] = {}
    diagnostics = []
    arms = cfg["arms"]
    for fold in folds:
        protocol = read(parent / "fold_protocols" / f"fold_{fold}.json")
        groups = [
            (
                np.flatnonzero(ids["face"][: cfg["official_rows"]] == name),
                np.flatnonzero(ids["voice"][: cfg["official_rows"]] == name),
            )
            for name in protocol["official_fit_ids"]
        ]
        for seed in seeds:
            key = f"B_s{seed}_f{fold}"
            plan = read(parent / "plans" / f"s{seed}_f{fold}_final.json")
            with np.load(
                record(parent / "plans" / f"s{seed}_f{fold}_final.npz"),
                allow_pickle=False,
            ) as z:
                sampling = {k: z[k] for k in z.files}
            assert steps <= len(sampling["classes"])
            initial = load_checkpoint(
                record(parent / "training" / (key + "_final") / "final.pt")
            )
            assert not set(plan["classes"]) & set(protocol["outer_ids"])
            donor = root / cfg["reference_run"]
            with np.load(record(donor / (key + "_auxiliary_plan.npz"))) as z:
                auxiliary = {side: z[side][:steps] for side in ids}
            positives = int(cfg["positive_fraction"] * cfg["auxiliary_batch_size"])
            true_labels = torch.cat(
                [
                    torch.ones(positives),
                    torch.zeros(cfg["auxiliary_batch_size"] - positives),
                ]
            ).to(cfg["device"])
            for side in ids:
                assert set(ids[side][auxiliary[side]].ravel()) <= set(
                    protocol["official_fit_ids"]
                )
            np.savez_compressed(out / (key + "_auxiliary_plan.npz"), **auxiliary)
            frozen = {
                side: embed(
                    features[: cfg["official_rows"]], initial["state"], side, cfg
                )
                for side, features in [("face", face), ("voice", voice)]
            }
            frozen_scale = float(
                initial["objective_state"]["log_scale"]
                .exp()
                .clamp(*cfg["scale_bounds"])
            )
            frozen_center = float(initial["objective_state"]["center"])
            frozen_logits = frozen_scale * (
                (
                    frozen["face"][auxiliary["face"]]
                    * frozen["voice"][auxiliary["voice"]]
                ).sum(-1)
                - frozen_center
            )
            labels_np = true_labels.cpu().numpy()
            signed = frozen_logits * (2 * labels_np - 1)
            difficulty = cfg["hardness_floor"] + np.exp(-np.logaddexp(0, signed))
            for part in [slice(0, positives), slice(positives, None)]:
                difficulty[:, part] /= difficulty[:, part].mean(axis=1, keepdims=True)
            np.savez_compressed(
                out / (key + "_difficulty.npz"),
                logits=frozen_logits,
                weights=difficulty,
            )
            difficulty_tensor = torch.as_tensor(
                difficulty, dtype=torch.float32, device=cfg["device"]
            )
            for arm in arms:
                torch.manual_seed(seed + fold)
                model = Matcher(
                    face.shape[1], voice.shape[1], len(plan["classes"]), cfg
                ).to(cfg["device"])
                model.load_state_dict(initial["state"], strict=True)
                objective = Objective(cfg).to(cfg["device"])
                objective.load_state_dict(initial["objective_state"], strict=True)
                optimizer = torch.optim.AdamW(
                    [*model.parameters(), *objective.parameters()],
                    lr=cfg["student_learning_rate"],
                    weight_decay=cfg["weight_decay"],
                )
                torch.manual_seed(seed + fold)
                for step in range(steps):
                    model.train()
                    fi, vi = (
                        sampling["face_indices"][step],
                        sampling["voice_indices"][step],
                    )
                    labels = torch.as_tensor(
                        np.repeat(sampling["classes"][step], cfg["per_id"]),
                        device=cfg["device"],
                    )
                    ef, _ = model.face(xf[fi])
                    ev, _ = model.voice(xv[vi])
                    main_loss = objective(ef, ev, labels)
                    # 训练身份辅助分支使用eval模式，保持与原对照一致的BN更新范围。
                    model.eval()
                    aux_face = xf[auxiliary["face"][step]]
                    aux_voice = xv[auxiliary["voice"][step]]
                    hard_target = true_labels
                    af, _ = model.face(aux_face)
                    av, _ = model.voice(aux_voice)
                    logits = objective.log_scale.exp().clamp(*cfg["scale_bounds"]) * (
                        (af * av).sum(1) - objective.center
                    )
                    per_pair = F.binary_cross_entropy_with_logits(
                        logits, hard_target, reduction="none"
                    )
                    raw_auxiliary_loss = per_pair.mean()
                    pair_weights = torch.ones_like(per_pair)
                    if arm != "hard_labels":
                        pair_weights = difficulty_tensor[step]
                        if arm == "shuffled_hardness":
                            pair_weights = torch.cat(
                                [
                                    torch.roll(
                                        pair_weights[:positives], cfg["shuffle_shift"]
                                    ),
                                    torch.roll(
                                        pair_weights[positives:], cfg["shuffle_shift"]
                                    ),
                                ]
                            )
                    auxiliary_loss = (pair_weights * per_pair).mean()
                    weight = cfg["auxiliary_weight"]
                    total = main_loss + weight * auxiliary_loss
                    optimizer.zero_grad()
                    total.backward()
                    optimizer.step()
                    log(
                        {
                            "stage": "student",
                            "head": key,
                            "arm": arm,
                            "step": step + 1,
                            "train/loss_main": float(main_loss.detach()),
                            "train/loss_aux/hard_labels": float(
                                raw_auxiliary_loss.detach()
                            ),
                            "train/loss_aux/reweighted": float(auxiliary_loss.detach()),
                            "train/pair_weight_std": float(
                                pair_weights.std(unbiased=False)
                            ),
                            "train/hard_labels_weight": weight,
                            "train/loss_weighted/hard_labels": float(
                                (weight * auxiliary_loss).detach()
                            ),
                            "train/loss_total": float(total.detach()),
                            "learning_rate": cfg["student_learning_rate"],
                        }
                    )
                state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                assert any(
                    not torch.equal(state[k], initial["state"][k])
                    for k in state
                    if k.endswith("weight")
                )
                if arm == "hard_labels" and args.mode == "full":
                    reference_control = load_checkpoint(
                        record(root / cfg["reference_run"] / f"{key}_hard_labels.pt")
                    )["state"]
                    differences = {
                        name: float((state[name] - reference_control[name]).abs().max())
                        for name in state
                    }
                    assert max(differences.values()) <= cfg["control_replay_tolerance"]
                    replay_checks.append(
                        {
                            "head": key,
                            "max_weight_difference": max(differences.values()),
                        }
                    )
                torch.save(
                    {
                        "state": state,
                        "objective_state": objective.state_dict(),
                        "config": cfg,
                        "research_only": True,
                    },
                    out / f"{key}_{arm}.pt",
                )
                ef = embed(face[: cfg["official_rows"]], state, "face", cfg)
                ev = embed(voice[: cfg["official_rows"]], state, "voice", cfg)
                stats = fit(ef, ev, groups, cfg)
                np.savez_compressed(
                    out / f"{key}_{arm}_embeddings.npz", face=ef, voice=ev
                )
                np.savez_compressed(
                    out / f"{key}_{arm}_stats.npz", allow_pickle=False, **stats
                )
                with np.load(
                    record(parent / "predictions.npz"), allow_pickle=False
                ) as old:
                    for proto in cfg["protocols"]:
                        pairs = old[f"fold{fold}_{proto}_pairs"].astype(int)
                        predictions[f"{key}_{proto}_{arm}"] = score(
                            ef, ev, pairs, stats
                        )
                        predictions[f"fold{fold}_{proto}_pairs"] = pairs
                        cosine = np.sum(ef[pairs[:, 0]] * ev[pairs[:, 1]], axis=1)
                        logits = float(
                            objective.log_scale.detach()
                            .exp()
                            .clamp(*cfg["scale_bounds"])
                        ) * (cosine - float(objective.center.detach()))
                        history.append(
                            {
                                "stage": "validation",
                                "head": key,
                                "arm": arm,
                                "step": steps,
                                "protocol": proto,
                                "val/loss": float(
                                    np.mean(
                                        np.logaddexp(0, logits) - pairs[:, 2] * logits
                                    )
                                ),
                                "val/eer": eer(
                                    pairs[:, 2], predictions[f"{key}_{proto}_{arm}"]
                                ),
                            }
                        )
                LOG.info("阶段=学生完成 head=%s arm=%s", key, arm)
    with np.load(record(parent / "predictions.npz"), allow_pickle=False) as old:
        for fold in folds:
            for proto in cfg["protocols"]:
                pairs = predictions[f"fold{fold}_{proto}_pairs"]
                row = {"fold": fold, "protocol": proto}
                for arm in arms:
                    values = np.mean(
                        [predictions[f"B_s{s}_f{fold}_{proto}_{arm}"] for s in seeds],
                        axis=0,
                    )
                    row[arm] = eer(pairs[:, 2], values)
                baseline = np.mean(
                    [old[f"B_s{s}_f{fold}_{proto}_gaussian"] for s in seeds], axis=0
                )
                row["original"] = eer(pairs[:, 2], baseline)
                diagnostics.append(row)
    np.savez_compressed(
        out / "diagnostic_predictions.npz", allow_pickle=False, **predictions
    )
    result = {
        "research_only": True,
        "formal_submission_allowed": False,
        "control_replay": replay_checks,
        "objective_transfer": "frozen_hardness",
        "uses_evaluation_information": False,
        "diagnostics": diagnostics,
        "diagnostic_mean": {
            arm: float(np.mean([r[arm] for r in diagnostics]))
            for arm in ["original", *arms]
        }
        if diagnostics
        else {},
        "elapsed_seconds": time.monotonic() - start,
    }
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (out / "history.json").write_text(json.dumps(history, ensure_ascii=False))
    (out / "input_hashes.json").write_text(json.dumps(hashes, indent=2))
    if tracker is not None:
        try:
            tracker.summary.update(result["diagnostic_mean"])
            tracker.finish()
        except Exception:
            LOG.exception("W&B关闭失败")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties

        font = FontProperties(fname=cfg["plot_font"])
        (out / "plots").mkdir()
        fig, axis = plt.subplots(figsize=(10, 5))
        for arm in arms:
            subset = [r for r in history if r["stage"] == "student" and r["arm"] == arm]
            for metric in [
                "train/loss_aux/hard_labels",
                "train/loss_main",
                "train/loss_weighted/hard_labels",
                "train/loss_total",
                "train/loss_aux/reweighted",
                "train/pair_weight_std",
            ]:
                axis.plot(
                    range(1, steps + 1),
                    [
                        np.mean([r[metric] for r in subset if r["step"] == i])
                        for i in range(1, steps + 1)
                    ],
                    label=arm + ":" + metric,
                )
        for ax in [axis]:
            ax.set_xlabel("更新步数", fontproperties=font)
            ax.set_ylabel("损失", fontproperties=font)
            ax.legend(prop=font, fontsize=7)
        fig.savefig(out / "plots/loss.png", dpi=cfg["plot_dpi"], bbox_inches="tight")
        plt.close(fig)
    except Exception:
        LOG.exception("绘图失败，保留完整JSON")
    LOG.info("阶段=完成 output=%s result=%s", out, result)


if __name__ == "__main__":
    main()
