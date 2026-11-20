"""固定预算对比增加机器筛选英语身份前后的训练结果。"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from protocol_loss import condition_loss, negative_masks
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from exps.v480.run import Matcher, Objective

SIDES = ("face", "voice")
LOG = logging.getLogger(__name__)
Pools = dict[str, tuple[np.ndarray, np.ndarray]]


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def save(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def feature_matrix(blocks: list[list[str]]) -> np.ndarray:
    columns = []
    for path, key in blocks:
        with np.load(ROOT / path, allow_pickle=False) as z:
            columns.append(z[key])
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def load_sources(
    ref: dict[str, Any], screened: dict[str, Any] | None
) -> tuple[list[torch.Tensor], list[list[dict[str, str]]], Pools]:
    raw = []
    historical = []
    newly_selected = []
    blocks = ref["feature_blocks"]
    for side in SIDES:
        old = read(ROOT / ref["metadata"][side][0])
        offset = len(old)
        historical.append(old)
        parts = (
            [[blocks["face"][i], blocks["face_additional"][i]] for i in (0, 1)]
            if side == "face"
            else [blocks["voice_train"], blocks["voice_extra"]]
        )
        first = feature_matrix(parts[0])
        assert len(first) == offset and first.shape[1] == sum(
            ref[f"{side}_block_dimensions"]
        )
        if screened is None:
            newly_selected.append({})
            raw.append(torch.from_numpy(first).to(ref["device"]))
            continue
        new = read(ROOT / ref["metadata"][side][1])
        sel = np.array(screened[f"selected_{side}_rows"], dtype=np.int64)
        assert np.array_equal(np.unique(sel), sel) and all(
            new[i]["src"] == "vox2t" for i in sel
        )
        grouped: dict[str, list[int]] = {}
        for i in sel:
            grouped.setdefault(new[i]["id"], []).append(int(i) + offset)
        newly_selected.append(grouped)
        second = feature_matrix(parts[1])
        assert len(second) == len(new) and first.shape[1] == second.shape[1]
        tensor = torch.from_numpy(np.concatenate((first, second))).to(ref["device"])
        raw.append(tensor)
        LOG.info("载入 %s，历史行 %d，外部行 %d", side, offset, len(new))
    assert newly_selected[0].keys() == newly_selected[1].keys()
    extra = {
        n: (np.asarray(v), np.asarray(newly_selected[1][n]))
        for n, v in newly_selected[0].items()
    }
    assert len(extra) == (0 if screened is None else screened["identities"])
    return raw, historical, extra


def draw_rows(
    pools: list[np.ndarray], classes: np.ndarray, seed: int, count: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = np.empty((*classes.shape, count), dtype=np.int64)
    for k, pool in enumerate(pools):
        use = classes == k
        ranks = rng.random((int(use.sum()), len(pool)))
        rows[use] = pool[np.argpartition(ranks, count - 1, axis=1)[:, :count]]
    return rows.reshape(len(classes), -1)


def recording_weights(
    indices: list[np.ndarray],
    videos: list[np.ndarray],
    identities: int,
    count: int,
    mode: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    # [步, 身份, 人脸样本, 语音样本]，两侧录制字段均存在时才判为同视频。
    f, v = [
        video[ix].reshape(len(ix), identities, count)
        for ix, video in zip(indices, videos, strict=True)
    ]
    same = (f[:, :, :, None] >= 0) & (f[:, :, :, None] == v[:, :, None, :])
    eligible = ~same.reshape(len(f), identities, count * count)
    fallback = ~eligible.any(axis=-1)
    eligible[fallback] = True
    kept = eligible.sum(axis=-1)
    if mode == "shuffled":
        perm = np.random.default_rng(seed).random(eligible.shape).argsort(axis=-1)
        eligible = np.take_along_axis(eligible, perm, axis=-1)
        np.testing.assert_array_equal(eligible.sum(axis=-1), kept)
    weights = (eligible / kept[:, :, None]).reshape(same.shape).astype(np.float32)
    np.testing.assert_allclose(weights.sum(axis=(2, 3)), 1, atol=1e-6)
    return weights, {
        "all_positive_pairs": same.size,
        "known_same_video_pairs": int(same.sum()),
        "excluded_pairs": int(eligible.size - eligible.sum()),
        "remaining_same_video_pairs": int(np.sum(same & (weights > 0))),
        "all_same_fallback_groups": int(fallback.sum()),
        "identity_groups": fallback.size,
    }


def positive_loss(
    obj: Objective,
    f: torch.Tensor,
    v: torch.Tensor,
    weights: torch.Tensor,
    classes: torch.Tensor,
    negative: torch.Tensor,
) -> torch.Tensor:
    n, k, _ = weights.shape
    logits = obj.log_scale.exp().clamp(*obj.bounds) * (f @ v.T - obj.center)
    positives = logits.reshape(n, k, n, k)[classes, :, classes, :]
    return 0.5 * (
        (F.softplus(-positives) * weights).sum(dim=(1, 2)).mean()
        + F.softplus(logits[negative]).mean()
    )


def stage(
    source: Pools,
    class_count: int,
    data: list[torch.Tensor],
    settings: dict[str, Any],
    cfg: dict[str, Any],
    ref: dict[str, Any],
    fold: int,
    seed: int,
    initial: dict[str, torch.Tensor] | None = None,
    videos: list[np.ndarray] | None = None,
    positive_mode: str = "all",
    condition: str | None = None,
    strata: dict[str, tuple[str, str]] | None = None,
    condition_weight: float = 0.0,
) -> tuple[dict[str, torch.Tensor], list[float], dict[str, Any]]:
    offset = (seed - 1) * cfg["replicate_stride"]
    count = cfg["per_identity"]
    names = sorted(source)
    rng = np.random.default_rng(settings["class_seed"] + fold + offset)
    classes = np.stack(
        [
            rng.choice(len(names), settings["identities"], replace=False)
            for _ in range(settings["steps"])
        ]
    )
    draws = []
    device = ref["device"]
    for j in range(len(SIDES)):
        ix = draw_rows(
            [source[n][j] for n in names],
            classes,
            settings["media_seed"] + 2 * fold + j + offset,
            count,
        )
        draws.append(ix)
    weights = None
    diagnostics: dict[str, Any] = {}
    if positive_mode != "all":
        assert videos is not None
        w, recording_counts = recording_weights(
            draws,
            videos,
            settings["identities"],
            count,
            positive_mode,
            settings["media_seed"] + 2 * fold + offset + len(SIDES),
        )
        diagnostics.update(recording_counts)
        weights = torch.from_numpy(w).to(device)
    indices = [torch.from_numpy(ix).to(device) for ix in draws]
    torch.manual_seed(cfg["torch_seed"] + fold + offset)
    model = Matcher(data[0].shape[1], data[1].shape[1], class_count, ref).to(device)
    if initial is not None:
        model.load_state_dict(initial)
    obj = Objective(ref).to(device)
    auxiliary = None
    masks = None
    parameters = [*model.parameters(), *obj.parameters()]
    if condition is not None:
        assert strata is not None and 0 < condition_weight < 1
        mask, counts = negative_masks(
            names,
            classes,
            strata,
            condition,
            settings["media_seed"] + 2 * fold + offset + len(SIDES),
        )
        masks = torch.from_numpy(mask).to(device)
        auxiliary = Objective(ref).to(device)
        parameters.extend(auxiliary.parameters())
        diagnostics["condition_counts"] = counts
    with torch.no_grad():
        obj.log_scale.fill_(math.log(cfg["initial_scale"]))
        obj.center.fill_(cfg["initial_center"])
        if auxiliary is not None:
            auxiliary.load_state_dict(obj.state_dict())
    opt = torch.optim.AdamW(
        parameters,
        lr=settings["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    class_index = torch.arange(settings["identities"], device=device)
    labels = class_index.repeat_interleave(count)
    negative = labels[:, None] != labels[None, :]
    losses = []
    components = []
    for step in range(settings["steps"]):
        # 每批由多个身份构成，同一身份的人脸与声音分别随机抽样。
        f, _ = model.face(data[0][indices[0][step]])
        v, _ = model.voice(data[1][indices[1][step]])
        if weights is None:
            loss = obj(f, v, labels)
        else:
            loss = positive_loss(obj, f, v, weights[step], class_index, negative)
            if step == 0:
                with torch.no_grad():
                    uniform = torch.full_like(weights[step], 1 / count**2)
                    torch.testing.assert_close(
                        positive_loss(obj, f, v, uniform, class_index, negative),
                        obj(f, v, labels),
                        rtol=1e-6,
                        atol=1e-7,
                    )
        if auxiliary is not None and masks is not None:
            conditional_loss = condition_loss(
                auxiliary,
                f,
                v,
                class_index,
                count,
                weights[step] if weights is not None else None,
                masks[step],
            )
            components.append(torch.stack((loss.detach(), conditional_loss.detach())))
            loss = (1 - condition_weight) * loss + condition_weight * conditional_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.detach())
    result = torch.stack(losses).cpu().tolist()
    if auxiliary is not None:
        diagnostics["standard_conditional_loss"] = (
            torch.stack(components).cpu().tolist()
        )
        diagnostics["conditional_weight"] = condition_weight
        diagnostics["scale_center"] = {
            name: [
                float(task.log_scale.detach().exp().clamp(*task.bounds).cpu()),
                float(task.center.detach().cpu()),
            ]
            for name, task in (("standard", obj), ("conditional", auxiliary))
        }
    assert np.isfinite(result).all()
    return (
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
        result,
        diagnostics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-mode", choices=("all", "cross_video", "shuffled"), default="all"
    )
    parser.add_argument(
        "--baseline-replicates",
        action="store_true",
        help="仅用原始英语训练池，以另外三个独立种子构造等模型数对照",
    )
    args = parser.parse_args()
    mode = args.positive_mode
    independent = args.baseline_replicates
    if independent and mode != "all":
        parser.error("等模型数对照沿用原模型的全部正例目标")
    cfg = read(HERE / "config.json")
    ref = read(ROOT / cfg["reference_config"])
    assert (
        Path(sys.executable).parent.parent
        == Path(sys.prefix)
        == Path(cfg["python_prefix"])
    )
    start = time.monotonic()
    suffix = "original_independent" if independent else mode
    out = HERE / "output" / (datetime.now().strftime("%m-%d-%H%M%S") + "-" + suffix)
    out.mkdir()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "训练.log"), logging.StreamHandler()],
    )
    torch.set_num_threads(cfg["resource_cpu_threads"])
    torch.cuda.set_per_process_memory_fraction(cfg["resource_gpu_fraction"])
    LOG.info("运行 %s 环境 %s 解释器 %s", out.name, sys.prefix, sys.executable)
    save(
        out / "实际配置.json",
        {
            "experiment": cfg,
            "positive_mode": mode,
            "training_pool": "original" if independent else "expanded",
            "replicate_seeds": [
                s + len(ref["seeds"]) if independent else s for s in ref["seeds"]
            ],
            "matcher": ref,
            "argv": sys.argv,
            "executable": sys.executable,
            "prefix": sys.prefix,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
        },
    )
    screened = None if independent else read(HERE / "output/英语片段筛选.json")
    raw, historical, extra = load_sources(ref, screened)
    videos = None
    if mode != "all":
        meta = [
            old + read(ROOT / ref["metadata"][side][1])
            for side, old in zip(SIDES, historical, strict=True)
        ]
        keys = {
            key: i
            for i, key in enumerate(
                sorted({r["video"] for rows in meta for r in rows if r["video"]})
            )
        }
        videos = [np.array([keys.get(r["video"], -1) for r in rows]) for rows in meta]
    split = ROOT / ref["parent_run"]
    history = {}
    membership = {}
    with np.load(ROOT / cfg["english_pools"], allow_pickle=False) as manifest:
        for fold in ref["folds"]:
            protocol = read(split / f"fold_protocols/fold_{fold}.json")
            names = sorted(
                key.split("__")[1]
                for key in manifest.files
                if key.startswith(f"{fold}_final__") and key.endswith("__face")
            )
            base: Pools = {
                n: (
                    manifest[f"{fold}_final__{n}__face"],
                    manifest[f"{fold}_final__{n}__voice"],
                )
                for n in names
            }
            assert base.keys().isdisjoint(extra)
            expanded = {**base, **extra}
            assert set(expanded).isdisjoint(protocol["outer_ids"])
            for n, rows in base.items():
                for records, idx in zip(historical, rows, strict=True):
                    assert all(
                        records[i]["lang"] == "English"
                        and ref["canonical_alias"].get(
                            records[i]["id"], records[i]["id"]
                        )
                        == n
                        for i in idx
                    )
            official: Pools = {
                n: (
                    base[n][0][base[n][0] < ref["official_rows"]],
                    base[n][1][base[n][1] < ref["official_rows"]],
                )
                for n in sorted(set(base) & set(protocol["official_fit_ids"]))
            }
            for n, rows in official.items():
                assert all(
                    len(ix) >= cfg["per_identity"]
                    and all(meta[i]["id"] == n for i in ix)
                    for meta, ix in zip(historical, rows, strict=True)
                )
            membership[fold] = {
                "base": names,
                "added": sorted(extra),
                "official_adapt": sorted(official),
                "held": protocol["outer_ids"],
            }
            for seed in ref["seeds"]:
                plan = read(split / f"plans/s{seed}_f{fold}_final.json")
                for j, side in enumerate(SIDES):
                    approved = plan[side + "_indices_by_class"]
                    assert all(
                        set(rows[j]) <= set(approved[n]) for n, rows in expanded.items()
                    )
                tag = f"fold{fold}_s{seed}"
                sources = {suffix: expanded}
                effective_seed = seed + len(ref["seeds"]) if independent else seed
                if (
                    not independent
                    and mode == "all"
                    and fold == ref["folds"][0]
                    and seed == ref["seeds"][0]
                ):
                    sources = {"control": base, **sources}
                for label, source in sources.items():
                    initial, primary, diagnostics = stage(
                        source,
                        len(source),
                        raw,
                        cfg["pretrain"],
                        cfg,
                        ref,
                        fold,
                        effective_seed,
                        videos=videos,
                        positive_mode=mode,
                    )
                    final, adaptation, _ = stage(
                        official,
                        len(source),
                        raw,
                        cfg["adapt"],
                        cfg,
                        ref,
                        fold,
                        effective_seed,
                        initial,
                    )
                    history[f"{label}_{tag}"] = {
                        "pretrain": primary,
                        "adapt": adaptation,
                        "positive_pair_counts": diagnostics,
                    }
                    if label == "control":
                        old = torch.load(
                            ROOT / cfg["baseline_heads"] / f"{tag}.pt",
                            weights_only=True,
                            map_location="cpu",
                        )
                        deltas = {
                            k: float((x - old[k]).abs().max()) for k, x in final.items()
                        }
                        save(out / "旧模型复跑差异.json", deltas)
                        LOG.info("旧模型权重复跑最大绝对差 %.8g", max(deltas.values()))
                    else:
                        torch.save(initial, out / f"{tag}_pretrain.pt")
                        torch.save(final, out / f"{tag}.pt")
                    save(out / "训练损失.json", history)
                    LOG.info(
                        "%s %s 预训练 %d 身份，适配 %d 身份，末损失 %.5f / %.5f",
                        label,
                        tag,
                        len(source),
                        len(official),
                        primary[-1],
                        adaptation[-1],
                    )
    save(out / "训练身份划分.json", membership)
    save(
        out / "训练完成.json",
        {
            "heads": len(ref["folds"]) * len(ref["seeds"]),
            "elapsed": time.monotonic() - start,
        },
    )
    LOG.info("已完成五折全部种子，耗时 %.1fs", time.monotonic() - start)


if __name__ == "__main__":
    main()
