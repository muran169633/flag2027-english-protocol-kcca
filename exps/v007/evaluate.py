"""冻结模型，使用训练身份原型评估原始余弦与自适应 S-norm。"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import roc_curve

from .config import CONFIG, Config, configure
from .data import Features, file_hash, load_cached_features, split_speakers, write_json
from .model import MatchingModel
from .protocol import eer, groups, validation_pairs

LOGGER = logging.getLogger("evaluate")


class Budget:
    def __init__(self, seconds: int) -> None:
        self.deadline = time.monotonic() + seconds

    def check(self) -> None:
        if time.monotonic() >= self.deadline:
            raise TimeoutError("评分对照达到本轮计算预算。")


@torch.inference_mode()
def encode(
    model: MatchingModel,
    features: Features,
    train: list[str],
    valid: list[str],
    config: Config,
    budget: Budget,
) -> dict[str, NDArray[np.float32]]:
    result: dict[str, NDArray[np.float32]] = {}
    for kind, tokens, globals_, labels in (
        ("face", features.faces, features.face_globals, features.face_speakers),
        ("voice", features.voices, features.voice_globals, features.voice_speakers),
    ):
        values = np.zeros((len(labels), config.embedding_dimension), dtype=np.float32)
        # 先按父实验相同的验证索引与 batch 形状编码，以核对原始评分重现。
        for speakers in (valid, train):
            allowed = set(speakers)
            indices = np.asarray(
                [i for i, speaker in enumerate(labels) if speaker in allowed],
                dtype=np.int64,
            )
            for start in range(0, len(indices), config.evaluation_batch_size):
                budget.check()
                selected = indices[start : start + config.evaluation_batch_size]
                local = torch.from_numpy(
                    np.asarray(tokens[selected], dtype=np.float32).copy()
                ).to(config.device)
                global_values = torch.from_numpy(
                    np.asarray(globals_[selected], dtype=np.float32).copy()
                ).to(config.device)
                if kind == "face":
                    output = model.encode_face(local, global_values)
                else:
                    mask = torch.from_numpy(features.voice_masks[selected].copy()).to(
                        config.device
                    )
                    output = model.encode_voice(local, mask, global_values)
                if not bool(torch.isfinite(output).all()):
                    raise ValueError("模型编码包含非有限值。")
                values[selected] = output.cpu().numpy()
        result[kind] = values
    return result


def prototypes(
    embeddings: NDArray[np.float32],
    labels: list[str],
    training_ids: list[str],
    config: Config,
) -> NDArray[np.float32]:
    grouped = groups(labels, training_ids)
    result = np.stack(
        [embeddings[grouped[speaker]].mean(0) for speaker in training_ids]
    )
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if not np.isfinite(result).all() or (norms < config.normalization_epsilon).any():
        raise ValueError("训练身份原型无效。")
    return result / norms


@torch.inference_mode()
def cohort_statistics(
    queries: NDArray[np.float32], cohort: NDArray[np.float32], config: Config
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    k = min(config.cohort_top_k, len(cohort))
    if k <= config.cohort_std_correction:
        raise ValueError("参照队列太小，无法估计分数标准差。")
    scores = (
        torch.from_numpy(queries).to(config.device)
        @ torch.from_numpy(cohort).to(config.device).T
    )
    selected = scores.topk(k, dim=1).values  # [N_query, K]，选最相似的训练身份原型
    mean = selected.mean(1)
    std = selected.std(1, correction=config.cohort_std_correction).clamp_min(
        config.cohort_std_floor
    )
    return mean.cpu().numpy(), std.cpu().numpy()


def normalized_scores(
    raw: NDArray[np.float32],
    face_indices: NDArray[np.int64],
    voice_indices: NDArray[np.int64],
    face_mean: NDArray[np.float32],
    face_std: NDArray[np.float32],
    voice_mean: NDArray[np.float32],
    voice_std: NDArray[np.float32],
) -> NDArray[np.float32]:
    return (
        0.5
        * (
            (raw - face_mean[face_indices]) / face_std[face_indices]
            + (raw - voice_mean[voice_indices]) / voice_std[voice_indices]
        )
    ).astype(np.float32)


def plot_scores(
    config: Config,
    run_dir: Path,
    scores: dict[
        str, tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.float32]]
    ],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_manager.fontManager.addfont(config.plot_font_file)
    plt.rcParams["font.sans-serif"] = [
        font_manager.FontProperties(fname=config.plot_font_file).get_name(),
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(
        1, len(scores), figsize=(6 * len(scores), 4), squeeze=False
    )
    for axis, (name, (labels, raw, norm)) in zip(
        axes.flat, scores.items(), strict=True
    ):
        for values, label in ((raw, "原始余弦"), (norm, "自适应 S-norm")):
            fpr, tpr, _ = roc_curve(labels, values, drop_intermediate=False)
            axis.plot(
                fpr * 100,
                (1 - tpr) * 100,
                label=f"{label}，EER={eer(labels, values) * 100:.2f}%",
            )
        axis.set_title("标准配对" if name == "standard" else "同性别配对")
        axis.set_xlabel("误接受率（%）")
        axis.set_ylabel("误拒绝率（%）")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    figure.suptitle("同一冻结模型与配对：训练身份 cohort 分数归一化")
    figure.tight_layout()
    directory = run_dir / "plots"
    directory.mkdir(exist_ok=True)
    figure.savefig(directory / "DET.png", dpi=config.plot_dpi)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "smoke"], default=CONFIG.mode)
    args = parser.parse_args()
    config = configure(args.mode)
    expected = Path(config.training_python).resolve().parent.parent
    if (
        Path(sys.prefix).resolve() != expected
        or Path(sys.executable).resolve().parent.parent != expected
    ):
        raise RuntimeError("必须在已有 my_project_env 中运行评分对照。")
    if config.cohort_top_k < 2 or config.cohort_std_floor <= 0:
        raise ValueError("cohort 至少取两个原型，标准差下限必须为正。")
    run_id = datetime.now(UTC).strftime("%m-%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = (
        config.project_root / config.output_directory / config.experiment_id / run_id
    )
    run_dir.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "run.log", encoding="utf-8"),
        ],
    )
    effective = json.loads(json.dumps(asdict(config), default=str))
    effective.update(
        run_id=run_id,
        executable=sys.executable,
        prefix=sys.prefix,
        entrypoint=f"{config.training_python} -m exps.{config.experiment_id}.evaluate",
    )
    write_json(run_dir / "config.json", effective)
    LOGGER.info("解释器=%s prefix=%s run=%s", sys.executable, sys.prefix, run_dir)
    started = time.monotonic()
    tracker: Any = None
    state: dict[str, Any] = {"run_id": run_id, "mode": config.mode, "status": "running"}
    write_json(run_dir / "state.json", state)
    try:
        budget = Budget(config.compute_budget_seconds)
        torch.set_num_threads(config.cpu_threads)
        torch.manual_seed(config.seed)
        if config.device.startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(
                config.gpu_memory_fraction, config.device
            )
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32
        features = load_cached_features(config, run_dir)
        training_ids, validation_ids = split_speakers(features, config)
        assert not set(training_ids) & set(validation_ids)
        source = config.project_root / config.comparison_run
        checkpoint_path = source / "finetune/best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if config.mode == "full" and checkpoint["class_map"] != {
            speaker: index for index, speaker in enumerate(training_ids)
        }:
            raise ValueError("训练 cohort 与来源模型训练身份不符。")
        model = MatchingModel(config, checkpoint["num_classes"]).to(config.device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.requires_grad_(False).eval()
        embeddings = encode(
            model, features, training_ids, validation_ids, config, budget
        )
        face_cohort = prototypes(
            embeddings["face"], features.face_speakers, training_ids, config
        )
        voice_cohort = prototypes(
            embeddings["voice"], features.voice_speakers, training_ids, config
        )
        face_mean, face_std = cohort_statistics(
            embeddings["face"], voice_cohort, config
        )
        voice_mean, voice_std = cohort_statistics(
            embeddings["voice"], face_cohort, config
        )
        np.savez(run_dir / "cohort.npz", faces=face_cohort, voices=voice_cohort)
        np.savez(
            run_dir / "embeddings_and_stats.npz",
            faces=embeddings["face"],
            voices=embeddings["voice"],
            face_mean=face_mean,
            face_std=face_std,
            voice_mean=voice_mean,
            voice_std=voice_std,
        )
        write_json(
            run_dir / "cohort.json",
            {
                "speaker_ids": training_ids,
                "validation_speaker_ids": validation_ids,
                "disjoint": True,
                "source_checkpoint": str(
                    checkpoint_path.relative_to(config.project_root)
                ),
                "checkpoint_sha256": file_hash(checkpoint_path),
                "effective_top_k": min(config.cohort_top_k, len(training_ids)),
            },
        )
        result: dict[str, Any] = {"raw": {}, "adaptive_s_norm": {}}
        plotting: dict[
            str, tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.float32]]
        ] = {}
        parent_pairs = (
            np.load(source / "finetune/validation_pairs.npz")
            if config.mode == "full"
            else None
        )
        pair_data = {}
        for name, same_gender in (("standard", False), ("same_gender", True)):
            budget.check()
            pairs = validation_pairs(features, validation_ids, config, same_gender)
            if len(pairs) == 0:
                continue
            if parent_pairs is not None and not np.array_equal(
                pairs, parent_pairs[name]
            ):
                raise ValueError("当前评估配对与来源模型不一致。")
            raw = np.asarray(
                [
                    float(embeddings["face"][i] @ embeddings["voice"][j])
                    for i, j, _ in pairs
                ],
                dtype=np.float32,
            )
            norm = normalized_scores(
                raw,
                pairs[:, 0],
                pairs[:, 1],
                face_mean,
                face_std,
                voice_mean,
                voice_std,
            )
            if not np.isfinite(raw).all() or not np.isfinite(norm).all():
                raise ValueError("归一化分数包含非有限值。")
            result["raw"][name] = eer(pairs[:, 2], raw)
            result["adaptive_s_norm"][name] = eer(pairs[:, 2], norm)
            plotting[name] = (pairs[:, 2], raw, norm)
            pair_data.update(
                {f"{name}_pairs": pairs, f"{name}_raw": raw, f"{name}_normalized": norm}
            )
        np.savez(run_dir / "scores.npz", allow_pickle=False, **pair_data)
        if config.mode == "full":
            parent_metrics = json.loads((source / "result.json").read_text())[
                "finetune"
            ]["best_metrics"]
            if (
                abs(result["raw"]["standard"] - parent_metrics["val/eer"]) > 1e-8
                or abs(
                    result["raw"]["same_gender"] - parent_metrics["val/eer_same_gender"]
                )
                > 1e-8
            ):
                raise ValueError("原始 EER 未能精确重现来源结果。")
        result["delta_percentage_points"] = {
            key: (result["adaptive_s_norm"][key] - value) * 100
            for key, value in result["raw"].items()
        }
        write_json(run_dir / "result.json", result)
        if config.wandb_mode != "disabled":
            try:
                import wandb

                tracker = wandb.init(
                    project=config.wandb_project,
                    mode=config.wandb_mode,
                    dir=str(run_dir),
                    group=config.experiment_id,
                    job_type="score_normalization",
                    id=run_id,
                    config=effective,
                )
            except Exception:
                LOGGER.exception("W&B 初始化失败，保留本地记录。")
        with (run_dir / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            fields = ["phase", "global_step", "val/eer", "val/eer_same_gender"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for step, phase in enumerate(["raw", "adaptive_s_norm"]):
                row = {
                    "phase": phase,
                    "global_step": step,
                    "val/eer": result[phase].get("standard"),
                    "val/eer_same_gender": result[phase].get("same_gender"),
                }
                writer.writerow(row)
                if tracker is not None:
                    try:
                        tracker.log(row, step=step)
                    except Exception:
                        LOGGER.exception("W&B 记录失败，本地历史已保存。")
        try:
            plot_scores(config, run_dir, plotting)
        except Exception:
            LOGGER.exception("绘图失败，分数和指标已保存。")
        write_json(
            run_dir / "environment.json",
            {
                "executable": sys.executable,
                "prefix": sys.prefix,
                "torch": str(torch.__version__),
                "device": config.device,
                "source_sha256": {
                    path.name: file_hash(path)
                    for path in Path(__file__).parent.glob("*.py")
                },
            },
        )
        if config.mode == "full":
            documentation = Path(__file__).parent / "docs"
            score_path = documentation / "score.md"
            relative = (
                f"../../../{run_dir.relative_to(config.project_root)}/result.json"
            )
            timestamp = datetime.now(UTC).isoformat(timespec="seconds")
            row = f"| {timestamp} | [{run_id}]({relative}) | {result['raw']['standard'] * 100:.3f}% | {result['raw'].get('same_gender', 0) * 100:.3f}% | {result['adaptive_s_norm']['standard'] * 100:.3f}% | {result['adaptive_s_norm'].get('same_gender', 0) * 100:.3f}% | {time.monotonic() - started:.1f} 秒 |"
            score_path.write_text(
                score_path.read_text().replace("\n\n## LB", "\n" + row + "\n\n## LB", 1)
            )
            index = config.project_root / "exps.md"
            lines = index.read_text().splitlines()
            index_row = f"| [{config.experiment_id}](exps/{config.experiment_id}/docs/exp.md) | AS-norm 标准 EER {result['adaptive_s_norm']['standard'] * 100:.3f}% | — | 冻结来源模型；仅用训练身份 cohort 归一化，内部验证 |"
            existing = next(
                (
                    i
                    for i, line in enumerate(lines)
                    if line.startswith(f"| [{config.experiment_id}](")
                ),
                None,
            )
            if existing is None:
                lines.append(index_row)
            else:
                lines[existing] = index_row
            index.write_text("\n".join(lines) + "\n")
        state.update(
            status="smoke_complete" if config.mode == "smoke" else "complete",
            results=result,
        )
        LOGGER.info("评分对照完成：%s", json.dumps(result, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        state.update(status="stopped", reason="用户中断")
        return 130
    except Exception as error:
        state.update(status="failed", reason=str(error))
        LOGGER.exception("评分对照失败，保留产物。")
        raise
    finally:
        if tracker is not None:
            try:
                tracker.finish()
            except Exception:
                LOGGER.exception("W&B 关闭失败，本地结果已保存。")
        write_json(run_dir / "state.json", state)
        write_json(run_dir.parent / "latest.json", state)
        write_json(
            run_dir / "timing.json", {"wall_seconds": time.monotonic() - started}
        )


if __name__ == "__main__":
    raise SystemExit(main())
