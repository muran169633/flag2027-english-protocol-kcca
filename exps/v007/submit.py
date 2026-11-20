"""从四份官方开发集配对表生成 Codabench 距离分数提交包。"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn.utils.rnn import pad_sequence

from .config import CONFIG, Config, SubmissionCell, configure
from .data import file_hash, write_json
from .encoders import FaceEncoder, VoiceEncoder, load_face, load_voice
from .evaluate import Budget, cohort_statistics
from .model import MatchingModel

LOGGER = logging.getLogger("submit")


@dataclass(frozen=True)
class Trial:
    key: str
    voice: Path
    face: Path


def read_trials(config: Config, cell: SubmissionCell) -> tuple[list[Trial], Path]:
    root = (
        config.project_root / config.submission_data_directory / cell.track
    ).resolve()
    path = root / cell.trials_file
    trials: list[Trial] = []
    keys: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_number} 必须恰好包含三个字段。")
        key, voice_name, face_name = parts
        if key in keys:
            raise ValueError(f"配对编号重复：{key}")
        keys.add(key)
        media: list[Path] = []
        for name in (voice_name, face_name):
            relative = Path(name)
            resolved = (root / relative).resolve()
            if (
                relative.is_absolute()
                or not resolved.is_relative_to(root)
                or not resolved.is_file()
            ):
                raise ValueError(f"配对媒体缺失或路径越界：{name}")
            media.append(resolved)
        trials.append(Trial(key, media[0], media[1]))
    if len(trials) != cell.expected_pairs:
        raise ValueError(f"{path} 配对数量错误：{len(trials)} != {cell.expected_pairs}")
    if config.mode == "smoke":
        trials = trials[: config.submission_smoke_pairs]
    return trials, path


@torch.inference_mode()
def encode_trials(
    trials: list[Trial],
    face_encoder: FaceEncoder,
    voice_encoder: VoiceEncoder,
    model: MatchingModel,
    config: Config,
    budget: Budget,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    faces: list[NDArray[np.float32]] = []
    voices: list[NDArray[np.float32]] = []
    for start in range(0, len(trials), config.submission_batch_size):
        budget.check()
        batch = trials[start : start + config.submission_batch_size]
        images = torch.stack([load_face(trial.face, config) for trial in batch]).to(
            config.device
        )
        face_tokens, face_globals = face_encoder(images)
        # 训练时局部缓存为 FP16，读取后转换 FP32；推理保留相同量化步骤。
        face_tokens = face_tokens.to(torch.float16).to(torch.float32)
        face_embeddings = model.encode_face(face_tokens, face_globals)
        waveforms = [load_voice(trial.voice, config) for trial in batch]
        padded = pad_sequence(waveforms, batch_first=True).to(
            config.device
        )  # [B, T_max]
        lengths = torch.tensor(
            [len(value) / padded.shape[1] for value in waveforms], device=config.device
        )
        voice_tokens, mask, voice_globals = voice_encoder(padded, lengths)
        voice_tokens = voice_tokens.to(torch.float16).to(torch.float32)
        voice_embeddings = model.encode_voice(voice_tokens, mask, voice_globals)
        for values, destination in (
            (face_embeddings, faces),
            (voice_embeddings, voices),
        ):
            if not bool(torch.isfinite(values).all()):
                raise ValueError("原始媒体推理产生非有限嵌入。")
            destination.append(values.cpu().numpy().astype(np.float32))
        batch_number = start // config.submission_batch_size + 1
        if batch_number % config.submission_log_batches == 0 or start + len(
            batch
        ) == len(trials):
            LOGGER.info("推理进度=%d/%d", start + len(batch), len(trials))
    return np.concatenate(faces), np.concatenate(voices)


def verify_archive(
    archive_path: Path,
    expected: dict[str, tuple[list[Trial], NDArray[np.float32]]],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("ZIP 文件集合与四个官方协议不符。")
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC 校验失败。")
        for name, (trials, scores) in expected.items():
            rows = [
                line.split() for line in archive.read(name).decode("utf-8").splitlines()
            ]
            if len(rows) != len(trials) or any(len(row) != 2 for row in rows):
                raise ValueError(f"提交行数或列数错误：{name}")
            keys = [row[0] for row in rows]
            if keys != [trial.key for trial in trials] or len(set(keys)) != len(keys):
                raise ValueError(f"提交编号或顺序不符：{name}")
            actual = np.asarray([float(row[1]) for row in rows], dtype=np.float64)
            if not np.isfinite(actual).all() or not np.allclose(
                actual, scores, rtol=0, atol=1e-9
            ):
                raise ValueError(f"提交分数无效或与真实预测不符：{name}")
            counts[name] = len(rows)
    return {
        "valid": True,
        "pairs": sum(counts.values()),
        "files": counts,
        "sha256": file_hash(archive_path),
    }


def load_cohort(
    config: Config, checkpoint_hash: str, class_map: dict[str, int]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    source = config.project_root / config.submission_cohort_run
    provenance = json.loads((source / "cohort.json").read_text())
    if provenance["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("AS-norm 队列与匹配模型检查点不对应。")
    if provenance["speaker_ids"] != sorted(class_map) or set(
        provenance["speaker_ids"]
    ) & set(provenance["validation_speaker_ids"]):
        raise ValueError("AS-norm 队列身份与模型训练身份不对应。")
    if provenance["effective_top_k"] != config.cohort_top_k:
        raise ValueError("AS-norm 队列近邻数与已验证方案不对应。")
    with np.load(source / "cohort.npz", allow_pickle=False) as arrays:
        faces = arrays["faces"].astype(np.float32)
        voices = arrays["voices"].astype(np.float32)
    for values in (faces, voices):
        if (
            values.shape != (len(class_map), config.embedding_dimension)
            or not np.isfinite(values).all()
        ):
            raise ValueError("AS-norm 队列形状或数值无效。")
        if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5):
            raise ValueError("AS-norm 队列不是单位向量。")
    return faces, voices


def run(config: Config, run_dir: Path) -> dict[str, Any]:
    budget = Budget(config.submission_budget_seconds)
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(
            config.gpu_memory_fraction, config.device
        )
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    checkpoint_path = config.project_root / config.comparison_run / "finetune/best.pt"
    checkpoint_hash = file_hash(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cohort_faces, cohort_voices = load_cohort(
        config, checkpoint_hash, checkpoint["class_map"]
    )
    model = MatchingModel(config, checkpoint["num_classes"]).to(config.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False).eval()
    source_hashes = {
        str(checkpoint_path.relative_to(config.project_root)): checkpoint_hash
    }
    for filename, expected_hash in (
        (config.face_weights, config.face_weights_sha256),
        (config.voice_weights, config.voice_weights_sha256),
    ):
        actual_hash = file_hash(config.project_root / filename)
        if actual_hash != expected_hash:
            raise ValueError(f"冻结编码器权重哈希不符：{filename}")
        source_hashes[filename] = actual_hash
    for filename in ("cohort.npz", "cohort.json", "config.json"):
        path = config.project_root / config.submission_cohort_run / filename
        source_hashes[str(path.relative_to(config.project_root))] = file_hash(path)
    platform_path = config.project_root / config.submission_platform_snapshot
    source_hashes[config.submission_platform_snapshot] = file_hash(platform_path)
    face_encoder = FaceEncoder(config).to(config.device)
    voice_encoder = VoiceEncoder(config).to(config.device)
    if (
        torch.backends.cuda.matmul.allow_tf32 != config.allow_tf32
        or torch.backends.cudnn.allow_tf32 != config.allow_tf32
    ):
        raise ValueError("编码器导入后 TF32 设置发生漂移。")
    expected: dict[str, tuple[list[Trial], NDArray[np.float32]]] = {}
    metadata: list[dict[str, Any]] = []
    for cell in config.submission_cells:
        trials, protocol_path = read_trials(config, cell)
        LOGGER.info("开始协议=%s/%s 配对数=%d", cell.track, cell.language, len(trials))
        faces, voices = encode_trials(
            trials, face_encoder, voice_encoder, model, config, budget
        )
        raw = np.sum(faces * voices, axis=1, dtype=np.float32)
        face_mean, face_std = cohort_statistics(faces, cohort_voices, config)
        voice_mean, voice_std = cohort_statistics(voices, cohort_faces, config)
        similarity = (
            0.5 * ((raw - face_mean) / face_std + (raw - voice_mean) / voice_std)
        ).astype(np.float32)
        # Codabench Overview 明确要求 distance；参考评测代码先取负再计算 ROC。
        if config.submission_score_direction != "distance":
            raise ValueError("提交协议必须使用越低越匹配的距离分数。")
        scores = -similarity
        if not np.isfinite(scores).all():
            raise ValueError("AS-norm 提交分数包含非有限值。")
        name = f"{cell.track}/{cell.score_file}"
        target = run_dir / "scores" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(
                f"{trial.key} {float(score):.10f}\n"
                for trial, score in zip(trials, scores, strict=True)
            ),
            encoding="utf-8",
        )
        audit_path = run_dir / "predictions" / f"{cell.track}_{cell.language}.npz"
        audit_path.parent.mkdir(exist_ok=True)
        np.savez(
            audit_path,
            pair_ids=np.asarray([trial.key for trial in trials]),
            faces=faces,
            voices=voices,
            cosine=raw,
            similarity=similarity,
            distance=scores,
            face_mean=face_mean,
            face_std=face_std,
            voice_mean=voice_mean,
            voice_std=voice_std,
        )
        expected[name] = trials, scores
        metadata.append(
            {
                "track": cell.track,
                "language": cell.language,
                "pairs": len(trials),
                "protocol": str(protocol_path.relative_to(config.project_root)),
                "protocol_sha256": file_hash(protocol_path),
                "scores_sha256": file_hash(target),
                "minimum": float(scores.min()),
                "maximum": float(scores.max()),
            }
        )
        LOGGER.info(
            "完成协议=%s/%s 距离范围=[%.4f, %.4f]",
            cell.track,
            cell.language,
            scores.min(),
            scores.max(),
        )
    budget.check()
    archive_path = run_dir / config.submission_archive
    with ZipFile(archive_path, "x", compression=ZIP_DEFLATED) as archive:
        for name in expected:
            archive.write(run_dir / "scores" / name, arcname=name)
    verification = verify_archive(archive_path, expected)
    write_json(run_dir / "verification.json", verification)
    if config.mode == "full":
        destination = config.project_root / config.submission_archive
        if destination.exists():
            raise FileExistsError(
                f"根目录已有提交包，保留新运行产物而不覆盖：{destination}"
            )
        temporary = destination.with_suffix(".zip.tmp")
        shutil.copyfile(archive_path, temporary)
        temporary.replace(destination)
        if file_hash(destination) != verification["sha256"]:
            raise ValueError("根目录提交包复制校验失败。")
    sources = {
        path.name: file_hash(path) for path in Path(__file__).parent.glob("*.py")
    }
    write_json(
        run_dir / "sources.json",
        {"input_hashes": source_hashes, "source_hashes": sources, "cells": metadata},
    )
    return {
        "status": "complete",
        "archive": str(archive_path),
        "score_direction": config.submission_score_direction,
        "phase": "Progress",
        **verification,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "smoke"], default=CONFIG.mode)
    args = parser.parse_args()
    config = configure(args.mode)
    expected_environment = Path(config.training_python).resolve().parent.parent
    if (
        Path(sys.prefix).resolve() != expected_environment
        or Path(sys.executable).resolve().parent.parent != expected_environment
    ):
        raise RuntimeError("原始编码器推理必须使用已有 my_project_env。")
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
        torch_version=str(torch.__version__),
        entrypoint=f"{config.training_python} -m exps.{config.experiment_id}.submit --mode {config.mode}",
    )
    write_json(run_dir / "config.json", effective)
    LOGGER.info("解释器=%s prefix=%s run=%s", sys.executable, sys.prefix, run_dir)
    started = time.monotonic()
    write_json(run_dir / "state.json", {"status": "running", "mode": config.mode})
    try:
        result = run(config, run_dir)
        result.update(
            elapsed_seconds=time.monotonic() - started, run_id=run_id, mode=config.mode
        )
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "state.json", result)
        LOGGER.info(
            "提交包已验证：%s 配对=%d 耗时=%.1f秒",
            result["archive"],
            result["pairs"],
            result["elapsed_seconds"],
        )
        return 0
    except BaseException as error:
        write_json(
            run_dir / "state.json",
            {
                "status": "failed",
                "error": repr(error),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        LOGGER.exception("提交包生成失败。")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
