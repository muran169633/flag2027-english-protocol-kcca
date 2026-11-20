"""仅提取官方训练数据的冻结年龄属性与语音隐藏表示。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits

from .traits import FrozenTraits

Array = NDArray[Any]
ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("嵌套身份验证")


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unit(values: Array, floor: float) -> Array:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), floor)


def eer(labels: Array, values: Array) -> float:
    fpr, tpr, _ = roc_curve(labels, values, drop_intermediate=False)
    difference = fpr + tpr - 1
    right = int(np.flatnonzero(difference >= 0)[0])
    if right == 0:
        return float(fpr[0])
    left = right - 1
    return float(
        fpr[left]
        - difference[left]
        * (fpr[right] - fpr[left])
        / (difference[right] - difference[left])
    )


class History:
    """W&B 故障不影响模型计算，JSONL 始终独立保存。"""

    def __init__(self, config: dict[str, Any], directory: Path) -> None:
        self.directory = directory
        self.rows: list[dict[str, Any]] = []
        self.tracker: Any = None
        if config["mode"] == "full":
            try:
                import wandb

                self.tracker = wandb.init(
                    project=config["wandb"]["project"],
                    mode=config["wandb"]["mode"],
                    dir=str(directory),
                    group=config["experiment_id"],
                    job_type=config["wandb"]["job_type"],
                    id=config["run_id"],
                    config=config,
                )
            except Exception:
                LOGGER.exception("W&B初始化失败，继续独立记录")

    def log(self, values: dict[str, Any]) -> None:
        row = {"step": len(self.rows), **values}
        self.rows.append(row)
        with (self.directory / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.tracker is not None:
            try:
                self.tracker.log(row, step=row["step"])
            except Exception:
                LOGGER.exception("W&B记录失败，JSONL已保存")

    def finish(self) -> None:
        fields = sorted(set().union(*(r.keys() for r in self.rows)))
        with (self.directory / "history.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)
        if self.tracker is not None:
            try:
                self.tracker.finish()
            except Exception:
                LOGGER.exception("W&B关闭失败，保留本地记录")


def execute(
    config: dict[str, Any], directory: Path, history: History, deadline: float
) -> dict[str, Any]:
    source = ROOT / config["data"]["manifest"]
    require(digest(source) == config["data"]["manifest_sha256"], "训练媒体清单指纹变化")
    records = read(source)
    raw_lines = [
        line.split()
        for line in (ROOT / config["inputs"]["train_list"]["path"])
        .read_text()
        .splitlines()
    ]
    require(
        digest(ROOT / config["inputs"]["train_list"]["path"])
        == config["inputs"]["train_list"]["sha256"],
        "训练输入指纹变化",
    )
    require(
        len(records) == config["data"]["rows"]
        and all(
            r["role"] == "training"
            and r["language"] == "English"
            and r["pair_id"] == line[0]
            and r["identity"] == line[4]
            for r, line in zip(records, raw_lines, strict=True)
        ),
        "只允许同序官方训练英语样本",
    )
    alignment_path = ROOT / config["face_resource"]["alignment_log"]
    require(
        digest(alignment_path) == config["face_resource"]["alignment_sha256"],
        "单图检测缓存指纹变化",
    )
    alignments = [json.loads(line) for line in alignment_path.read_text().splitlines()]
    require(
        len(alignments) == len(records)
        and all(
            a["pair_id"] == r["pair_id"]
            for a, r in zip(alignments, records, strict=True)
        ),
        "单图检测缓存行序不同",
    )
    count = (
        len(records)
        if config["mode"] == "full"
        else config["data"][f"{config['mode']}_rows"]
    )
    selected = records[:count]
    save(directory / "manifest.json", selected)
    started = time.monotonic()
    model = FrozenTraits(config)
    require(
        not any(parameter.requires_grad for parameter in model.audio.parameters())
        and not model.audio.training,
        "公开模型必须冻结且处于eval",
    )
    load_seconds = time.monotonic() - started
    faces = np.full((count, 3), np.nan, dtype=np.float32)
    voices = np.full((count, 4), np.nan, dtype=np.float32)
    hidden: Array | None = None
    audio_cache: dict[str, tuple[Array, Array]] = {}
    face_cache: dict[str, Array] = {}
    processed = 0
    try:
        for index, record in enumerate(selected):
            if time.monotonic() > deadline:
                raise TimeoutError("冻结属性提取达到整批预算")
            face_path, voice_path = (
                ROOT / record["face_path"],
                ROOT / record["voice_path"],
            )
            require(
                digest(face_path) == record["face_sha256"]
                and digest(voice_path) == record["voice_sha256"],
                "官方训练媒体内容变化",
            )
            face_key = record["face_sha256"] + json.dumps(alignments[index]["bbox"])
            if face_key not in face_cache:
                face_cache[face_key] = model.image(face_path, alignments[index])
            if record["voice_sha256"] not in audio_cache:
                audio_cache[record["voice_sha256"]] = model.voice(voice_path)
            face, (voice, embedding) = (
                face_cache[face_key],
                audio_cache[record["voice_sha256"]],
            )
            if hidden is None:
                hidden = np.full((count, len(embedding)), np.nan, dtype=np.float32)
            faces[index], voices[index], hidden[index] = face, voice, embedding
            processed = index + 1
            if processed % config["data"]["log_every_rows"] == 0 or processed == count:
                row = {
                    "phase": "frozen_features",
                    "rows": processed,
                    "unique_audio": len(audio_cache),
                    "unique_face": len(face_cache),
                    "elapsed_seconds": time.monotonic() - started,
                }
                history.log(row)
                LOGGER.info(
                    "阶段=冻结提取 rows=%d/%d unique_audio=%d elapsed=%.1fs",
                    processed,
                    count,
                    len(audio_cache),
                    row["elapsed_seconds"],
                )
        if hidden is None:
            raise ValueError("未生成任何冻结语音隐藏表示")
        require(
            bool(np.isfinite(faces).all())
            and bool(np.isfinite(voices).all())
            and bool(np.isfinite(hidden).all()),
            "属性特征缺失或非有限",
        )
        repeat_face = model.image(ROOT / selected[0]["face_path"], alignments[0])
        repeat_voice, repeat_hidden = model.voice(ROOT / selected[0]["voice_path"])
        repeat_error = float(
            max(
                np.max(np.abs(faces[0] - repeat_face)),
                np.max(np.abs(voices[0] - repeat_voice)),
                np.max(np.abs(hidden[0] - repeat_hidden)),
            )
        )
        require(
            repeat_error <= config["compatibility_tolerance"],
            "处理其他样本后属性预测改变",
        )
        np.savez(directory / "features.npz", face=faces, voice=voices, hidden=hidden)
        elapsed = time.monotonic() - started
        estimate = load_seconds + (elapsed - load_seconds) / count * len(records)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.font_manager import FontProperties

            target = directory / "plots"
            target.mkdir()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(
                faces[:, 2] * config["face_resource"]["age_scale"],
                voices[:, 0] * config["face_resource"]["age_scale"],
                s=4,
                alpha=0.25,
            )
            font = FontProperties(fname=config["plot_font"])
            ax.set_xlabel("冻结人脸模型预测年龄", fontproperties=font)
            ax.set_ylabel("冻结语音模型预测年龄", fontproperties=font)
            ax.set_title("仅官方训练样本；预测值不是真实年龄", fontproperties=font)
            fig.tight_layout()
            fig.savefig(target / "训练年龄预测.png", dpi=config["plot_dpi"])
            plt.close(fig)
        except Exception:
            LOGGER.exception("属性绘图失败，保留特征")
        return {
            "status": "complete",
            "mode": config["mode"],
            "rows": count,
            "full_training_rows": len(records),
            "load_seconds": load_seconds,
            "feature_seconds": elapsed,
            "estimated_full_seconds": estimate,
            "unique_audio": len(audio_cache),
            "unique_face": len(face_cache),
            "feature_sha256": digest(directory / "features.npz"),
            "repeat_error": repeat_error,
            "progress_evaluation_used": False,
            "audio_encoder_updated": False,
            "age_ground_truth_used": False,
        }
    except Exception:
        np.savez(
            directory / "partial_features.npz",
            face=faces,
            voice=voices,
            hidden=np.empty((0,)) if hidden is None else hidden,
            processed=np.array(processed),
        )
        raise


def main() -> int:
    config_path = Path(__file__).with_name("config.json")
    config = read(config_path)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("smoke", "profile", "full"), default=config["mode"]
    )
    config["mode"] = parser.parse_args().mode
    expected = Path(config["training_python"]).resolve().parent.parent
    require(
        Path(sys.prefix).resolve() == expected
        and Path(sys.executable).resolve().parent.parent == expected,
        "训练必须使用my_project_env",
    )
    config["run_id"] = (
        datetime.now().astimezone().strftime("%m-%d-%H%M%S")
        + "-"
        + config["mode"]
        + "-"
        + uuid.uuid4().hex[:6]
    )
    config["entrypoint"] = (
        f"{sys.executable} -m exps.{config['experiment_id']}.{Path(__file__).stem} --mode {config['mode']}"
    )
    directory = (
        ROOT / config["output_root"] / config["experiment_id"] / config["run_id"]
    )
    directory.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(directory / "run.log", encoding="utf-8"),
        ],
    )
    save(directory / "config.json", config)
    save(
        directory / "environment.json",
        {
            "executable": sys.executable,
            "prefix": sys.prefix,
            "packages": {
                p: importlib.metadata.version(p)
                for p in [
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "wandb",
                    "matplotlib",
                    "torch",
                    "transformers",
                    "insightface",
                    "onnxruntime",
                ]
            },
        },
    )
    snapshot = directory / "source"
    snapshot.mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, snapshot / source.name)
    shutil.copy2(config_path, snapshot / config_path.name)
    LOGGER.info(
        "阶段=启动 run=%s 预算=%ds", config["run_id"], config["compute_budget_seconds"]
    )
    torch.set_num_threads(config["cpu_threads"])
    require(torch.cuda.is_available(), "需要CUDA设备")
    total_memory = torch.cuda.get_device_properties(config["device"]).total_memory
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, config["gpu_memory_limit_gib"] * 1024**3 / total_memory),
        device=config["device"],
    )
    torch.backends.cuda.matmul.allow_tf32 = config["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = config["allow_tf32"]
    started = time.monotonic()
    history = History(config, directory)
    try:
        with threadpool_limits(config["cpu_threads"]):
            result = execute(
                config, directory, history, started + config["compute_budget_seconds"]
            )
        result["wall_seconds"] = time.monotonic() - started
        save(directory / "result.json", result)
        (directory / "exit_code.txt").write_text("0\n")
        LOGGER.info(
            "阶段=完成 结果=%s",
            json.dumps(
                {k: v for k, v in result.items() if k != "folds"}, ensure_ascii=False
            ),
        )
        return 0
    except Exception:
        LOGGER.exception("阶段=失败")
        (directory / "exit_code.txt").write_text("1\n")
        raise
    finally:
        history.finish()


if __name__ == "__main__":
    raise SystemExit(main())
