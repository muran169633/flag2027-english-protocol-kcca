"""官方训练身份验证选模，逐配对独立推断并生成新阶段距离分数。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F

LOGGER = logging.getLogger("独立推断")
ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class Projection(nn.Module):
    """结构完全由冻结训练配置确定；推断只使用两条投影分支。"""

    def __init__(self, config: dict[str, Any], classes: int) -> None:
        super().__init__()
        self.epsilon = config["normalization_epsilon"]
        dimension = config["fixed_embedding_dimension"]

        def branch(input_dimension: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dimension, dimension),
                nn.BatchNorm1d(
                    dimension,
                    eps=config["batchnorm_epsilon"],
                    momentum=config["batchnorm_momentum"],
                ),
                nn.ReLU()
                if config.get("projection_activation", "relu") == "relu"
                else nn.Identity(),
                nn.Dropout(config["dropout"]),
            )

        self.face = branch(config["face_dimension"])
        self.voice = branch(config["voice_dimension"])
        self.fusion_weights = nn.Parameter(torch.zeros(2))
        self.classifier = nn.Linear(dimension, classes)

    def forward(self, faces: Tensor, voices: Tensor) -> Tensor:
        face = F.normalize(self.face(faces), dim=-1, eps=self.epsilon)
        voice = F.normalize(self.voice(voices), dim=-1, eps=self.epsilon)
        return (face * voice).sum(-1)  # [B, D] -> [B]，只比较同行


@torch.inference_mode()
def similarities(
    model: Projection,
    faces: NDArray[np.float32],
    voices: NDArray[np.float32],
    config: dict[str, Any],
) -> NDArray[np.float32]:
    batches = []
    for start in range(0, len(faces), config["head_batch_size"]):
        stop = start + config["head_batch_size"]
        batches.append(
            model(
                torch.from_numpy(faces[start:stop].copy()).to(config["device"]),
                torch.from_numpy(voices[start:stop].copy()).to(config["device"]),
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(batches)


def audit_selection(config: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for reference in config["selection_candidates"]:
        path = ROOT / reference["run"]
        for filename in ("config", "result"):
            require(
                file_hash(path / f"{filename}.json") == reference[f"{filename}_sha256"],
                f"候选来源指纹变化：{path}/{filename}.json",
            )
        result = read_json(path / "result.json")
        provenance = read_json(path / "data_sources.json")
        require(
            result["status"] == "complete"
            and result["mode"] == "full"
            and result["folds_completed"] == 5
            and provenance["source"] == "official_v4_only"
            and provenance["history_training_used"] is False,
            f"候选不是已完成的官方训练身份五折：{path}",
        )
        candidates.append({"run": reference["run"], "metrics": result["metrics"]})
    selected = {}
    for track, source in config["sources"].items():
        metric = source["validation_metric"]
        best = min(candidates, key=lambda x: x["metrics"][metric]["fold_mean"])
        require(best["run"] == source["run"], "固定来源不是候选内最低训练验证值")
        selected[track] = {
            "run": best["run"],
            "metric": metric,
            "fold_mean": best["metrics"][metric]["fold_mean"],
        }
    return {
        "candidates": candidates,
        "selected": selected,
        "selected_english_proxy_mean": sum(
            entry["fold_mean"] for entry in selected.values()
        )
        / len(selected),
        "scope": "既有候选的英语训练身份隔离CV，选择后代理；非独立外层CV或官方四项成绩",
        "evaluation_scores_used": False,
    }


def load_models(
    config: dict[str, Any], directory: Path
) -> tuple[dict[str, list[Projection]], list[dict[str, Any]]]:
    models: dict[str, list[Projection]] = {}
    audit = []
    for track, source in config["sources"].items():
        path = ROOT / source["run"]
        trained = read_json(path / "config.json")
        result = read_json(path / "result.json")
        provenance = read_json(path / "data_sources.json")
        cache = ROOT / trained["feature_cache"]
        cache_manifest = read_json(cache / "manifest.json")
        for name, expected in provenance["input_hashes"].items():
            require(file_hash(ROOT / trained[name]) == expected, "训练输入指纹变化")
        require(
            file_hash(cache / "face.npy") == cache_manifest["array_hashes"]["face"],
            "训练人脸缓存指纹变化",
        )
        voice_path = ROOT / trained["voice_feature_run"] / "features.npy"
        require(
            file_hash(voice_path) == provenance["voice_feature_sha256"],
            "训练语音缓存指纹变化",
        )
        faces = np.load(cache / "face.npy", allow_pickle=False)
        voices = np.load(voice_path, allow_pickle=False)
        train_ids = {
            line.split()[4]
            for line in (ROOT / trained["training_list"]).read_text().splitlines()
        }
        held_ids: list[str] = []
        models[track] = []
        for summary in result["folds"]:
            fold_path = path / f"fold_{summary['fold']}"
            split = read_json(fold_path / "split.json")
            train, valid = (
                set(split["train_speakers"]),
                set(split["validation_speakers"]),
            )
            require(not train & valid and train | valid == train_ids, "身份划分不隔离")
            held_ids.extend(valid)
            checkpoint_path = fold_path / "best.pt"
            require(
                file_hash(checkpoint_path) == summary["checkpoint_sha256"],
                "匹配头权重指纹变化",
            )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            model = Projection(trained, checkpoint["num_classes"])
            model.load_state_dict(checkpoint["model"], strict=True)
            model = model.to(config["device"]).requires_grad_(False).eval()
            with np.load(fold_path / "oof_scores.npz", allow_pickle=False) as saved:
                metric = source["validation_metric"]
                pairs = saved[f"{metric}_pairs"][
                    : config["compatibility_pairs_per_fold"]
                ]
                predicted = similarities(
                    model, faces[pairs[:, 0]], voices[pairs[:, 1]], config
                )
                error = float(
                    np.max(np.abs(predicted - saved[f"{metric}_scores"][: len(pairs)]))
                )
            require(error <= config["score_tolerance"], "新实现未复现原验证预测")
            audit.append(
                {
                    "track": track,
                    "fold": summary["fold"],
                    "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                    "sha256": summary["checkpoint_sha256"],
                    "compatibility_rows": len(pairs),
                    "max_abs_error": error,
                    "train_speakers": len(train),
                    "validation_speakers": len(valid),
                }
            )
            models[track].append(model)
        require(
            len(held_ids) == len(set(held_ids)) == len(train_ids), "五折未恰好覆盖身份"
        )
        snapshot = directory / "training_sources" / trained["experiment_id"]
        snapshot.mkdir(parents=True)
        source_code = ROOT / "exps" / trained["experiment_id"]
        for file in source_code.glob("*.py"):
            shutil.copy2(file, snapshot / file.name)
        for filename in (
            "config.json",
            "result.json",
            "data_sources.json",
            "folds.json",
        ):
            shutil.copy2(path / filename, snapshot / filename)
    return models, audit


class FrozenVoice:
    def __init__(self, config: dict[str, Any]) -> None:
        os.environ["USE_TF"] = "0"
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

        self.config = config
        self.spec = config["voice_encoder"]
        self.path = ROOT / self.spec["local_directory"]
        verified = read_json(self.path / "verified.json")
        require(
            verified["revision"] == self.spec["revision"]
            and verified["repository"] == self.spec["repository"],
            "公开语音编码器版本不符",
        )
        for filename, item in verified["files"].items():
            require(
                file_hash(self.path / filename) == item["sha256"], "公开编码器指纹错误"
            )
        self.processor: Any = AutoFeatureExtractor.from_pretrained(
            str(self.path), local_files_only=True, trust_remote_code=False
        )
        self.model: Any = (
            AutoModelForAudioXVector.from_pretrained(
                str(self.path),
                local_files_only=True,
                trust_remote_code=False,
                weights_only=True,
            )
            .to(config["device"])
            .requires_grad_(False)
            .eval()
        )

    @torch.inference_mode()
    def encode(self, path: Path) -> NDArray[np.float32]:
        import librosa

        wave, rate = sf.read(path, dtype="float32", always_2d=True)
        values = wave.mean(1)
        if rate != self.spec["sample_rate"]:
            values = librosa.resample(
                values, orig_sr=rate, target_sr=self.spec["sample_rate"]
            )
        length = int(self.spec["sample_rate"] * self.spec["maximum_seconds"])
        if len(values) > length:
            start = (len(values) - length) // 2
            values = values[start : start + length]
        require(
            len(values) >= self.spec["minimum_samples"]
            and bool(np.isfinite(values).all()),
            "音频无效",
        )
        inputs = self.processor(
            [values],
            sampling_rate=self.spec["sample_rate"],
            padding="max_length",
            max_length=length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        tensors = {
            name: value.to(self.config["device"]) for name, value in inputs.items()
        }
        with torch.autocast(
            device_type=self.config["device"].split(":")[0],
            dtype=torch.bfloat16,
            enabled=self.spec["precision"] == "bfloat16",
        ):
            output = self.model(**tensors, output_hidden_states=False)
        embedding = F.normalize(
            output.embeddings.float(), dim=-1, eps=self.spec["normalization_epsilon"]
        )
        require(
            tuple(embedding.shape) == (1, self.spec["dimension"]), "语音特征维度错误"
        )
        require(bool(torch.isfinite(embedding).all()), "语音特征包含非有限值")
        return embedding.cpu().numpy()[0]


def execute(config: dict[str, Any], directory: Path, deadline: float) -> dict[str, Any]:
    selection = audit_selection(config)
    write_json(directory / "选模依据.json", selection)
    LOGGER.info(
        "阶段=选模 候选=%d 英语选择后代理=%.6f",
        len(selection["candidates"]),
        selection["selected_english_proxy_mean"],
    )
    models, assets = load_models(config, directory)
    write_json(directory / "权重与兼容性.json", assets)
    encoder = FrozenVoice(config)
    source = ROOT / next(iter(config["sources"].values()))["run"]
    trained = read_json(source / "config.json")
    voice_root = ROOT / trained["voice_feature_run"]
    old = np.load(voice_root / "features.npy", allow_pickle=False)
    training_manifest = read_json(voice_root / "manifest.json")
    errors = []
    for i, row in enumerate(
        training_manifest["rows"][
            : config["voice_encoder"]["training_compatibility_samples"]
        ]
    ):
        errors.append(float(np.max(np.abs(encoder.encode(Path(row["path"])) - old[i]))))
    require(
        max(errors) <= config["voice_encoder"]["training_compatibility_max_abs"],
        "语音预处理与训练来源不兼容",
    )
    write_json(
        directory / "语音兼容性.json",
        {
            "rows": len(errors),
            "errors": errors,
            "note": "训练来源batch=4；当前始终逐条batch=1，允许浮点运算差异",
        },
    )
    records = read_json(ROOT / config["manifest"])
    expected = {}
    independence = []
    all_voices = []
    for track, language, condition, count in config["cells"]:
        rows = [r for r in records if r["track"] == track and r["language"] == language]
        require(len(rows) == count, "新阶段协议行数错误")
        protocol = ROOT / config["protocol_root"] / track / f"{language}_valid.txt"
        require(
            [r["pair_id"] for r in rows]
            == [s.split()[0] for s in protocol.read_text().splitlines()],
            "配对编号行序与官方清单不符",
        )
        csv_path = (
            ROOT
            / config["protocol_root"]
            / track
            / "features"
            / f"{language}_valid_faces.csv"
        )
        faces = np.loadtxt(csv_path, delimiter=",", dtype=np.float32, ndmin=2)
        require(
            faces.shape == (count, trained["face_dimension"])
            and bool(np.isfinite(faces).all()),
            "官方人脸特征错误",
        )
        if config["mode"] == "smoke":
            rows = rows[: config["smoke_pairs_per_cell"]]
            faces = faces[: len(rows)]
        voices = []
        for index, row in enumerate(rows):
            if time.monotonic() >= deadline:
                raise TimeoutError("独立推断达到配置预算")
            voice_path = ROOT / row["voice_path"]
            require(file_hash(voice_path) == row["voice_sha256"], "音频指纹变化")
            voices.append(encoder.encode(voice_path))
            if (index + 1) % config["voice_encoder"]["log_every_rows"] == 0:
                LOGGER.info(
                    "阶段=逐条编码 协议=%s/%s 完成=%d/%d",
                    track,
                    language,
                    index + 1,
                    len(rows),
                )
        voice_matrix = np.stack(voices)
        all_voices.append(voice_matrix)
        per_model = np.stack(
            [
                similarities(model, faces, voice_matrix, config)
                for model in models[track]
            ]
        )
        distances = 1 - per_model.astype(np.float64).mean(0)
        n = min(config["verification_rows"], len(rows))
        for model in models[track]:
            baseline = similarities(model, faces[:n], voice_matrix[:n], config)
            singles = np.concatenate(
                [
                    similarities(
                        model, faces[i : i + 1], voice_matrix[i : i + 1], config
                    )
                    for i in range(n)
                ]
            )
            reverse = similarities(
                model, faces[:n][::-1], voice_matrix[:n][::-1], config
            )[::-1]
            error = float(
                max(
                    np.max(np.abs(baseline - singles)),
                    np.max(np.abs(baseline - reverse)),
                )
            )
            require(
                error <= config["independence_test_atol"], "单条/批量/重排预测不一致"
            )
            independence.append(
                {"track": track, "language": language, "rows": n, "max_error": error}
            )
        # 音频从头重复编码一条，确认输出不依赖先前处理的其他trial。
        repeated = encoder.encode(ROOT / rows[0]["voice_path"])
        require(bool(np.array_equal(repeated, voice_matrix[0])), "逐条音频重算不一致")
        name = f"{track}/sub_score_v4_{language}_{condition}.txt"
        output = directory / "scores" / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(
                f"{row['pair_id']} {value:.{config['score_decimals']}f}\n"
                for row, value in zip(rows, distances, strict=True)
            )
        )
        expected[name] = [r["pair_id"] for r in rows]
        LOGGER.info("阶段=分数完成 协议=%s/%s 行数=%d", track, language, len(rows))
    np.save(directory / "voice_features.npy", np.concatenate(all_voices))
    write_json(directory / "独立性核验.json", independence)
    filename = f"submission_evaluation_{config['experiment_id']}.zip"
    archive_path = directory / filename
    with ZipFile(archive_path, "x", ZIP_DEFLATED) as archive:
        for name in expected:
            archive.write(directory / "scores" / name, name)
    with ZipFile(archive_path) as archive:
        require(
            set(archive.namelist()) == set(expected) and archive.testzip() is None,
            "ZIP结构或CRC错误",
        )
        for name, ids in expected.items():
            parsed = [line.split() for line in archive.read(name).decode().splitlines()]
            require(
                len(ids) == len(set(ids)) and [r[0] for r in parsed] == ids,
                "ZIP编号覆盖错误",
            )
            require(
                all(len(r) == 2 and np.isfinite(float(r[1])) for r in parsed),
                "ZIP分数格式错误",
            )
    result = {
        "status": "complete",
        "mode": config["mode"],
        "rows": sum(map(len, expected.values())),
        "cells": {k: len(v) for k, v in expected.items()},
        "zip": str(archive_path.relative_to(ROOT)),
        "sha256": file_hash(archive_path),
        "bytes": archive_path.stat().st_size,
        "score_direction": "lower_is_match",
        "evaluation_eer": None,
        "training_performed": False,
        "test_statistics_used_for_scoring": False,
        "cross_trial_scoring": False,
        "posthoc_normalization": False,
        "weights": "每协议五个冻结头，各1/5，推断前固定",
        "compliance_review": "按用户提供FAQ实现；不代表主办方已经审核",
    }
    if config["mode"] == "full":
        delivery = ROOT / config["delivery_directory"] / filename
        require(not delivery.exists(), "不覆盖已有交付包")
        shutil.copy2(archive_path, delivery)
        require(file_hash(delivery) == result["sha256"], "交付包复制后指纹不同")
        result["delivery"] = str(delivery.relative_to(ROOT))
    return result


def main() -> int:
    config_path = Path(__file__).with_name("config.json")
    config = read_json(config_path)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default=config["mode"])
    config["mode"] = parser.parse_args().mode
    environment = Path(config["training_python"]).parent.parent.resolve()
    require(
        Path(sys.prefix).resolve() == environment
        and Path(sys.executable).parent.parent.resolve() == environment,
        "必须使用my_project_env解释器",
    )
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = config["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = config["allow_tf32"]
    torch.cuda.set_per_process_memory_fraction(
        min(
            1.0,
            config["gpu_memory_limit_gib"]
            * 1024**3
            / torch.cuda.get_device_properties(config["device"]).total_memory,
        ),
        config["device"],
    )
    run_id = (
        datetime.now().astimezone().strftime("%m-%d-%H%M%S")
        + "-"
        + config["mode"]
        + "-"
        + uuid.uuid4().hex[:6]
    )
    directory = ROOT / config["output_directory"] / config["experiment_id"] / run_id
    directory.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(directory / "run.log", encoding="utf-8"),
        ],
    )
    config["run_id"] = run_id
    config["entrypoint"] = (
        f"{sys.executable} -m exps.{config['experiment_id']}.submit --mode {config['mode']}"
    )
    write_json(directory / "config.json", config)
    write_json(
        directory / "environment.json",
        {
            "executable": sys.executable,
            "prefix": sys.prefix,
            "python": sys.version,
            "packages": {
                p: importlib.metadata.version(p)
                for p in ["torch", "transformers", "numpy", "soundfile", "librosa"]
            },
        },
    )
    source_snapshot = directory / "source"
    source_snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, source_snapshot / path.name)
    shutil.copy2(config_path, source_snapshot / config_path.name)
    LOGGER.info(
        "阶段=启动 run=%s 解释器=%s 预算=%ds",
        run_id,
        sys.executable,
        config["compute_budget_seconds"],
    )
    started = time.monotonic()
    try:
        result = execute(config, directory, started + config["compute_budget_seconds"])
        result["wall_seconds"] = time.monotonic() - started
        write_json(directory / "result.json", result)
        (directory / "exit_code.txt").write_text("0\n")
        LOGGER.info("阶段=完成 结果=%s", json.dumps(result, ensure_ascii=False))
        return 0
    except Exception:
        LOGGER.exception("阶段=失败")
        (directory / "exit_code.txt").write_text("1\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
