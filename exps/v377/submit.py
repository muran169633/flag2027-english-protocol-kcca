"""复用逐配对独立基线分数，加入训练身份固定的年龄一致性项。"""

from __future__ import annotations

import argparse
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
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from .traits import FrozenTraits

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


class Detector:
    def __init__(self, config: dict[str, Any]) -> None:
        import onnxruntime as ort
        from insightface.model_zoo.scrfd import SCRFD

        self.config = config
        spec = config["detector"]
        path = ROOT / spec["path"]
        require(file_hash(path) == spec["sha256"], "冻结人脸检测器指纹不符")
        options = ort.SessionOptions()
        options.intra_op_num_threads = config["cpu_threads"]
        options.inter_op_num_threads = config["face_resource"]["inter_threads"]
        options.log_severity_level = config["face_resource"]["log_severity"]
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=config["face_resource"]["providers"],
        )
        self.detector: Any = SCRFD(str(path), session=session)
        self.detector.prepare(
            0, input_size=spec["input_size"], det_thresh=spec["threshold"]
        )
        self.detector.nms_thresh = spec["nms_threshold"]

    def box(self, path: Path) -> list[float] | None:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取人脸图像：{path}")
        boxes, _ = self.detector.detect(image, max_num=0)
        if not len(boxes):
            return None
        height, width = image.shape[:2]
        center = (boxes[:, :2] + boxes[:, 2:4]) / 2
        distance = ((center - np.array([width / 2, height / 2])) ** 2).sum(1)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        chosen = int(
            np.argmax(area - self.config["detector"]["center_penalty"] * distance)
        )
        return boxes[chosen, :4].tolist()


def fixed_parameters(
    config: dict[str, Any], directory: Path
) -> dict[str, list[dict[str, float]]]:
    spec = config["submission"]
    source = ROOT / spec["age_validation_run"]
    for filename, expected in spec["age_files_sha256"].items():
        require(file_hash(source / filename) == expected, "固定年龄参数来源变化")
    result = read_json(source / "result.json")
    require(
        result["promotion_passed"] and not result["progress_evaluation_used"],
        "年龄方案未通过预先固定的训练验证门槛",
    )
    parameters: dict[str, list[dict[str, float]]] = {"no_gender": [], "gender": []}
    for fold in config["folds"]:
        model = read_json(source / f"fold_{fold}" / "age_model.json")
        require(len(set(model["train_identities"])) == 56, "年龄映射必须来自56训练身份")
        for track, protocol in (("no_gender", "standard"), ("gender", "same_gender")):
            parameters[track].append(
                {**model["mapping"], "weight": model["weights"][protocol]}
            )
    write_json(directory / "冻结年龄参数.json", parameters)
    return parameters


def age_distance(
    face: float, voice: float, parameters: list[dict[str, float]]
) -> float:
    return float(
        np.mean(
            [
                p["weight"] * (face - p["slope"] * voice - p["intercept"]) ** 2
                for p in parameters
            ]
        )
    )


def execute(config: dict[str, Any], directory: Path, deadline: float) -> dict[str, Any]:
    spec = config["submission"]
    parameters = fixed_parameters(config, directory)
    baseline = ROOT / spec["baseline_zip"]
    require(file_hash(baseline) == spec["baseline_sha256"], "逐trial独立基线包指纹变化")
    base_result = read_json(ROOT / spec["baseline_result"])
    require(
        base_result["sha256"] == spec["baseline_sha256"]
        and not base_result["cross_trial_scoring"]
        and not base_result["test_statistics_used_for_scoring"]
        and not base_result["posthoc_normalization"],
        "基线不满足独立评分条件",
    )
    require(
        file_hash(ROOT / spec["manifest"]) == spec["manifest_sha256"],
        "新阶段媒体清单变化",
    )
    model = FrozenTraits(config)
    detector = Detector(config)
    training_run = ROOT / config["trait_feature_run"]
    with np.load(training_run / "features.npz", allow_pickle=False) as cache:
        old_face, old_voice = cache["face"].copy(), cache["voice"].copy()
    training = read_json(training_run / "manifest.json")
    compatibility = []
    for index, row in enumerate(training[: config["independence_rows"]]):
        face = model.image(
            ROOT / row["face_path"], {"bbox": detector.box(ROOT / row["face_path"])}
        )
        voice, _ = model.voice(ROOT / row["voice_path"])
        error = float(
            max(
                np.max(np.abs(face - old_face[index])),
                np.max(np.abs(voice - old_voice[index])),
            )
        )
        require(
            error <= config["compatibility_tolerance"], "公开属性推断与训练特征不兼容"
        )
        compatibility.append(error)
    write_json(directory / "训练属性兼容性.json", compatibility)
    records = read_json(ROOT / spec["manifest"])
    expected: dict[str, list[str]] = {}
    raw_features = []
    independence = []
    with ZipFile(baseline) as archive:
        for track, language, condition, count in spec["cells"]:
            name = f"{track}/sub_score_v4_{language}_{condition}.txt"
            original = [
                line.split() for line in archive.read(name).decode().splitlines()
            ]
            rows = [
                r for r in records if r["track"] == track and r["language"] == language
            ]
            protocol = ROOT / spec["protocol_root"] / track / f"{language}_valid.txt"
            ids = [line.split()[0] for line in protocol.read_text().splitlines()]
            require(
                len(rows) == count
                and [r["pair_id"] for r in rows] == ids == [r[0] for r in original],
                "协议或基线编号覆盖不符",
            )
            if config["mode"] == "smoke":
                rows = rows[: spec["smoke_pairs_per_cell"]]
                original = original[: len(rows)]
            predicted = []
            age_outputs = []
            for index, (row, base) in enumerate(zip(rows, original, strict=True)):
                if time.monotonic() > deadline:
                    raise TimeoutError("独立年龄推断达到预算")
                face_path, voice_path = (
                    ROOT / row["face_path"],
                    ROOT / row["voice_path"],
                )
                require(
                    file_hash(face_path) == row["face_sha256"]
                    and file_hash(voice_path) == row["voice_sha256"],
                    "媒体内容指纹变化",
                )
                face = model.image(face_path, {"bbox": detector.box(face_path)})
                voice, _ = model.voice(voice_path)
                value = float(base[1]) + age_distance(
                    float(face[2]), float(voice[0]), parameters[track]
                )
                require(np.isfinite(value), "距离分数非有限")
                predicted.append(value)
                age_outputs.append(np.concatenate([face, voice]))
                if (index + 1) % config["data"]["log_every_rows"] == 0:
                    LOGGER.info(
                        "阶段=独立评分 track=%s language=%s rows=%d/%d",
                        track,
                        language,
                        index + 1,
                        len(rows),
                    )
            repeated_face = model.image(
                ROOT / rows[0]["face_path"],
                {"bbox": detector.box(ROOT / rows[0]["face_path"])},
            )
            repeated_voice, _ = model.voice(ROOT / rows[0]["voice_path"])
            error = float(
                np.max(
                    np.abs(
                        np.concatenate([repeated_face, repeated_voice]) - age_outputs[0]
                    )
                )
            )
            require(
                error <= config["compatibility_tolerance"],
                "处理其他trial后属性输出改变",
            )
            independence.append(
                {"track": track, "language": language, "repeat_error": error}
            )
            output = directory / "scores" / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                "".join(
                    f"{row['pair_id']} {value:.{spec['score_decimals']}f}\n"
                    for row, value in zip(rows, predicted, strict=True)
                )
            )
            expected[name] = [r["pair_id"] for r in rows]
            raw_features.extend(age_outputs)
            LOGGER.info(
                "阶段=协议完成 track=%s language=%s rows=%d", track, language, len(rows)
            )
    write_json(directory / "独立性核验.json", independence)
    np.save(directory / "frozen_attributes.npy", np.stack(raw_features))
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
                len(ids) == len(set(ids))
                and [line[0] for line in parsed] == ids
                and all(
                    len(line) == 2 and np.isfinite(float(line[1])) for line in parsed
                ),
                "ZIP编号或分数格式错误",
            )
    result = {
        "status": "complete",
        "mode": config["mode"],
        "rows": sum(map(len, expected.values())),
        "cells": {name: len(ids) for name, ids in expected.items()},
        "zip": str(archive_path.relative_to(ROOT)),
        "sha256": file_hash(archive_path),
        "bytes": archive_path.stat().st_size,
        "score_direction": "lower_is_match",
        "evaluation_eer": None,
        "training_performed": False,
        "test_statistics_used_for_scoring": False,
        "cross_trial_scoring": False,
        "posthoc_normalization": False,
        "baseline_source": spec["baseline_zip"],
        "age_parameters_source": spec["age_validation_run"],
    }
    if config["mode"] == "full":
        delivery = ROOT / spec["delivery_directory"] / filename
        require(not delivery.exists(), "不覆盖已有交付包")
        shutil.copy2(archive_path, delivery)
        require(file_hash(delivery) == result["sha256"], "交付文件复制指纹不同")
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
    directory = ROOT / config["output_root"] / config["experiment_id"] / run_id
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
        config["submission"]["maximum_seconds"],
    )
    started = time.monotonic()
    try:
        result = execute(
            config, directory, started + config["submission"]["maximum_seconds"]
        )
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
