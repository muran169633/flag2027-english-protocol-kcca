"""生成四协议 Progress 开发集的冻结图距离分数 ZIP，不上传平台。"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .audio import extract
from .common import digest, save, start_run
from .face import FaceEncoder
from .graph import Array, infer, normalize

logger = logging.getLogger(__name__)


def load_records(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Array]]:
    entity_cache = json.loads(Path(cfg["entities_manifest"]).read_text())
    entities = {
        (r["track"], r["language"], r["pair_id"]): r
        for r in entity_cache["rows"]
        if r["split"] == "dev"
    }
    records, queries = [], {}
    for cell in cfg["cells"]:
        base = Path(cfg["development_root"]) / cell["track"]
        text = base / (cell["language"] + "_test.txt")
        if digest(text) != entity_cache["source_hashes"][str(text)]:
            raise ValueError("配对列表与实体缓存不同步")
        lines = [line.split() for line in text.read_text().splitlines()]
        if len(lines) != cell["rows"] or any(len(line) != 3 for line in lines):
            raise ValueError("正式配对列表行数或列数错误")
        if len({line[0] for line in lines}) != len(lines):
            raise ValueError("协议内出现重复配对编号")
        indices = []
        for identifier, voice, face in lines:
            cached = entities[cell["track"], cell["language"], identifier]
            row = {
                "pair_id": identifier,
                "track": cell["track"],
                "language": cell["language"],
                "face_path": str(base / face),
                "voice_path": str(base / voice),
            }
            for modality in ["face", "voice"]:
                path = Path(row[modality + "_path"])
                if not path.resolve().is_relative_to(base.resolve()) or path != Path(
                    cached[modality + "_path"]
                ):
                    raise ValueError("媒体路径与官方实体清单不同")
                row[modality + "_sha256"] = digest(path)
                if row[modality + "_sha256"] != cached[modality + "_sha256"]:
                    raise ValueError("正式媒体内容改变")
            indices.append(len(records))
            records.append(row)
        queries[cell["track"] + "/" + cell["language"]] = np.c_[indices, indices]
    return records, queries


def cached_faces(
    records: list[dict[str, Any]], cfg: dict[str, Any], output: Path
) -> Array:
    root = Path(cfg["face_cache_run"])
    source_cfg = json.loads((root / "config.json").read_text())
    for key, old in [
        ("recognition_file", "recognition_file"),
        ("detection_file", "detection_file"),
        ("detection_size", "detection_size"),
        ("detection_threshold", "detection_threshold"),
        ("nms_threshold", "nms_threshold"),
        ("center_penalty", "selection_center_penalty"),
        ("recognition_size", "recognition_size"),
        ("onnx_providers", "providers"),
    ]:
        if cfg[key] != source_cfg[old]:
            raise ValueError("人脸缓存预处理不兼容: " + key)
    if (
        Path(cfg["face_weights_root"]).resolve()
        != Path(source_cfg["weights_directory"]).resolve()
    ):
        raise ValueError("人脸缓存模型目录不一致")
    manifest = json.loads((root / "manifest.json").read_text())
    result = json.loads((root / "result.json").read_text())
    if (
        not result["full_development_features"]
        or digest(root / "features.npy") != result["feature_sha256"]
    ):
        raise ValueError("人脸缓存未完整通过指纹校验")
    if manifest["pretrained_manifest_sha256"] != digest(
        Path(cfg["face_weights_root"]) / "verified.json"
    ):
        raise ValueError("人脸缓存权重来源不同")
    lookup = {
        (r["track"], r["language"], r["pair_id"]): i
        for i, r in enumerate(manifest["rows"])
    }
    values = np.load(root / "features.npy")
    indices = []
    for row in records:
        index = lookup[row["track"], row["language"], row["pair_id"]]
        old = manifest["rows"][index]
        path = Path(row["face_path"])
        if (
            Path(old["path"]).resolve() != path.resolve()
            or old["bytes"] != path.stat().st_size
            or old["mtime_ns"] != path.stat().st_mtime_ns
        ):
            raise ValueError("人脸缓存原始媒体状态不同")
        indices.append(index)
    pooled = normalize(values[indices])
    selected = set(
        np.linspace(0, len(records) - 1, cfg["face_check_rows"], dtype=int).tolist()
    )
    for line in (root / "alignments.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["method"] != "five_point" and row["index"] in indices:
            selected.add(indices.index(row["index"]))
    check = sorted(selected)
    fresh, _ = FaceEncoder(cfg).encode([records[i] for i in check])
    difference = float(np.max(np.abs(fresh - pooled[check])))
    if difference > cfg["face_check_tolerance"]:
        raise ValueError("独立人脸实现未复现缓存: " + str(difference))
    save(
        output / "face_cache_verification.json",
        {
            "source": str(root),
            "rows": len(records),
            "recomputed_rows": check,
            "max_absolute_difference": difference,
            "preprocessing_compatible": True,
        },
    )
    return pooled


def representations(
    ecapa: Array, eres: Array, cfg: dict[str, Any]
) -> tuple[dict[str, Array], dict[str, float]]:
    root = Path(cfg["calibration_run"])
    with np.load(root / "wccn.npz") as weights:
        ec, er = normalize(ecapa @ weights["ecapa"]), normalize(eres @ weights["eres"])
    thresholds = json.loads((root / "calibration.json").read_text())["thresholds"]
    return {
        "ecapa_wccn": ec,
        "eres_wccn": er,
        "equal_concat": normalize(np.c_[ec, er]),
    }, thresholds


def graph_replay(cfg: dict[str, Any]) -> float:
    root = Path(cfg["graph_reference_run"])
    records = json.loads((root / "manifest.json").read_text())
    with np.load(root / "features.npz") as features:
        face, ec, er = (
            normalize(features["face"]),
            normalize(features["ecapa"]),
            normalize(features["eres"]),
        )
    reps, thresholds = representations(ec, er, cfg)
    with np.load(root / "new_trial_0_positive_10.npz") as reference:
        cells = [
            key.removesuffix("_pairs")
            for key in reference.files
            if key.endswith("_pairs")
        ]
        queries = {key: reference[key + "_pairs"][:, :2] for key in cells}
        started = time.monotonic()
        scores, _, _ = infer(
            queries,
            face,
            reps,
            np.array([r["face_sha256"] for r in records]),
            np.array([r["voice_sha256"] for r in records]),
            thresholds,
            cfg,
        )
        elapsed = time.monotonic() - started
        for cell in cells:
            if not np.allclose(
                scores[cell], reference[cell + "_max3"], rtol=0, atol=1e-12
            ):
                raise AssertionError("独立提交图实现未复现冻结历史基线")
    return elapsed


def pack(
    records: list[dict[str, Any]],
    queries: dict[str, Array],
    scores: dict[str, Array],
    cfg: dict[str, Any],
    output: Path,
) -> Path:
    if cfg["score_direction"] != "distance_lower_is_match":
        raise ValueError("提交必须使用距离方向")
    archive_path = output / (
        "submission.zip" if cfg["mode"] == "all" else "smoke_submission.zip"
    )
    stats: list[dict[str, Any]] = []
    with ZipFile(
        archive_path,
        "w",
        compression=ZIP_DEFLATED,
        compresslevel=cfg["compression_level"],
    ) as archive:
        for cell in cfg["cells"]:
            key = cell["track"] + "/" + cell["language"]
            ids = [records[i]["pair_id"] for i in queries[key][:, 0]]
            values = -np.asarray(scores[key])
            if len(values) != len(ids) or not np.isfinite(values).all():
                raise ValueError("提交分数长度或数值错误")
            if cfg["mode"] == "all" and (
                len(values) != cell["rows"] or len(np.unique(values)) < 2
            ):
                raise ValueError("正式提交行数不足或全部为常数")
            name = f"{cell['track']}/sub_score_v4_{cell['language']}_{cell['condition']}.txt"
            text = "".join(
                f"{identifier} {value:.{cfg['score_decimals']}f}\n"
                for identifier, value in zip(ids, values, strict=True)
            )
            archive.writestr(name, text)
            stats.append(
                {
                    "file": name,
                    "rows": len(ids),
                    "unique_scores": len(np.unique(values)),
                    "distance_min": float(values.min()),
                    "distance_max": float(values.max()),
                }
            )
    with ZipFile(archive_path) as archive:
        if (
            set(archive.namelist()) != {r["file"] for r in stats}
            or archive.testzip() is not None
        ):
            raise ValueError("ZIP 成员或 CRC 检查失败")
        for cell, row in zip(cfg["cells"], stats, strict=True):
            lines = [
                line.split() for line in archive.read(row["file"]).decode().splitlines()
            ]
            key = cell["track"] + "/" + cell["language"]
            expected_ids = [records[i]["pair_id"] for i in queries[key][:, 0]]
            if (
                any(len(line) != 2 for line in lines)
                or [line[0] for line in lines] != expected_ids
            ):
                raise ValueError("ZIP 回读编号或列数错误")
            if not np.allclose(
                np.array([float(line[1]) for line in lines]),
                -scores[key],
                rtol=0,
                atol=10 ** (-cfg["score_decimals"]),
            ):
                raise ValueError("ZIP 回读距离方向错误")
    save(
        output / "zip_verification.json",
        {
            "sha256": digest(archive_path),
            "files": stats,
            "total_rows": sum(r["rows"] for r in stats),
            "score_direction": cfg["score_direction"],
            "crc_passed": True,
            "uploaded": False,
        },
    )
    return archive_path


def run(cfg: dict[str, Any], output: Path) -> None:
    started = time.monotonic()
    records, queries = load_records(cfg)
    total = len(records)
    if cfg["mode"] == "smoke":
        keep = np.concatenate(
            [q[: cfg["smoke_rows_per_cell"], 0] for q in queries.values()]
        )
        records = [records[i] for i in keep]
        queries = {
            cell["track"] + "/" + cell["language"]: np.c_[
                np.arange(
                    i * cfg["smoke_rows_per_cell"], (i + 1) * cfg["smoke_rows_per_cell"]
                ),
                np.arange(
                    i * cfg["smoke_rows_per_cell"], (i + 1) * cfg["smoke_rows_per_cell"]
                ),
            ]
            for i, cell in enumerate(cfg["cells"])
        }
    save(output / "manifest.json", records)
    face = cached_faces(records, cfg, output)
    voices = {r["voice_sha256"]: r for r in records}
    unique = list(voices.values())
    full_unique = len(unique)
    if cfg["mode"] == "profile":
        unique = unique[: cfg["voice_profile_rows"]]
    audio_output = output / "voice_features"
    audio_output.mkdir()
    ec, er = extract(unique, cfg, audio_output, cfg["mode"])
    if cfg["mode"] == "profile":
        extraction = json.loads((audio_output / "extraction.json").read_text())
        graph_seconds = graph_replay(cfg)
        scale = (total / 280) ** 2
        estimate = (
            extraction["seconds"] * full_unique / len(unique) * cfg["profile_margin"]
            + graph_seconds * scale * cfg["profile_margin"]
            + cfg["profile_extra_seconds"]
        )
        save(
            output / "result.json",
            {
                "mode": cfg["mode"],
                "seconds": time.monotonic() - started,
                "voice_profile_rows": len(unique),
                "unique_voices": full_unique,
                "historical_graph_replay_seconds": graph_seconds,
                "conservative_full_seconds": estimate,
            },
        )
        return
    lookup = {r["voice_sha256"]: i for i, r in enumerate(unique)}
    ecapa = np.stack([ec[lookup[r["voice_sha256"]]] for r in records])
    eres = np.stack([er[lookup[r["voice_sha256"]]] for r in records])
    np.savez_compressed(output / "features.npz", face=face, ecapa=ecapa, eres=eres)
    reps, thresholds = representations(ecapa, eres, cfg)
    logger.info(
        "开始四协议联合完整图推断，配对=%s 唯一语音=%s", len(records), full_unique
    )
    scores, diagnostics, storage = infer(
        queries,
        face,
        reps,
        np.array([r["face_sha256"] for r in records]),
        np.array([r["voice_sha256"] for r in records]),
        thresholds,
        cfg,
    )
    saved: dict[str, Any] = dict(storage)
    for key in queries:
        saved[key.replace("/", "_") + "_pairs"] = queries[key]
        saved[key.replace("/", "_") + "_similarity"] = scores[key]
    np.savez_compressed(output / "predictions.npz", **saved)
    save(output / "graph_diagnostics.json", diagnostics)
    archive = pack(records, queries, scores, cfg, output)
    elapsed = time.monotonic() - started
    if elapsed > cfg["time_limit_seconds"]:
        raise TimeoutError("提交作业超过总预算，未交付到根目录")
    result: dict[str, Any] = {
        "mode": cfg["mode"],
        "rows": len(records),
        "seconds": elapsed,
        "archive": str(archive),
        "sha256": digest(archive),
        "official_eer": None,
        "uploaded": False,
        "method": "frozen_full_loo_max3",
    }
    if cfg["mode"] == "all":
        destination = Path(cfg["delivery_path"])
        if destination.exists():
            backup = destination.with_name(
                cfg["previous_archive_prefix"] + "_" + output.name + ".zip"
            )
            shutil.copy2(destination, backup)
            result["previous_archive_backup"] = str(backup)
            result["previous_archive_sha256"] = digest(backup)
        shutil.copy2(archive, destination)
        shutil.copy2(archive, cfg["versioned_delivery_path"])
        if digest(destination) != digest(archive):
            raise AssertionError("交付副本与归档不同")
        result["delivery"] = str(destination)
    save(output / "result.json", result)
    logger.info("完成 ZIP=%s 耗时=%.1f 秒", archive, elapsed)


def main() -> None:
    cfg = json.loads(Path(__file__).with_name("config.json").read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["smoke", "profile", "all"], default=cfg["default_mode"]
    )
    cfg["mode"] = parser.parse_args().mode
    if sys.prefix != "/home/xxc/miniconda3/envs/my_project_env":
        raise RuntimeError("必须使用指定 my_project_env")
    torch.set_num_threads(cfg["cpu_threads"])
    output = start_run(cfg, "frozen-graph-submission")
    paths = list(Path(__file__).parent.glob("*.py")) + [
        Path(cfg["entities_manifest"]),
        Path(cfg["platform_snapshot"]),
        Path(cfg["official_evaluator"]),
    ]
    paths += [
        Path(cfg["face_cache_run"]) / n
        for n in ["manifest.json", "config.json", "features.npy", "result.json"]
    ]
    paths += [
        Path(cfg["calibration_run"]) / n for n in ["calibration.json", "wccn.npz"]
    ]
    paths += [
        Path(cfg["weights_root"]) / cfg["weights_file"],
        Path(cfg["ecapa_weights"]),
    ]
    save(output / "inputs.json", {str(p): digest(p) for p in paths})
    save(
        output / "protocol_lock.json",
        {
            "config_sha256": digest(output / "config.json"),
            "model_locked_before_dev_prediction": True,
            "hidden_labels_used": False,
            "automatic_upload": False,
            "candidate": "frozen_full_loo_max3",
        },
    )
    logger.info("开始 mode=%s 产物=%s", cfg["mode"], output)
    with threadpool_limits(limits=cfg["cpu_threads"]):
        run(cfg, output)


if __name__ == "__main__":
    main()
