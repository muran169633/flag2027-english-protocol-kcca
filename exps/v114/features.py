"""冻结AdaFace特征缓存，保持SHA实体映射与中断恢复。"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch

from .common import Array, digest, save
from .encoder import Encoder

logger = logging.getLogger(__name__)


def collect(
    cfg: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Array], list[Path]]:
    entities, mapping, paths = [], {}, []
    index: dict[str, int] = {}
    for source in cfg["sources"]:
        path = Path(source["manifest"])
        records = json.loads(path.read_text())
        lookup = []
        for record in records:
            sha = record["face_sha256"]
            if sha not in index:
                if digest(Path(record["face_path"])) != sha:
                    raise AssertionError("输入人脸文件内容改变")
                index[sha] = len(entities)
                entities.append({"face_path": record["face_path"], "face_sha256": sha})
            lookup.append(index[sha])
        mapping[source["name"]] = np.array(lookup, dtype=int)
        paths.append(path)
    return entities, mapping, paths


def specification(cfg: dict[str, Any]) -> tuple[dict[str, Any], str]:
    files = [
        Path(__file__).with_name(n)
        for n in ["model.py", "encoder.py", "inspection.py", "common.py"]
    ]
    files += [
        Path(cfg["weight_root"]) / cfg["weight_filename"],
        Path(cfg["face_weights_root"]) / cfg["detection_file"],
    ]
    value = {
        "parameters": {k: cfg[k] for k in cfg["feature_parameter_keys"]},
        "files": {str(p): digest(p) for p in files},
        "versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "opencv": cv2.__version__,
        },
    }
    sha = hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return value, sha


def run(cfg: dict[str, Any], output: Path, tracker: Any) -> None:
    started = time.monotonic()
    entities, mapping, paths = collect(cfg)
    spec, signature = specification(cfg)
    save(output / "feature_spec.json", {"specification": spec, "sha256": signature})
    save(output / "entity_manifest.json", entities)
    count = (
        len(entities)
        if cfg["mode"] == "all"
        else min(len(entities), cfg["smoke_face_count"])
    )
    values = np.zeros((count, cfg["face_dimension"]), dtype=np.float32)
    norms = np.zeros((count, 1), dtype=np.float32)
    completed = np.zeros(count, dtype=bool)
    methods = [""] * count
    for folder in cfg["feature_resume_runs"]:
        root = Path(folder)
        reference = json.loads((root / "feature_spec.json").read_text())
        if reference["sha256"] != signature:
            raise AssertionError("续跑特征规格不相容")
        if not (root / "features.npz").exists():
            for i, record in enumerate(entities[:count]):
                file = root / "face_cache" / (record["face_sha256"] + ".npz")
                if file.exists():
                    with np.load(file) as z:
                        if str(z["specification_sha256"]) != signature:
                            raise AssertionError("单实体缓存规格不同")
                        values[i], norms[i], methods[i] = (
                            z["face"],
                            z["norm"],
                            str(z["method"]),
                        )
                        completed[i] = True
            continue
        old_entities = json.loads((root / "entity_manifest.json").read_text())
        old_methods = json.loads((root / "methods.json").read_text())
        cache = {r["face_sha256"]: i for i, r in enumerate(old_entities)}
        with np.load(root / "features.npz") as z:
            old_face, old_norm = z["face"], z["norm"]
            for i, record in enumerate(entities[:count]):
                j = cache.get(record["face_sha256"])
                if j is not None and j < len(old_face):
                    values[i], norms[i], methods[i] = (
                        old_face[j],
                        old_norm[j],
                        old_methods[j],
                    )
                    completed[i] = True
    encoder = Encoder(cfg)
    checks = []
    history = []
    (output / "face_cache").mkdir()
    missing = np.flatnonzero(~completed)
    for start in range(0, len(missing), cfg["face_batch_size"]):
        indices = missing[start : start + cfg["face_batch_size"]]
        feature, magnitude, method, diagnostic = encoder.encode(
            [entities[int(i)] for i in indices]
        )
        checks.extend(diagnostic)
        values[indices], norms[indices] = feature, magnitude
        for offset, i in enumerate(indices):
            methods[int(i)] = method[offset]
            target = output / "face_cache" / (entities[int(i)]["face_sha256"] + ".npz")
            temporary = target.with_suffix(".partial.npz")
            np.savez_compressed(
                temporary,
                face=feature[offset],
                norm=magnitude[offset],
                method=method[offset],
                specification_sha256=signature,
            )
            temporary.replace(target)
        completed[indices] = True
        if (
            int(completed.sum()) % cfg["feature_log_interval"] < cfg["face_batch_size"]
            or completed.all()
        ):
            logger.info("AdaFace特征 %s/%s", int(completed.sum()), count)
            row = {
                "processed_entities": int(completed.sum()),
                "total_entities": count,
                "seconds": time.monotonic() - started,
            }
            save(output / "progress.json", row)
            history.append(row)
            save(output / "history.json", history)
            if tracker is not None:
                try:
                    tracker.log(row)
                except Exception:
                    logger.exception("W&B特征进度失败")
        if time.monotonic() - started > cfg["time_limit_seconds"]:
            raise TimeoutError("特征作业达到预算，已保留单实体缓存")
    if not completed.all() or not np.allclose(
        np.linalg.norm(values, axis=1), 1, atol=cfg["reference_tolerance"]
    ):
        raise AssertionError("特征未完成或未归一化")
    for name, lookup in mapping.items():
        mapping[name] = np.where(lookup < count, lookup, -1)
    np.savez_compressed(
        output / "features.npz", face=values, norm=norms, **dict[str, Any](mapping)
    )
    save(output / "methods.json", methods)
    save(output / "reference_check.json", checks)
    save(
        output / "inputs.json",
        {str(p): digest(p) for p in paths + list(Path(__file__).parent.glob("*.py"))},
    )
    save(
        output / "result.json",
        {
            "mode": cfg["mode"],
            "faces": count,
            "seconds": time.monotonic() - started,
            "features_sha256": digest(output / "features.npz"),
            "feature_spec_sha256": signature,
            "resize_fallbacks": methods.count("resize_fallback"),
            "quality_labels_used": False,
        },
    )
    if cfg["mode"] == "all":
        plot(output, history, cfg)


def plot(output: Path, history: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties

        font = FontProperties(fname=cfg["plot_font_file"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            [r["seconds"] for r in history], [r["processed_entities"] for r in history]
        )
        ax.set_xlabel("实际经过时间（秒）", fontproperties=font)
        ax.set_ylabel("完成的人脸实体", fontproperties=font)
        (output / "plots").mkdir()
        fig.tight_layout()
        fig.savefig(output / "plots/AdaFace提取进度.png", dpi=cfg["plot_dpi"])
        plt.close(fig)
    except Exception:
        logger.exception("绘图失败，原始进度已保留")
