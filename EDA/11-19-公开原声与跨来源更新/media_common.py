"""读取当前候选和独立验证的固定产物，集中核对配置和环境。"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_curve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SIDES = ("face", "voice")
TRACKS = {"standard": "no_gender", "same_gender": "gender"}


def read(p: Path) -> Any:
    return json.loads(p.read_bytes())


def write(p: Path, obj: Any) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(p)


def sha(p: Path) -> str:
    with p.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def arrays(p: Path) -> dict[str, np.ndarray]:
    with np.load(p, allow_pickle=False) as z:
        return dict(z)


CFG = read(HERE / "config.json")
PC = read(ROOT / CFG["parent_training"])
REF = read(ROOT / PC["reference_config"])
EC = read(ROOT / CFG["external_config"])
PARENT = ROOT / CFG["parent_run"]
STUDY = ROOT / CFG["study_run"]
LOC = ROOT / CFG["loc"]
N = REF["official_rows"]


def setup(out: Path, log: str) -> None:
    assert (
        Path(sys.executable).parent.parent
        == Path(sys.prefix)
        == Path(PC["python_prefix"])
    )
    torch.set_num_threads(PC["resource_cpu_threads"])
    torch.cuda.set_per_process_memory_fraction(PC["resource_gpu_fraction"])
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / log)],
        force=True,
    )


def feature(blocks: Iterable, rows: np.ndarray | slice = slice(None)) -> np.ndarray:
    vals = []
    for path, name in blocks:
        with np.load(ROOT / path, allow_pickle=False) as z:
            vals.append(z[name][rows])
    data = np.ascontiguousarray(np.column_stack(vals), dtype=np.float32)
    assert np.isfinite(data).all()
    return data


def train_features() -> list[np.ndarray]:
    parts = REF["feature_blocks"]
    return [
        feature([parts["face"][0], parts["face_additional"][0]]),
        feature(parts["voice_train"]),
    ]


def official_meta() -> tuple[list[dict], list[list[dict]]]:
    media = read(STUDY / "媒体索引.json")["flag"]
    meta = [read(ROOT / REF["metadata"][s][0]) for s in SIDES]
    assert len(media) == N
    for side, bank in zip(SIDES, meta, strict=True):
        assert all(
            (v["id"], ROOT / v[side]) == (m["id"], Path(m["path"]))
            and m["lang"] == "English"
            and m["src"] == "v4"
            for v, m in zip(media, bank[:N], strict=True)
        )
    return media, meta


def fold_members(fold: int) -> dict:
    return read(ROOT / REF["parent_run"] / f"fold_protocols/fold_{fold}.json")


def allowed_names(fold: int, seed: int) -> dict:
    return read(ROOT / REF["parent_run"] / f"plans/s{seed}_f{fold}_final.json")


def eer(y: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    fp, tp, _ = roc_curve(y, score, sample_weight=weight, drop_intermediate=False)
    return float(100 * np.interp(0, fp + tp - 1, fp))
