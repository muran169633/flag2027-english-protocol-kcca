"""将新教师指导后的学生按冻结Gaussian评分打包，仅用于隔离研究。"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from exps.v428.run import embed, load_checkpoint, score


def main() -> None:
    """真实加载15个蒸馏学生并评分，完整检查后保存研究ZIP。"""
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    cfg = json.loads((here / "config.json").read_text())
    package = cfg["package"]
    assert sys.prefix == cfg["python_prefix"]
    assert cfg["research_only"] and not cfg["formal_submission_allowed"]
    torch.set_num_threads(cfg["cpu_threads"])
    threadpool_limits(cfg["cpu_threads"])
    torch.cuda.set_per_process_memory_fraction(cfg["gpu_fraction"])
    torch.backends.cuda.matmul.allow_tf32 = False
    source = root / package["source_run"]
    destination = root / package["destination"]
    assert not destination.exists()
    assert json.loads((source / "independent_audit.json").read_text())["passed"]
    run_id = datetime.now().strftime("%m-%d-%H%M%S") + "-package-" + uuid4().hex[:6]
    out = root / cfg["output_root"] / cfg["experiment_id"] / run_id
    out.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")],
    )
    logger = logging.getLogger(__name__)
    start = time.monotonic()
    hashes: dict[str, str] = {}

    def record(path: Path) -> Path:
        hashes[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        return path

    def read(path: Path) -> Any:
        return json.loads(record(path).read_text())

    def feature(spec: list[str]) -> np.ndarray:
        with np.load(record(root / spec[0]), allow_pickle=False) as data:
            return data[spec[1]].astype(np.float32)

    record(Path(__file__))
    record(here / "run.py")
    record(here / "config.json")
    record(source / "result.json")
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    (out / "environment.json").write_text(
        json.dumps(
            {
                "executable": sys.executable,
                "prefix": sys.prefix,
                "torch": torch.__version__,
                "entry": sys.argv,
            },
            indent=2,
        )
    )
    trials = read(root / cfg["evaluation_manifest"])
    for side, path in cfg["eval_metadata"].items():
        rows = read(root / path)
        assert len(rows) == len(trials) == cfg["expected_eval_rows"]
        assert all(
            (a["pair_id"], a["track"], a["lang"])
            == (b["pair_id"], b["track"], b["language"])
            for a, b in zip(rows, trials, strict=True)
        ), side
    inputs = {
        side: np.concatenate([feature(spec) for spec in specs], axis=1)
        for side, specs in cfg["eval_features"].items()
    }
    assert all(np.isfinite(values).all() for values in inputs.values())
    index = np.arange(len(trials))
    pairs = np.column_stack([index, index, np.zeros(len(index), dtype=int)])
    predictions = []
    heads = []
    for seed in cfg["seeds"]:
        for fold in cfg["folds"]:
            key = f"B_s{seed}_f{fold}_{package['student_arm']}"
            checkpoint = load_checkpoint(record(source / f"{key}.pt"))
            assert checkpoint["research_only"]
            assert checkpoint["config"]["experiment_id"] == cfg["experiment_id"]
            with np.load(
                record(source / f"{key}_stats.npz"), allow_pickle=False
            ) as data:
                stats = {name: data[name] for name in data.files}
            face = embed(inputs["face"], checkpoint["state"], "face", cfg)
            voice = embed(inputs["voice"], checkpoint["state"], "voice", cfg)
            values = score(face, voice, pairs, stats)
            assert values.shape == (len(trials),) and np.isfinite(values).all()
            predictions.append(values)
            heads.append(key)
            logger.info("阶段=评分 head=%s rows=%s", key, len(values))
            assert time.monotonic() - start < cfg["max_seconds"]
    assert len(heads) == package["expected_heads"]
    raw_scores = np.mean(predictions, axis=0)
    distances = 0.5 - np.arctan(raw_scores / package["distance_scale"]) / np.pi
    assert np.all((distances > 0) & (distances < 1))
    files: dict[str, list[str]] = defaultdict(list)
    expected_ids: dict[str, list[str]] = defaultdict(list)
    for trial, distance in zip(trials, distances, strict=True):
        condition = "heard" if trial["language"] == "English" else "unheard"
        name = f"{trial['track']}/sub_score_v4_{trial['language']}_{condition}.txt"
        files[name].append(f"{trial['pair_id']} {distance:.{package['decimals']}f}\n")
        expected_ids[name].append(trial["pair_id"])
    assert len(files) == len(package["expected_counts"])
    for cell, count in package["expected_counts"].items():
        track, language = cell.split("/")
        condition = "heard" if language == "English" else "unheard"
        name = f"{track}/sub_score_v4_{language}_{condition}.txt"
        assert len(files[name]) == count
    with ZipFile(record(root / package["reference_zip"])) as reference:
        assert reference.testzip() is None and set(reference.namelist()) == set(files)
        for name, ids in expected_ids.items():
            assert [
                line.split()[0] for line in reference.read(name).decode().splitlines()
            ] == ids
            assert len(ids) == len(set(ids))
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial.zip")
    with ZipFile(partial, "x", compression=ZIP_DEFLATED) as archive:
        for name in sorted(files):
            archive.writestr(name, "".join(files[name]))
    with ZipFile(partial) as archive:
        assert archive.testzip() is None and set(archive.namelist()) == set(files)
        for name, ids in expected_ids.items():
            rows = [line.split() for line in archive.read(name).decode().splitlines()]
            assert all(len(row) == 2 for row in rows)
            assert [row[0] for row in rows] == ids
            assert np.isfinite([float(row[1]) for row in rows]).all()
    partial.rename(destination)
    np.savez_compressed(
        out / "scores.npz",
        head_scores=np.asarray(predictions),
        mean_scores=raw_scores,
        distances=distances,
    )
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    report = {
        "experiment_id": cfg["experiment_id"],
        "source_run": package["source_run"],
        "heads": heads,
        "zip": package["destination"],
        "sha256": digest,
        "counts": {name: len(lines) for name, lines in files.items()},
        "research_only": True,
        "formal_submission_allowed": False,
        "uses_evaluation_information_in_training": True,
        "uploaded": False,
        "elapsed_seconds": time.monotonic() - start,
    }
    (out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out / "input_hashes.json").write_text(json.dumps(hashes, indent=2))
    destination.with_suffix(".说明.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    logger.info("阶段=完成 result=%s", report)


if __name__ == "__main__":
    main()
