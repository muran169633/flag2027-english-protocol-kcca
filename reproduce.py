"""Recalculate the local English metrics or score the archived Evaluation feature set."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "EDA/11-19-公开原声与跨来源更新"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing archived input: {path}")
    if digest(path) != expected:
        raise ValueError(f"Archived SHA-256 differs: {path}")


def metrics() -> dict[str, object]:
    resources = json.loads((ROOT / "resources.json").read_text())
    record = resources["validation_scores"]
    scores = ROOT / record["path"]
    verify(scores, record["sha256"])
    comparison = json.loads((scores.parent / "核方向与分协议结果.json").read_text())
    saved = comparison["results"]["development"]["cells"]
    folds: dict[str, list[float]] = {"standard": [], "same_gender": []}
    trials = 0
    with np.load(scores, allow_pickle=False) as arrays:
        for protocol, values in folds.items():
            for fold in range(5):
                cell = f"fold{fold}_{protocol}"
                # The saved fold scores are similarities. Each query row holds two indices and its binary target.
                queries = arrays[f"development/q__{cell}"]
                weights = arrays[f"development/w__{cell}"]
                similarity = arrays[f"development/specialist_plus_rbf__{cell}"]
                fpr, tpr, _ = roc_curve(
                    queries[:, 2],
                    similarity,
                    sample_weight=weights,
                    drop_intermediate=False,
                )
                eer = float(100 * np.interp(0, fpr + tpr - 1, fpr))
                np.testing.assert_allclose(
                    eer, saved[cell]["eer"]["specialist_plus_rbf"], atol=1e-10, rtol=0
                )
                values.append(eer)
                trials += len(queries)
    means = {k: float(np.mean(v)) for k, v in folds.items()}
    return {
        "english_trials": trials,
        "eer_by_fold_percent": folds,
        "protocol_mean_percent": means,
        "mean_eer_percent": float(np.mean(list(means.values()))),
    }


def replay(resource_root: Path, output: Path) -> dict[str, object]:
    inventory = json.loads((ROOT / "resources.json").read_text())
    reference = ROOT / "reference/candidate.zip"
    verify(reference, inventory["package_sha256"])
    for name, item in inventory["external_resources"].items():
        if "replay" in item["stages"]:
            verify(resource_root / name, item["sha256"])
    for name, expected in inventory["parameters"].items():
        verify(ROOT / name, expected)
    if output.exists():
        raise FileExistsError(f"Choose a new output path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(STUDY))
    package = importlib.import_module("anchor_package")
    # Keep the original cell batches and BLAS thread count for the 15-decimal archived output.
    with threadpool_limits(limits=2):
        matcher = package.FrozenPairMatcher()
        if matcher.fingerprint != inventory["parameters"]:
            raise ValueError("The loaded parameters differ from the archived set")
        result = package.write_package(
            output,
            {"no_gender": matcher.distance, "gender": matcher.distance},
            resource_root=resource_root,
        )
    if result["pairs"] != inventory["cells"]:
        raise ValueError("Protocol cells or trial counts differ from the archive")
    with ZipFile(reference) as old, ZipFile(output) as new:
        identical = {name: new.read(name) == old.read(name) for name in old.namelist()}
    if not all(identical.values()) or result["sha256"] != inventory["package_sha256"]:
        raise ValueError(
            f"Replay differs from the archive: {identical}; result at {output}"
        )
    return {
        "cells": result["pairs"],
        "identical_members": identical,
        "zip_sha256": result["sha256"],
        "path": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser(
        "metrics", help="Recalculate the five-fold English development EERs"
    )
    score = actions.add_parser(
        "replay", help="Run the frozen matcher and verify the complete ZIP"
    )
    score.add_argument(
        "--resources",
        type=Path,
        required=True,
        help="Authorized root containing the recorded feature arrays, indexes and trial lists",
    )
    score.add_argument(
        "--output", type=Path, required=True, help="Unused output ZIP path"
    )
    args = parser.parse_args()
    result = (
        metrics()
        if args.action == "metrics"
        else replay(args.resources.resolve(), args.output.resolve())
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
