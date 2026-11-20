"""在参数冻结后，以单对输入考察训练人物核方向及现有协议专用系统。"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from anchor_baselines import sources
from anchor_model import SPEC, normalized, project, scores
from anchor_train import stable
from evaluate import uncertainty
from media_common import CFG, REF, ROOT, SIDES, arrays, read, setup, sha, write

LOG = logging.getLogger(__name__)


def choices(
    q: np.ndarray,
    face_id: np.ndarray,
    voice_id: np.ndarray,
    sex: dict,
    candidates: dict,
    key: str,
) -> dict:
    """相同脸在本人和固定异人声音间二选一，先对各人物内部平均。"""
    rng = np.random.default_rng(stable("paired-choice", key))
    by = {}
    for i, (f, _v, y) in enumerate(q):
        by.setdefault(f, ([], []))[y].append(i)
    pos = []
    neg = []
    for genuine, impostor in ((p, n) for n, p in by.values() if n and p):
        for i in genuine:
            keep = rng.choice(
                impostor,
                size=min(len(impostor), SPEC["paired_negatives"]),
                replace=False,
            )
            pos.extend([i] * len(keep))
            neg.extend(keep)
    if not pos:
        return {}
    pi, ni = np.array(pos), np.array(neg)
    np.testing.assert_array_equal(q[pi, 0], q[ni, 0])
    assert np.all(q[pi, 2] == 1) and np.all(q[ni, 2] == 0)
    owners = face_id[q[pi, 0]]
    gs = np.array([sex.get(n, "?") for n in owners])
    gt = np.array([sex.get(n, "?") for n in voice_id[q[ni, 1]]])
    known = np.isin(gs, ["m", "f"]) & np.isin(gt, ["m", "f"])
    res = {}
    for group, mask in (
        ("all", np.ones(len(pi), bool)),
        ("same_gender", known & (gs == gt)),
        ("different_gender", known & (gs != gt)),
    ):
        if not mask.any():
            continue
        who, idx, counts = np.unique(
            owners[mask], return_inverse=True, return_counts=True
        )
        w = 1 / (len(who) * counts[idx])
        ans = {}
        for name, values in candidates.items():
            d = values[pi[mask]] - values[ni[mask]]
            ans[name] = (np.sign(d) + 1) / 2
        comparisons = {}
        for left, right in (
            ("specialist", "current"),
            ("specialist_plus_rbf", "specialist"),
            ("specialist_plus_rbf", "specialist_plus_permuted"),
        ):
            change = ans[left] - ans[right]
            comparisons[left + "-" + right] = {
                "corrected_percent": float(100 * (np.maximum(change, 0) @ w)),
                "broken_percent": float(100 * (np.maximum(-change, 0) @ w)),
            }
        res[group] = {
            "people": len(who),
            "comparisons": int(mask.sum()),
            "accuracy": {n: float(100 * (v @ w)) for n, v in ans.items()},
            "direction": comparisons,
        }
    return res


def native_predictions(
    group: str,
    frozen: Path,
    locked: dict,
    domains: dict,
    cells: dict,
    variants: dict[str, list[str]] | None = None,
) -> tuple:
    outputs, counts, checks = {}, {}, {}
    if variants is None:
        variants = {group: SPEC["models"], "official": ["rbf"]}
    for f in locked["folds"]:
        active = {k: c for k, c in cells.items() if f in c.folds}
        for regime, methods in variants.items():
            spec = locked["entries"][f"{regime}/fold{f}"]["fit"]
            for m in methods:
                name = "official" if regime == "official" else m
                path = frozen / regime / f"fold{f}" / (m + ".npz")
                assert sha(path) == spec["models"][m]["sha256"]
                par = arrays(path)
                for dom in sorted({c.domain for c in active.values()}):
                    ds = domains[dom]
                    subset = {k: c for k, c in active.items() if c.domain == dom}
                    ix = [
                        np.unique(np.concatenate([c.q[:, j] for c in subset.values()]))
                        for j in range(2)
                    ]
                    raw = (ds.face, ds.voice)
                    em = [
                        project(normalized(x[ids], side), side, par)
                        for side, x, ids in zip(SIDES, raw, ix, strict=True)
                    ]
                    for k, c in subset.items():
                        q = c.q.copy()
                        for j in range(2):
                            q[:, j] = np.searchsorted(ix[j], q[:, j])
                        scl, shift = par["calibration_" + c.track]
                        value = scl * scores(em, q) + shift
                        key = name + "__" + k
                        if key not in outputs:
                            max_error = 0.0
                            for i in np.linspace(0, len(q) - 1, 3, dtype=int):
                                v = [
                                    project(normalized(x[[q0]], side), side, par)[0]
                                    for side, x, q0 in zip(
                                        SIDES, raw, c.q[i, :2], strict=True
                                    )
                                ]
                                max_error = max(
                                    max_error,
                                    float(abs(scl * (v[0] @ v[1]) + shift - value[i])),
                                )
                            assert max_error < 1e-7
                            checks[key] = max_error
                            outputs[key] = value
                            counts[key] = 1
                        else:
                            counts[key] += 1
                            outputs[key] += value
    for key, values in outputs.items():
        assert counts[key] == len(cells[key.split("__", 1)[1]].folds)
        values /= counts[key]
    return outputs, checks


def run(frozen: Path, baseline: Path, out: Path) -> None:
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    setup(out, "核方向验证.log")
    lock = read(frozen / "冻结清单.json")
    assert lock["spec"] == SPEC and lock["folds"] == REF["folds"]
    f = baseline / "冻结评分.npz"
    assert sha(f) == read(baseline / "完成.json")["score_sha256"]
    transfer_source = ROOT / CFG["transfer_run"] / "冻结试验与逐对预测.npz"
    summary = read(ROOT / CFG["transfer_run"] / "各协议成绩.json")
    write(
        out / "评分前锁定.json",
        {
            "design": SPEC,
            "training_sha256": sha(frozen / "冻结清单.json"),
            "dev_baselines_sha256": sha(f),
            "transfer_baselines_sha256": sha(transfer_source),
            "bootstrap_draws": CFG["bootstrap_draws"],
        },
    )
    result = {}
    pack = {}
    mnames = [
        "current",
        "count_matched",
        "same_gender",
        "specialist",
        "midpoint",
        "rbf",
        "linear",
        "permuted",
        "official",
        "specialist_plus_rbf",
        "specialist_plus_linear",
        "specialist_plus_permuted",
    ]
    for group, file in (("development", f), ("transfer", transfer_source)):
        data, cells, sex = sources(lock, group)
        out_scores, checks = native_predictions(group, frozen, lock, data, cells)
        ranks = {}
        with np.load(file, allow_pickle=False) as base:
            for k, c in cells.items():
                if group == "development":
                    np.testing.assert_array_equal(c.q, base["q__" + k])
                    np.testing.assert_array_equal(c.weight, base["w__" + k])
                for name, legacy in (
                    ("current", "vox_cross"),
                    ("same_gender", "same_gender"),
                    ("count_matched", "count_matched"),
                ):
                    old = name if group == "development" else "equal_" + legacy
                    out_scores[name + "__" + k] = base[old + "__" + k]
                n = "count_matched" if c.track == "standard" else "same_gender"
                specialist = out_scores[n + "__" + k]
                out_scores["specialist__" + k] = specialist
                out_scores["midpoint__" + k] = (
                    specialist + out_scores["current__" + k]
                ) / 2
                for m in SPEC["models"]:
                    out_scores["specialist_plus_" + m + "__" + k] = (
                        specialist + out_scores[m + "__" + k]
                    )
                ds = data[c.domain]
                ranks[k] = choices(
                    c.q,
                    ds.face_ids,
                    ds.voice_ids,
                    sex,
                    {n: out_scores[n + "__" + k] for n in mnames},
                    k,
                )
        records, counts = uncertainty(
            cells,
            data,
            sex,
            out_scores,
            mnames,
            group == "transfer",
            SPEC["primary"],
            SPEC["comparisons"],
        )
        if group == "transfer":
            for k in cells:
                for n, source_arm in (
                    ("current", "vox_cross"),
                    ("same_gender", "same_gender"),
                    ("count_matched", "count_matched"),
                ):
                    np.testing.assert_allclose(
                        records[k]["eer"][n],
                        summary["eer"]["equal_" + source_arm][k],
                        atol=1e-8,
                    )
        else:
            old = read(ROOT / SPEC["release"])["scores"]
            for k in ("flag_two", "flag_standard", "flag_same_gender"):
                for n in ("current", "count_matched", "same_gender"):
                    np.testing.assert_allclose(
                        records[k]["eer"][n], old[k]["eer"][n], atol=1e-8
                    )
        result[group] = {
            "cells": records,
            "counts": counts,
            "per_face_choice": ranks,
            "single_pair_errors": checks,
        }
        pack.update({group + "/" + k: v for k, v in out_scores.items()})
        for k, c in cells.items():
            pack[group + "/q__" + k] = c.q
            pack[group + "/w__" + k] = c.weight
        LOG.info("完成 %s 共 %d 个条件", group, len(cells))
    file = out / "所有独立配对分数.npz"
    np.savez_compressed(file, allow_pickle=False, **pack)
    write(
        out / "核方向与分协议结果.json",
        {
            "results": result,
            "score_sha256": sha(file),
            "lock_sha256": sha(out / "评分前锁定.json"),
            "seconds": time.monotonic() - start,
        },
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for n in ("frozen", "baseline", "out"):
        cli.add_argument(n, type=Path)
    a = cli.parse_args()
    run(a.frozen, a.baseline, a.out)
