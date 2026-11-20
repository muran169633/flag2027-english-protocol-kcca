"""重载冻结参数，逐对比较固定系统，并按同一批人物对 EER 差值重抽样。"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from bootstrap import contrasts, intervals
from data_eval import development, transfer
from freeze import unpack
from grouped_gaussian import precise_project, scored, single_pairs
from media_common import CFG, TRACKS, eer, read, setup, sha, write
from media_eval_data import Cell

LOG = logging.getLogger(__name__)


def uncertainty(
    cells: dict,
    domains: dict,
    sex: dict,
    results: dict,
    names: list[str],
    xfer: bool,
    primary: str,
    pairs: list | None = None,
) -> tuple[dict, dict]:
    points = {}
    draws = {}
    counts = {}
    for k, c in cells.items():
        ds = domains[c.domain]
        seed = CFG["seed"] + (
            c.folds[0] if c.domain == "flag" else 10 + sorted(domains).index(c.domain)
        )
        values = [results[n + "__" + k] for n in names]
        points[k], draws[k] = intervals(
            c.q, c.weight, ds.face_ids, ds.voice_ids, sex, values, seed
        )
        np.testing.assert_allclose(
            points[k], [eer(c.q[:, 2], v, c.weight) for v in values], atol=1e-8
        )
        counts[k] = {
            "faces": len(set(ds.face_ids[c.q[:, 0]])),
            "people": len(
                np.unique(np.r_[ds.face_ids[c.q[:, 0]], ds.voice_ids[c.q[:, 1]]])
            ),
            "positive": int(c.q[:, 2].sum()),
            "trials": len(c.q),
        }
    if xfer:
        summary = {
            "english_urdu_four": [k for k in cells if k.startswith("v1_")],
            "english_hindi_two": [k for k in cells if k.startswith("v2_")],
        }
    else:
        summary = {
            "flag_" + tr: [
                k for k in cells if k.startswith("fold") and k.endswith("_" + tr)
            ]
            for tr in TRACKS
        }
        summary["flag_two"] = [k for k in cells if k.startswith("fold")]
    for key, group in summary.items():
        assert len(group) >= 2
        points[key] = np.mean([points[k] for k in group], axis=0)
        draws[key] = np.mean([draws[k] for k in group], axis=0)
    report = {}
    for k, p in points.items():
        comparisons = {}
        for r in names:
            if r == primary:
                continue
            ix, iy = names.index(primary), names.index(r)
            delta = draws[k][:, ix] - draws[k][:, iy]
            valid = delta[np.isfinite(delta)]
            comparisons[r] = {
                "pp": float(p[ix] - p[iy]),
                "ci95": np.quantile(valid, [0.025, 0.975]).tolist()
                if len(valid)
                else None,
                "usable": len(valid),
            }
        report[k] = {
            "eer": dict(zip(names, p.tolist(), strict=True)),
            "primary_minus": comparisons,
        }
        if pairs:
            report[k]["paired"] = contrasts(p, draws[k], names, pairs)
    return report, counts


def run(frozen: Path, out: Path) -> None:
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    setup(out, "评估.log")
    locked = read(frozen / "冻结清单.json")
    is_transfer = locked["mode"]
    arms = locked["methods"]
    ds, cells, sex, bases, reference = (
        transfer(locked["members"]) if is_transfer else development()
    )
    allscores = {}
    independent = {}
    for f in locked["folds"]:
        for s in locked["seeds"]:
            for arm in arms:
                stem = f"{arm}/fold{f}_s{s}"
                for tr in ["plain"] if is_transfer else TRACKS:
                    key = stem if is_transfer else stem + "_" + tr
                    state, co = unpack(frozen / (key + ".npz"), locked["params"][key])
                    relevant: dict[str, Cell] = {
                        k: c
                        for k, c in cells.items()
                        if (is_transfer or tr == c.track) and f in c.folds
                    }
                    for domain in {c.domain for c in relevant.values()}:
                        media = ds[domain]
                        embed = precise_project(media, state)
                        for k, c in relevant.items():
                            if c.domain != domain:
                                continue
                            v = scored(embed, c.q, *co)
                            if f == locked["folds"][0] and s == locked["seeds"][0]:
                                idx = np.unique(
                                    np.linspace(0, len(c.q) - 1, 5, dtype=int)
                                )
                                expected = single_pairs(state, co, media, c.q[idx])
                                np.testing.assert_allclose(
                                    expected, v[idx], atol=2e-9, rtol=0
                                )
                                independent[arm + "/" + k] = float(
                                    np.abs(expected - v[idx]).max()
                                )
                            allscores.setdefault((arm, k), []).append(v)
        LOG.info("完成第 %d 折", f)
    preds = {}
    standalone = {}
    reference_error = {}
    for (arm, k), v in allscores.items():
        assert len(v) == len(cells[k].folds) * len(locked["seeds"])
        vs = np.mean(v, axis=0)
        standalone[arm + "__" + k] = vs
        preds[arm + "__" + k] = vs if is_transfer else 0.5 * (vs + bases[k])
        if arm == "baseline" and not is_transfer:
            delta = preds[arm + "__" + k] - reference[k]
            np.testing.assert_allclose(delta, 0, rtol=0, atol=1e-4)
            reference_error[k] = float(np.max(np.abs(delta)))
    r, counts = uncertainty(
        cells,
        ds,
        sex,
        preds,
        arms,
        is_transfer,
        locked.get("primary", CFG["primary"]),
        locked.get("paired_comparisons"),
    )
    branch = {
        arm: {
            k: eer(c.q[:, 2], standalone[arm + "__" + k], c.weight)
            for k, c in cells.items()
        }
        for arm in arms
    }
    path = out / "逐对冻结分数.npz"
    np.savez_compressed(
        path,
        **preds,
        **(
            {"branch_" + k: v for k, v in standalone.items()} if not is_transfer else {}
        ),
    )
    write(
        out / "研究结果.json",
        {
            "cells": r,
            "counts": counts,
            "unblended": branch,
            "score_sha256": sha(path),
            "frozen_sha256": sha(frozen / "冻结清单.json"),
            "single_pair_max": independent,
            "prior_max": reference_error,
            "seconds": time.monotonic() - start,
        },
    )
    LOG.info("已核验 %d 组单对推断，完成差值重抽", len(independent))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("frozen", type=Path)
    cli.add_argument("out", type=Path)
    a = cli.parse_args()
    run(a.frozen, a.out)
