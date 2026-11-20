"""用英语训练人物拟合跨模态核方向，并用训练集内未见身份校准分数。"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
from anchor_model import SPEC, fit, normalized, project, scores
from media_common import CFG, PC, REF, ROOT, SIDES, TRACKS, N, read, setup, sha, write
from plans import gender_table, pools
from sampling import load_sources
from sklearn.linear_model import LogisticRegression

LOG = logging.getLogger(__name__)


def stable(*keys: object) -> int:
    msg = "/".join(map(str, [SPEC["seed"], *keys]))
    return int.from_bytes(hashlib.sha256(msg.encode()).digest()[:8], "big")


def training_data() -> tuple:
    raw, old, extra = load_sources(
        {**REF, "device": "cpu"}, read(ROOT / CFG["screened_vox"])
    )
    meta = [
        rows + read(ROOT / REF["metadata"][side][1])
        for side, rows in zip(SIDES, old, strict=True)
    ]
    return [t.numpy() for t in raw], meta, extra


def prepare(pool: dict, data: list, meta: list, sx: dict) -> tuple:
    people = sorted(pool)
    source, sex = [], []
    center_lists = [[], []]
    samples = [[], []]
    videos = [[], []]
    actual = [[], []]
    selected = {}
    for n in people:
        selected[n] = []
        source.append(n.split("_")[0])
        sex.append(sx.get(n, "?"))
        for j, side in enumerate(SIDES):
            idx = pool[n][j]
            seen_files = set()
            video_keys = []
            keep = []
            for i in idx:
                r = meta[j][i]
                assert REF["canonical_alias"].get(r["id"], r["id"]) == n
                if r["path"] in seen_files:
                    continue
                seen_files.add(r["path"])
                video_keys.append(r["video"] or r["path"])
                keep.append(int(i))
            ix = np.array(keep)
            x = normalized(data[j][ix], side)
            groups, inv, count = np.unique(
                video_keys, return_inverse=True, return_counts=True
            )
            w = 1 / (len(groups) * count[inv])
            assert np.isclose(w.sum(), 1)
            center_lists[j].append(normalized((w @ x)[None, :], side)[0])
            # 先选不同录制，再补取不同媒体。每侧按独立的固定哈希排序。
            ranked = sorted(range(len(ix)), key=lambda i: stable(n, side, int(ix[i])))
            first = {}
            for i in ranked:
                first.setdefault(video_keys[i], i)
            order = list(first.values()) + [
                i for i in ranked if first[video_keys[i]] != i
            ]
            take = np.resize(np.asarray(order), SPEC["media_per_person"])
            samples[j].append(x[take])
            actual[j].append(ix[take])
            videos[j].append([meta[j][i]["video"] for i in ix[take]])
            selected[n].append(
                {
                    "rows": ix[take].tolist(),
                    "all_indices_sha256": hashlib.sha256(ix.tobytes()).hexdigest(),
                    "media": len(ix),
                    "recording_or_file_groups": len(groups),
                }
            )
    strata = np.array([a + "/" + b for a, b in zip(source, sex, strict=True)])
    return (
        people,
        strata,
        np.array(sex),
        [np.stack(a) for a in center_lists],
        [np.stack(a) for a in samples],
        [np.asarray(a) for a in videos],
        [np.stack(a) for a in actual],
        selected,
    )


def partition(names: list[str], strata: np.ndarray) -> np.ndarray:
    split = np.zeros(len(names), dtype=int)
    offset = 0
    for s in np.unique(strata):
        ix = sorted(
            np.flatnonzero(strata == s), key=lambda i: stable("inner", names[i])
        )
        for t, i in enumerate(ix):
            split[i] = (offset + t) % SPEC["inner_groups"]
        offset += len(ix)
    assert np.ptp(np.bincount(split)) <= 1
    return split


def queries(
    test: np.ndarray, sex: np.ndarray, vids: list, row_ids: list, track: str
) -> tuple[np.ndarray, np.ndarray]:
    slots = SPEC["media_per_person"]
    ai, bi = np.indices((len(test) * slots, len(test) * slots))
    a, b = test[ai // slots], test[bi // slots]
    pos = a == b
    fa, vo = vids[0][a, ai % slots], vids[1][b, bi % slots]
    known = (fa != "") & (vo != "")
    ok = ~pos | np.where(
        known, fa != vo, row_ids[0][a, ai % slots] != row_ids[1][b, bi % slots]
    )
    if track == "same_gender":
        ok &= (sex[a] != "?") & (sex[a] == sex[b])
    aa, bb, y = ai[ok], bi[ok], pos[ok].astype(int)
    g = 2 * (aa // slots) + y
    counts = np.bincount(g, minlength=2 * len(test)).reshape(-1, 2)
    valid = (counts[:, 0] > 0) & (counts[:, 1] > 0)
    valid = valid[aa // slots]
    aa, bb, y, g = aa[valid], bb[valid], y[valid], g[valid]
    assert len(np.unique(y)) == 2
    return np.column_stack((aa, bb, y)), 0.5 / counts.reshape(-1)[g]


def calibrate(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, dict]:
    m = float(np.average(x, weights=w))
    sd = float(np.sqrt(np.average((x - m) ** 2, weights=w)))
    assert sd > 1e-8
    model = LogisticRegression(
        C=SPEC["calibration_c"], solver="lbfgs", max_iter=250, tol=1e-10
    )
    model.fit(((x - m) / sd)[:, None], y, sample_weight=w)
    slope = float(model.coef_[0, 0]) / sd
    intercept = float(model.intercept_[0]) - slope * m
    pred = slope * x + intercept
    np.testing.assert_allclose(
        model.decision_function(((x - m) / sd)[:, None]), pred, atol=1e-10
    )
    return np.array([slope, intercept]), {
        "samples": len(y),
        "effective_people": float(w.sum()),
        "slope": slope,
        "intercept": intercept,
        "held_calibration_fit_bce": float(
            np.average(np.logaddexp(0, pred) - y * pred, weights=w)
        ),
    }


def fit_fold(
    folder: Path, prepared: tuple, fold: int, methods: list[str] | None = None
) -> dict:
    names, strata, sex, centers, samples, vids, rows, selection = prepared
    splits = partition(names, strata)
    if methods is None:
        methods = list(SPEC["models"])
    blocks = {m: {track: [] for track in TRACKS} for m in methods}
    for g in range(SPEC["inner_groups"]):
        test = np.where(splits == g)[0]
        train = np.where(splits != g)[0]
        assert set(test).isdisjoint(train)
        pairs = {t: queries(test, sex, vids, rows, t) for t in TRACKS}
        for method in methods:
            pars = fit(
                [x[train] for x in centers], strata[train], method, stable(fold, g)
            )
            emb = [
                project(v[test].reshape(-1, v.shape[-1]), side, pars)
                for side, v in zip(SIDES, samples, strict=True)
            ]
            for tr, (q, w) in pairs.items():
                blocks[method][tr].append((scores(emb, q), q[:, 2], w))
    seals = {}
    for method in methods:
        state = fit(centers, strata, method, stable(fold, "final"))
        record = {}
        for tr in TRACKS:
            x, y, w = (np.concatenate(a) for a in zip(*blocks[method][tr], strict=True))
            state["calibration_" + tr], record[tr] = calibrate(x, y, w)
        p = folder / (method + ".npz")
        np.savez_compressed(p, allow_pickle=False, **state)
        seals[method] = {
            "sha256": sha(p),
            "calibration": record,
            "correlations": state["correlations"].tolist(),
            "identity_correspondences_changed": int(
                np.sum(state["permutation"] != np.arange(len(names)))
            ),
        }
    return {
        "names": names,
        "strata": strata.tolist(),
        "inner_groups": splits.tolist(),
        "selected": selection,
        "models": seals,
    }


def run(out: Path, first: bool) -> None:
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    setup(out, "训练.log")
    data, meta, extra = training_data()
    sx = gender_table()
    folds = REF["folds"][:1] if first else REF["folds"]
    lock = {
        "spec": SPEC,
        "folds": folds,
        "argv": sys.argv,
        "python": sys.executable,
        "prefix": sys.prefix,
        "split_sha256": sha(ROOT / CFG["split_manifest"]),
        "english_pool_sha256": sha(ROOT / PC["english_pools"]),
        "screening_sha256": sha(ROOT / CFG["screened_vox"]),
    }
    write(out / "训练前锁定.json", lock)
    entries = {}
    with np.load(ROOT / PC["english_pools"], allow_pickle=False) as source:
        for f in folds:
            standard, _, m = pools(f, source, meta, extra, False)
            strict, _, ms = pools(f, source, meta, extra, True)
            official = {
                n: tuple(i[i < N] for i in axes)
                for n, axes in standard.items()
                if n.startswith("v4_") and all(np.sum(i < N) >= 2 for i in axes)
            }
            assert all(
                meta[j][i]["src"] == "v4" and meta[j][i]["lang"] == "English"
                for sides in official.values()
                for j, ix in enumerate(sides)
                for i in ix
            )
            for group, pool, split in (
                ("development", standard, m),
                ("transfer", strict, ms),
                ("official", official, m),
            ):
                folder = out / group / f"fold{f}"
                folder.mkdir(parents=True)
                assert not set(pool) & set(split["denied"])
                entries[f"{group}/fold{f}"] = {
                    "split": split,
                    "fit": fit_fold(folder, prepare(pool, data, meta, sx), f),
                }
                LOG.info("划分 %s 折 %d 训练人物 %d", group, f, len(pool))
    write(
        out / "冻结清单.json",
        {
            "folds": folds,
            "spec": SPEC,
            "entries": entries,
            "design_sha256": sha(out / "训练前锁定.json"),
            "seconds": time.monotonic() - start,
        },
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("out", type=Path)
    cli.add_argument("--first", action="store_true")
    args = cli.parse_args()
    run(args.out, args.first)
