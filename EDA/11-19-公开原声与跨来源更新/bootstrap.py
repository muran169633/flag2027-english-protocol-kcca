"""以相同人物重抽检验冻结系统之间 EER 的配对差值。"""

from __future__ import annotations

import csv

import numpy as np
from media_common import CFG, LOC, ROOT, STUDY, eer, read


def ranked(y: np.ndarray, score: np.ndarray) -> tuple:
    ix = np.argsort(-score, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(score[ix]) != 0), len(score) - 1]
    return ix, ends, y[ix]


def contrasts(
    point: np.ndarray, draws: np.ndarray, names: list[str], pairs: list
) -> dict:
    result = {}
    for left, right in pairs:
        if left not in names or right not in names:
            continue
        ix, iy = names.index(left), names.index(right)
        delta = draws[:, ix] - draws[:, iy]
        good = delta[np.isfinite(delta)]
        assert len(good) > 0
        result[left + "-" + right] = {
            "pp": float(point[ix] - point[iy]),
            "ci95": np.quantile(good, [0.025, 0.975]).tolist(),
            "draws": len(good),
        }
    return result


def fast(w: np.ndarray, rk: tuple) -> float:
    ix, ends, y = rk
    t = np.r_[0, np.cumsum(w[ix] * y)[ends]]
    f = np.r_[0, np.cumsum(w[ix] * (1 - y))[ends]]
    f /= f[-1]
    t /= t[-1]
    return float(100 * np.interp(0, f + t - 1, f))


def intervals(
    q: np.ndarray,
    w: np.ndarray,
    fid: np.ndarray,
    vid: np.ndarray,
    sex: dict,
    scores: list,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    a, b, y = q.T
    who = np.union1d(fid[a], vid[b])
    aa, bb = np.searchsorted(who, fid[a]), np.searchsorted(who, vid[b])
    labels = 2 * aa + y
    totals = np.bincount(labels, weights=w, minlength=2 * len(who))
    groups = np.array([sex[n] for n in who])
    rng = np.random.default_rng(seed)
    counts = np.zeros((CFG["bootstrap_draws"], len(who)), int)
    for g in np.unique(groups):
        ix = groups == g
        counts[:, ix] = rng.multinomial(
            int(ix.sum()), np.ones(ix.sum()) / ix.sum(), size=len(counts)
        )
    ranks = [ranked(y, s) for s in scores]
    point = np.array([fast(w, rk) for rk in ranks])
    np.testing.assert_allclose(point, [eer(y, s, w) for s in scores], atol=1e-9)
    samples = []
    for count in counts:
        inner = w * np.where(y == 1, 1, count[bb])
        after = np.bincount(labels, weights=inner, minlength=len(totals))
        if np.any((totals > 0) & (np.repeat(count, 2) > 0) & (after == 0)):
            samples.append([np.nan] * len(scores))
            continue
        weighted = inner * count[aa] * totals[labels] / np.maximum(after[labels], 1e-20)
        if not weighted[y == 1].any() or not weighted[y == 0].any():
            samples.append([np.nan] * len(scores))
            continue
        samples.append([fast(weighted, rk) for rk in ranks])
    return point, np.asarray(samples)


def genders() -> dict:
    src = read(STUDY / "媒体索引.json")
    sex = {r["id"]: r["gender"] for r in src["flag"]}
    sex.update({r["id"]: r["gender"][0].lower() for r in src["english"]})
    sex.update(
        {
            r["id"]: {"her": "f", "his": "m"}[r["catalog_pronoun"]]
            for r in read(LOC / "评分前锁定.json")["people"]
        }
    )
    with (ROOT / "data/external/voxceleb2_test/vox2_meta.csv").open(
        encoding="utf-8-sig"
    ) as fp:
        for row in list(csv.reader(fp))[1:]:
            person, gender = row[0].strip(), row[2].strip().lower()
            sex[person] = sex["vox2t_" + person] = gender
    with (ROOT / "data/external/CREMA_D/provenance/VideoDemographics.csv").open(
        encoding="utf-8-sig"
    ) as fp:
        sex.update(
            {"crema_" + r["ActorID"]: r["Sex"][0].lower() for r in csv.DictReader(fp)}
        )
    return sex
