"""只用允许的英语训练人物拟合并固化高斯参数。"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from grouped_gaussian import donor_groups, make_parameters, project
from media_common import (
    CFG,
    REF,
    ROOT,
    SIDES,
    TRACKS,
    N,
    allowed_names,
    read,
    setup,
    sha,
    train_features,
    write,
)
from media_eval_data import Media
from media_gaussian import fit

LOG = logging.getLogger(__name__)


def save(out: Path, key: str, state: dict, params: list) -> str:
    p = out / (key + ".npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        p,
        **{
            k: v.numpy()
            for k, v in state.items()
            if ".net." in k and not k.endswith("num_batches_tracked")
        },
        mean=params[0],
        delta=params[1],
        bias=params[2],
    )
    return sha(p)


def unpack(path: Path, h: str) -> tuple:
    assert sha(path) == h
    with np.load(path, allow_pickle=False) as z:
        return (
            {k: torch.from_numpy(z[k]) for k in z.files if ".net." in k},
            [z[k] for k in ("mean", "delta", "bias")],
        )


def run(training: Path, repeated: Path, out: Path) -> None:
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    setup(out, "冻结.log")
    locked = read(training / "训练前配置.json")
    finished = read(training / "完成.json")
    for prop, file in (
        ("history_sha256", "逐步记录.json"),
        ("members_sha256", "人物清单.json"),
        ("lock_sha256", "训练前配置.json"),
    ):
        assert sha(training / file) == finished[prop]
    hist = read(training / "逐步记录.json")
    members = read(training / "人物清单.json")
    repeat = read(repeated / "逐步记录.json")
    assert locked["study"] == CFG and locked["mode"] == "full" and len(repeat) == 1
    assert all(hist[k] == v for k, v in repeat.items())
    raw = train_features()
    meta = [read(ROOT / REF["metadata"][s][0]) for s in SIDES]
    ids = [np.asarray([r["id"] for r in side[:N]]) for side in meta]
    assert all(
        r["lang"] == "English" and r["src"] == "v4" for rows in meta for r in rows[:N]
    )
    donor = []
    dm = []
    if not locked["transfer"]:
        for mat, side in zip(raw, meta, strict=True):
            ix = [
                i
                for i, r in enumerate(side)
                if r["src"] == "v1" and r["lang"] == "English" and r["video"]
            ]
            donor.append(mat[ix])
            dm.append(([side[i]["id"] for i in ix], [side[i]["video"] for i in ix]))
    keys = {}
    calibration = {}
    for f in locked["folds"]:
        m = members[str(f)]
        names = m["fit"]
        ix = [np.flatnonzero(np.isin(a, names)) for a in ids]
        sub = Media(raw[0][ix[0]], raw[1][ix[1]], ids[0][ix[0]], ids[1][ix[1]])
        groups = [
            (np.flatnonzero(sub.face_ids == n), np.flatnonzero(sub.voice_ids == n))
            for n in names
        ]
        assert set(sub.face_ids) == set(sub.voice_ids) == set(names) and not set(
            names
        ) & set(m["denied"])
        for s in locked["seeds"]:
            tag = f"fold{f}_s{s}"
            allow = set(allowed_names(f, s)["classes"]) - set(m["denied"])
            dg = donor_groups(dm, allow) if donor else []
            calibration[tag] = {
                "official": names,
                "english_donor_count": len(dg[0]) if dg else 0,
            }
            for arm, rec in hist[tag]["arms"].items():
                path = training / arm / (tag + ".pt")
                assert sha(path) == rec["sha256"]
                state = torch.load(path, weights_only=True, map_location="cpu")
                if locked["transfer"]:
                    em = project(sub, state)
                    stat = fit(em[0], em[1], groups, REF)
                    co = [stat["mean"], -stat["precision_delta"], stat["constant"]]
                    k = f"{arm}/{tag}"
                    keys[k] = save(out, k, state, co)
                else:
                    for track in TRACKS:
                        st, co, _ = make_parameters(
                            state, track, sub, groups, donor, dg
                        )
                        k = f"{arm}/{tag}_{track}"
                        keys[k] = save(out, k, st, co)
        LOG.info("已冻结第 %d 折", f)
    write(
        out / "冻结清单.json",
        {
            "params": keys,
            "calibration": calibration,
            "train_sha256": sha(training / "完成.json"),
            "independent_first_sha256": sha(repeated / "完成.json"),
            "members": members,
            "mode": locked["transfer"],
            "methods": list(locked["methods"]),
            "primary": locked.get("primary", CFG["primary"]),
            **(
                {"paired_comparisons": locked["design"]["paired_comparisons"]}
                if "paired_comparisons" in locked.get("design", {})
                else {}
            ),
            "folds": locked["folds"],
            "seeds": locked["seeds"],
            "elapsed": time.monotonic() - start,
        },
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "repeated", "out"):
        ap.add_argument(name, type=Path)
    a = ap.parse_args()
    run(a.training, a.repeated, a.out)
