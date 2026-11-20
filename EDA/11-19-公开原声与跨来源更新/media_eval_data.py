"""还原各来源原时长配对、权重与当前候选预测。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from media_common import (
    EC,
    LOC,
    REF,
    ROOT,
    STUDY,
    N,
    arrays,
    feature,
    official_meta,
    read,
    sha,
    train_features,
)


@dataclass
class Media:
    face: np.ndarray
    voice: np.ndarray
    face_ids: np.ndarray
    voice_ids: np.ndarray


@dataclass
class Cell:
    domain: str
    track: str
    q: np.ndarray
    weight: np.ndarray
    parent: np.ndarray | None
    folds: tuple[int, ...]


def load() -> tuple[dict[str, Media], dict[str, Cell], list, list]:
    official, meta = official_meta()
    matrices = train_features()
    all_rows = read(STUDY / "媒体索引.json")
    a = arrays(STUDY / "逐对核验.npz")
    sources = {}
    names = np.array([r["id"] for r in official])
    sources["flag"] = Media(matrices[0][:N], matrices[1][:N], names, names)
    donor, donors_meta = [], []
    for j, bank in enumerate(meta):
        ix = [
            i
            for i, r in enumerate(bank)
            if r["src"] == "v1" and r["lang"] == "English" and r["video"]
        ]
        donor.append(matrices[j][ix])
        donors_meta.append(
            ([bank[i]["id"] for i in ix], [bank[i]["video"] for i in ix])
        )
    folder = ROOT / EC["crema"]
    face = feature([(folder / "face_v399.npz", k) for k in ("clip", "clipL")])
    core = [
        (folder / "voice_core_v399.npz", k)
        for k in ("wavlm", "ecapa", "eres", "resnet")
    ]
    attrs = [(folder / "voice_attrs_v399.npz", k) for k in ("whisper", "age6")]
    names = np.array([r["id"] for r in all_rows["crema"]])
    sources["crema"] = Media(face, feature(core + attrs), names, names)
    bn = ROOT / EC["bangla"] / "LRW-B"
    lock = read(bn / "同录音两秒复核.json")
    assert sha(bn / "同录音两秒特征.npz") == lock["features_sha256"]
    z = arrays(bn / "同录音两秒特征.npz")
    keys = {tuple(row): i for i, row in enumerate(lock["axes"])}
    group = sorted({r["id"] for r in all_rows["bangla"]})
    for domain in ("english", "bangla"):
        rows = all_rows[domain]
        ix = [
            keys[
                domain,
                int(r["id"]) if domain == "english" else group.index(r["id"]),
                r["day"],
            ]
            for r in rows
        ]
        names = np.array([r["id"] for r in rows])
        sources[domain] = Media(z["face"][ix], z["fresh"][ix], names, names)
    loc = read(LOC / "评分前锁定.json")
    path = LOC / "冻结匹配特征.npz"
    assert sha(path) == read(LOC / "特征复放.json")["features_sha256"]
    x = arrays(path)
    packed_loc = arrays(LOC / "冻结逐对分数.npz")
    people = np.array([r["id"] for r in loc["people"]])
    vm = np.array([r["person"] for r in loc["clips"]])
    primary = np.array([r["primary"] for r in loc["people"]])
    sources["loc"] = Media(x["face"], x["full"], people, people[vm])
    cells = {}
    folds = tuple(REF["folds"])
    for track in REF["protocols"]:
        for f in folds:
            k = f"fold{f}_{track}"
            q = a[k + "_trials"]
            cells[k] = Cell(
                "flag", track, q, np.ones(len(q)), a[k + "_full_parent"], (f,)
            )
        for domain in ("crema", "english", "bangla"):
            k = domain + "_" + track + "_full"
            cells[domain + "_" + track] = Cell(
                domain,
                track,
                a[k + "_trials"],
                a[k + "_weights"],
                a[k + "_parent"],
                folds,
            )
        q = np.asarray(loc["trials"][track])
        for dom, keep in (
            ("loc", np.ones(len(q), bool)),
            ("loc_primary", primary[q[:, 0]] & primary[vm[q[:, 1]]]),
        ):
            qs = q[keep]
            lab = 2 * qs[:, 0] + qs[:, 2]
            cells[dom + "_" + track] = Cell(
                "loc",
                track,
                qs,
                1 / np.bincount(lab)[lab],
                packed_loc[track + "_full_baseline"][keep],
                folds,
            )
    for c in cells.values():
        ds = sources[c.domain]
        np.testing.assert_array_equal(
            ds.face_ids[c.q[:, 0]] == ds.voice_ids[c.q[:, 1]], c.q[:, 2]
        )
        assert (
            c.parent is not None
            and len(c.q) == len(c.parent) == len(c.weight)
            and np.all(c.weight > 0)
        )
    return sources, cells, donor, donors_meta
