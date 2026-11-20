"""核对每一对的身份、语言、录制和权重，读取冻结的外部配对。"""

from __future__ import annotations

import csv

import numpy as np
from bootstrap import genders
from media_common import (
    CFG,
    REF,
    ROOT,
    SIDES,
    TRACKS,
    arrays,
    read,
    sha,
    train_features,
)
from media_eval_data import Cell, Media, load


def development() -> tuple:
    data, cs, _, _ = load()
    cells = {
        k: v
        for k, v in cs.items()
        if not k.startswith(("loc_standard", "loc_same_gender"))
    }
    p = ROOT / CFG["sf22_media"]
    z = ROOT / CFG["sf22_features"]
    features = z / "外部测试特征.npz"
    manifest = read(p / "外部测试媒体.json")
    lock = read(p / "外部测试评分前锁定.json")
    feature_check = read(z / "外部测试编码核对.json")
    assert (
        sha(p / "外部测试评分前锁定.json") == manifest["lock_sha256"]
        and sha(p / "外部测试媒体.json") == feature_check["manifest_sha256"]
        and sha(features) == feature_check["features_sha256"]
    )
    a = arrays(features)
    ids = np.array([str(r["subject"]) for r in manifest["rows"]])
    ix = {(r["subject"], r["day"]): i for i, r in enumerate(manifest["rows"])}
    assert list(ix) == list(map(tuple, a["ids"].tolist()))
    data["sf_test"] = Media(a["face"], a["voice"], ids, ids)
    sex = genders()
    sex.update({str(r["Sub_ID"]): r["Gender"][0].lower() for r in lock["subjects"]})
    folds = tuple(REF["folds"])
    for tr, rows in lock["pairs"].items():
        q = np.array([(ix[n, di], ix[m, dj], int(y)) for n, di, m, dj, y, _ in rows])
        assert all(
            di != dj and (tr == "standard" or sex[str(n)] == sex[str(m)])
            for n, di, m, dj, _, _ in rows
        )
        cells["sf_test_" + tr] = Cell(
            "sf_test", tr, q, np.array([r[-1] for r in rows]), None, folds
        )
    p = ROOT / CFG["public_source"]
    info = read(p / "评分前人物与来源.json")
    qs = read(p / "新发言人留出配对.json")
    assert (
        sha(p / "评分前人物与来源.json") == qs["source_sha256"]
        and sha(p / "公开图声特征.npz") == info["features_sha256"]
    )
    a = arrays(p / "公开图声特征.npz")
    data["new_english"] = Media(a["face"], a["voice"], a["face_ids"], a["voice_ids"])
    sex.update({r["person"]: r["gender"] for r in info["faces"]})
    for tr, v in qs["cells"].items():
        q = np.array(v["q"])
        assert (
            set(a["face_ids"][q[:, 0]])
            == set(a["voice_ids"][q[:, 1]])
            == set(qs["held"])
        )
        cells["new_english_" + tr] = Cell(
            "new_english", tr, q, np.array(v["weight"]), None, folds
        )
    last = ROOT / CFG["reference_run"]
    old = last / "所有固定条件逐对分数.npz"
    assert sha(old) == read(last / "跨日与原条件结果.json")["score_sha256"]
    prev = ROOT / CFG["pretrain_run"]
    main = prev / "全部逐对分数.npz"
    assert sha(main) == read(prev / "同协议成绩.json")["score_sha256"]
    bases = {}
    refs = {}
    with (
        np.load(old, allow_pickle=False) as oldpack,
        np.load(main, allow_pickle=False) as prior,
    ):
        for k, c in cells.items():
            ds = data[c.domain]
            np.testing.assert_array_equal(
                ds.face_ids[c.q[:, 0]] == ds.voice_ids[c.q[:, 1]], c.q[:, 2]
            )
            arc, system = (
                (prior, "blend_original")
                if c.domain == "new_english"
                else (oldpack, "blend_both")
            )
            bases[k] = arc["base__" + k]
            refs[k] = arc[system + "__" + k]
    return data, cells, sex, bases, refs


def transfer(members: dict) -> tuple:
    src = ROOT / CFG["transfer_run"]
    p = src / "冻结试验与逐对预测.npz"
    spec = read(src / "实际配置.json")["study"]
    manifest = read(ROOT / spec["v1_manifest"])
    assert (
        sha(ROOT / spec["v1_trials"]) == read(src / "隔离审计.json")["v1_frozen_sha256"]
    )
    assert all(sha(ROOT / k) == v for k, v in manifest["source_hashes"].items())
    metadata = [read(ROOT / REF["metadata"][s][0]) for s in SIDES]
    raw = train_features()
    sexes = {}
    with (ROOT / spec["v1_gender"]).open(encoding="utf-8-sig") as f:
        sexes.update(
            {"v1_" + r["ids"]: r["gender"].lower()[0] for r in csv.DictReader(f)}
        )
    data = {}
    cells = {}
    with (
        np.load(p, allow_pickle=False) as bank,
        np.load(ROOT / spec["v1_trials"], allow_pickle=False) as unmodified,
    ):
        for source, langs in (
            ("v1", ("English", "Urdu")),
            ("v2", ("English", "Hindi")),
        ):
            pool = [
                np.asarray([i for i, r in enumerate(rows) if r["src"] == source])
                for rows in metadata
            ]
            ids = [
                np.array([rows[i]["id"] for i in ix])
                for rows, ix in zip(metadata, pool, strict=True)
            ]
            visible = set(np.concatenate(ids))
            held = {REF["canonical_alias"].get(n, n) for n in visible}
            for m in members.values():
                assert not held.intersection(
                    set(m["pre"]) | set(m["fit"]) | set(m["adapt"])
                )
            data[source] = Media(raw[0][pool[0]], raw[1][pool[1]], ids[0], ids[1])
            if source == "v2":
                sexes.update(dict.fromkeys(visible, "all"))
            for tr in TRACKS if source == "v1" else ("standard",):
                for lang in langs:
                    k = f"{source}_{lang}_{tr}"
                    orig = bank["q_" + k]
                    w = bank["w_" + k]
                    q = orig.copy()
                    if source == "v1":
                        np.testing.assert_array_equal(
                            orig, unmodified[lang + "_" + tr + "_pairs"]
                        )
                        np.testing.assert_array_equal(
                            w, unmodified[lang + "_" + tr + "_weights"]
                        )
                    for j, ix in enumerate(pool):
                        q[:, j] = np.searchsorted(ix, orig[:, j])
                        np.testing.assert_array_equal(ix[q[:, j]], orig[:, j])
                    np.testing.assert_array_equal(
                        ids[0][q[:, 0]] == ids[1][q[:, 1]], q[:, 2]
                    )
                    assert all(
                        metadata[0][i]["lang"] == "English"
                        and metadata[1][j]["lang"] == lang
                        and (
                            not y or metadata[0][i]["video"] != metadata[1][j]["video"]
                        )
                        for i, j, y in orig
                    )
                    if tr == "same_gender":
                        assert all(
                            sexes[ids[0][i]] == sexes[ids[1][j]] for i, j, _ in q
                        )
                    cells[k] = Cell(source, tr, q, w, None, tuple(REF["folds"]))
                c, d = (cells[f"{source}_{lang}_{tr}"] for lang in langs)
                np.testing.assert_array_equal(c.q[:, [0, 2]], d.q[:, [0, 2]])
                np.testing.assert_array_equal(ids[1][c.q[:, 1]], ids[1][d.q[:, 1]])
                np.testing.assert_array_equal(c.weight, d.weight)
    return data, cells, sexes, {}, {}
