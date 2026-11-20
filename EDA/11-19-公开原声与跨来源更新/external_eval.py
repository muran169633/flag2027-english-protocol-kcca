"""在同台词同情绪的演员和两次独立录音的孟语作家上比较固定模型。"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from bootstrap import contrasts, intervals
from freeze import unpack
from grouped_gaussian import KEYS, precise_project, single_pairs
from media_common import (
    CFG,
    HERE,
    REF,
    ROOT,
    SIDES,
    TRACKS,
    arrays,
    read,
    setup,
    sha,
    write,
)
from media_eval_data import Media

SPEC = read(HERE / "演员验证配置.json")
LOG = logging.getLogger(__name__)


def corpus(actor_dir: Path) -> tuple[dict, dict, dict, dict, dict]:
    actor_lock = actor_dir / "解码前验证设计.json"
    actor_features = actor_dir / "两拍原始特征.npz"
    rec = read(actor_dir / "演员特征核验.json")
    assert (
        sha(actor_lock) == rec["lock_sha256"]
        and sha(actor_features) == rec["features_sha256"]
    )
    manifest = read(actor_lock)
    assert manifest["config"] == SPEC
    rows = manifest["rows"]
    a = arrays(actor_features)
    ids = np.array([r["id"] for r in rows])
    ds = {"actors": Media(a["face"], a["voice"], ids, ids)}
    spec = read(ROOT / SPEC["bangla_config"])
    src = ROOT / spec["recordings"]
    enc = ROOT / spec["encoding"]
    p = enc / "两份原件特征.npz"
    er = read(enc / "原件特征复核.json")
    assert (
        sha(p) == er["feature_sha256"]
        and sha(src / "第二份录音.json") == er["source_sha256"]
    )
    lock = src / "换录音前锁定.json"
    assert sha(lock) == read(src / "第二份录音.json")["lock_sha256"]
    authors = read(lock)["selected"]
    bn = arrays(p)
    assert list(bn["people"]) == [r["id"] for r in authors]
    for k in spec["recording_order"]:
        ds[k] = Media(bn["face"], bn[k], bn["people"], bn["voices"])
    sex = {r["id"]: r["gender"] for r in rows}
    sex.update(
        {r["id"]: {"her": "f", "his": "m"}[r["catalog_pronoun"]] for r in authors}
    )
    details = {"rows": rows, "portrait": [r["id"] for r in authors if r["primary"]]}
    files = {}
    for label, k, meta, file in [
        (
            "packages",
            "previous_packages",
            "两份包换录音比较.json",
            "两份包逐对分数.npz",
        ),
        (
            "baseline",
            "previous_baseline",
            "同人换录音复核.json",
            "同人换录音逐对分数.npz",
        ),
    ]:
        directory = ROOT / spec[k]
        assert sha(directory / file) == read(directory / meta)["score_sha256"]
        files[label] = arrays(directory / file)
    return (
        ds,
        sex,
        details,
        files,
        {
            "actor_lock": sha(actor_lock),
            "actors": sha(actor_features),
            "writers": sha(p),
            "writer_lock": sha(lock),
        },
    )


def banks(
    track: str, frozen: Path, lock: dict, packages: dict, prior: dict, base: dict
) -> Iterator[tuple[str, dict, list]]:
    group = ROOT / CFG["pretrain_run"]
    code = TRACKS[track]
    for f in lock["folds"]:
        for s in lock["seeds"]:
            tag = f"fold{f}_s{s}"
            stem = tag + "_" + track
            key = "评分参数/base/" + stem + ".npz"
            yield "fixed", *unpack(group / key, base[key])
            for name in lock["methods"]:
                key = name + "/" + stem
                yield name, *unpack(frozen / (key + ".npz"), lock["params"][key])
            key = "legacy/teacher/" + tag
            yield (
                "v428",
                *unpack(
                    ROOT / packages["frozen"] / (key + ".npz"), prior["parameters"][key]
                ),
            )
    for filename, dig in prior["packaged"][track].items():
        path = ROOT / filename
        assert sha(path) == dig
        with np.load(path, allow_pickle=False) as z:
            assert np.isclose(z["epsilon"], REF["bn_epsilon"])
            for f in lock["folds"]:
                for s in lock["seeds"]:
                    tag = f"s{s}_f{f}"
                    assert tag in z["tags"]
                    st = {
                        f"{side}.net.{k}": torch.from_numpy(z[f"{tag}.{side}.{k}"])
                        for side in SIDES
                        for k in KEYS
                    }
                    yield (
                        "current",
                        st,
                        [
                            z[tag + ".mean"],
                            z[f"{tag}.{code}.delta"],
                            z[f"{tag}.{code}.bias"],
                        ],
                    )


def grid(ds: Media, st: dict, co: list, check: bool) -> tuple[np.ndarray, float]:
    face, voice = precise_project(ds, st)
    mu, delta, bias = co
    dim = face.shape[1]
    f, v = face - mu[:dim], voice - mu[dim:]
    sf = np.einsum("ij,ij->i", f @ delta[:dim, :dim], f)
    sv = np.einsum("ij,ij->i", v @ delta[dim:, dim:], v)
    values = (
        0.5
        * (
            sf[:, None]
            + sv[None, :]
            + f @ (delta[:dim, dim:] + delta[dim:, :dim].T) @ v.T
        )
        + bias
    )
    err = 0.0
    if check:
        i = np.linspace(0, len(f) - 1, 5, dtype=int)
        j = np.linspace(len(v) - 1, 0, 5, dtype=int)
        q = np.column_stack((i, j, ds.face_ids[i] == ds.voice_ids[j]))
        err = float(abs(values[i, j] - single_pairs(st, co, ds, q)).max())
        assert err < 2e-9
    return values, err


def trials(
    domain: str, condition: str, ds: Media, data: dict, sex: dict, track: str
) -> tuple[np.ndarray, np.ndarray]:
    i, j = np.indices((len(ds.face), len(ds.voice)))
    keep = np.ones_like(i, dtype=bool)
    if domain == "actors":
        emotion = np.array([r["emotion"] for r in data["rows"]])
        text = np.array([r["sentence"] for r in data["rows"]])
        keep &= (emotion[i] != emotion[j]) == (
            condition in ("change_emotion", "change_both")
        )
        keep &= (text[i] != text[j]) == (
            condition in ("change_statement", "change_both")
        )
    elif condition == "independent_portrait":
        keep &= np.isin(ds.face_ids[i], data["portrait"]) & np.isin(
            ds.voice_ids[j], data["portrait"]
        )
    if track == "same_gender":
        sx = np.array([sex[v] for v in ds.face_ids])
        sy = np.array([sex[v] for v in ds.voice_ids])
        keep &= sx[i] == sy[j]
    i, j = i[keep], j[keep]
    y = ds.face_ids[i] == ds.voice_ids[j]
    g = 2 * i + y
    reps = np.bincount(g)
    if domain == "actors":
        assert len(np.unique(reps[::2])) == len(np.unique(reps[1::2])) == 1
        assert reps[0] == reps[1] * (11 if track == "same_gender" else 23)
    return np.column_stack((i, j, y)).astype(int), 1 / reps[g]


def comparisons(
    point: np.ndarray, draws: np.ndarray, models: list[str], pairs: list | None = None
) -> dict:
    if pairs is None:
        pairs = [
            ("current", "v428"),
            ("meta", "baseline"),
            ("meta", "stop"),
            ("meta", "joint"),
            ("meta", "permuted"),
            ("meta", "current"),
            ("meta", "v428"),
        ]
    return contrasts(point, draws, models, pairs)


def run(actors: Path, frozen: Path, out: Path) -> None:
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    setup(out, "固定拍次及录音复核.log")
    corpora, sex, info, old, hashes = corpus(actors)
    lock = read(frozen / "冻结清单.json")
    packages = read(ROOT / CFG["release_cfg"])
    p = ROOT / packages["frozen"] / "评分前全部冻结.json"
    assert (
        sha(p)
        == read(ROOT / CFG["reference_run"] / "跨日与原条件结果.json")["frozen_sha256"]
    )
    prior = read(p)
    assert prior["folds"] == lock["folds"] and prior["seeds"] == lock["seeds"]
    current = read(ROOT / packages["current_record"])
    teacher = read(ROOT / prior["spec"]["historical_package"] / "result.json")
    assert (
        sha(ROOT / current["package"]) == current["sha256"]
        and sha(ROOT / teacher["zip"]) == teacher["sha256"]
    )
    for tr, code in TRACKS.items():
        assert prior["packaged"][tr] == current["frozen_parameters"][code]
    design = out / "评分前来源与模型.json"
    write(
        design,
        {
            "corpora": hashes,
            "native_parameters_sha256": sha(frozen / "冻结清单.json"),
            "historical_parameters_sha256": sha(p),
            "packages": {"current": current["sha256"], "v428": teacher["sha256"]},
            "actors": SPEC,
            "bootstrap": CFG["bootstrap_draws"],
            "seed": CFG["seed"],
        },
    )
    fixed = read(ROOT / CFG["pretrain_run"] / "同协议成绩.json")["parameters"]
    names = lock["methods"] + SPEC["packaged_models"]
    summary, checks, overlap, saved = {}, {}, {}, {}
    for tr in TRACKS:
        sums = {}
        counts = dict.fromkeys(["fixed", *names], 0)
        for name, state, coeff in banks(tr, frozen, lock, packages, prior, fixed):
            for dom, ds in corpora.items():
                vals, err = grid(ds, state, coeff, counts[name] == 0)
                if counts[name] == 0:
                    checks[f"{tr}/{name}/{dom}"] = err
                    sums[name, dom] = vals
                else:
                    sums[name, dom] += vals
            counts[name] += 1
        assert all(
            v == len(lock["folds"]) * len(lock["seeds"]) * (2 if k == "current" else 1)
            for k, v in counts.items()
        )
        for dom, ds in corpora.items():
            score = {}
            for n in names:
                x = sums[n, dom] / counts[n]
                if n in lock["methods"]:
                    x = 0.5 * (x + sums["fixed", dom] / counts["fixed"])
                score[n] = x
                saved[f"{tr}__{dom}__{n}"] = x
            for cond in (
                SPEC["conditions"]
                if dom == "actors"
                else ["all", "independent_portrait"]
            ):
                q, w = trials(dom, cond, ds, info, sex, tr)
                key = f"{dom}__{cond}__{tr}"
                i, j, y = q.T
                traces = [score[n][i, j] for n in names]
                if dom != "actors":
                    for n in ["baseline", *SPEC["packaged_models"]]:
                        prev = (
                            old["baseline"][f"{cond}__{tr}__{dom}__{n}"]
                            if n == "baseline"
                            else old["packages"][f"{n}__{dom}__{cond}__{tr}"]
                        )
                        err = float(abs(prev - score[n][i, j]).max())
                        assert err < (1e-4 if n == "v428" else 2e-9), (n, err)
                        overlap[key + "__" + n] = err
                point, draws = intervals(
                    q, w, ds.face_ids, ds.voice_ids, sex, traces, CFG["seed"]
                )
                summary[key] = {
                    "positive": int(y.sum()),
                    "people": len(set(ds.face_ids[i])),
                    "trials": len(q),
                    "rows_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
                    "eer": dict(zip(names, point.tolist(), strict=True)),
                    "paired": comparisons(
                        point, draws, names, lock.get("paired_comparisons")
                    ),
                }
                saved[key + "__q"], saved[key + "__weights"] = q, w
            LOG.info("完成 %s %s", tr, dom)
    dest = out / "演员及作家逐对分数.npz"
    np.savez_compressed(dest, **saved)
    write(
        out / "两份包与新训练复核.json",
        {
            "cells": summary,
            "single_pair_errors": checks,
            "archive_errors": overlap,
            "design_sha256": sha(design),
            "scores_sha256": sha(dest),
            "seconds": time.monotonic() - start,
        },
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("actors", "frozen", "out"):
        cli.add_argument(k, type=Path)
    a = cli.parse_args()
    run(a.actors, a.frozen, a.out)
