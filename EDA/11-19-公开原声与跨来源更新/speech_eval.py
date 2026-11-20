"""冻结模型在普通与 Lombard 发声下逐对评分，逐人检验更换一端录制的影响。"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
from actors import encode_media, extract_pair
from bootstrap import contrasts, intervals
from external_eval import banks, grid
from identity_update import SPEC
from media_common import CFG, HERE, PC, ROOT, TRACKS, arrays, read, setup, sha, write
from media_eval_data import Media

from exps.v066.face import FaceEncoder

SRC = read(HERE / "发声条件验证配置.json")
LOCK = SPEC["speech_source"]
LOG = logging.getLogger(__name__)


@torch.inference_mode()
def encode(source: Path, out: Path, frontal_camera: bool = False) -> None:
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    setup(out, "两种发声编码.log")
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lock = source / "人物与来源预定.json"
    material = source / "两种发声原件.json"
    meta = read(material)
    assert sha(lock) == meta["lock_sha256"] and read(lock)["spec"] == SRC
    rows = read(lock)["rows"]
    table = {
        (r["person"], r["side"], r["slot"], r["style"]): r for r in meta["records"]
    }
    assert (
        len(table) == len(meta["records"]) and LOCK["face_slots"] == LOCK["voice_slots"]
    )
    dest = out / "逐拍前端输入"
    dest.mkdir()
    detector = (
        None
        if frontal_camera
        else FaceEncoder(
            {
                **read(ROOT / "exps/v066/config.json"),
                "cpu_threads": PC["resource_cpu_threads"],
            }
        )
    )
    write(
        out / "编码前取景.json",
        {
            "rule": "全部视频中间帧按短边中心方形取景"
            if frontal_camera
            else "每段视频逐帧按固定策略查找单人脸框",
            "source_sha256": sha(material),
        },
    )
    entries, images, waves, skipped = [], [], [], {}
    for row in rows:
        person = row["person"]
        if (person, "face", 0, LOCK["plain"]) not in table:
            continue
        person_entries = []
        for slot in range(LOCK["face_slots"]):
            for style in (LOCK["plain"], LOCK["lombard"]):
                face = table[person, "face", slot, style]
                voice = table[person, "voice", slot, style]
                pair = {
                    "face": face["path"],
                    "voice": voice["path"],
                    "face_sha256": face["sha256"],
                    "voice_sha256": voice["sha256"],
                }
                stem = dest / f"{person}_{slot}_{style}"
                try:
                    decoded = extract_pair(pair, stem, detector)
                except ValueError as exc:
                    skipped[person] = str(exc)
                    break
                wave, rate = sf.read(stem.with_suffix(".wav"), dtype="float32")
                spectrum = np.abs(np.fft.rfft(wave)) ** 2
                freq = np.fft.rfftfreq(len(wave), d=1 / rate)
                e = {
                    "person": person,
                    "gender": row["gender"],
                    "cohort": row["cohort"],
                    "slot": slot,
                    "style": style,
                    "face_word": face["word"],
                    "voice_word": voice["word"],
                    "level_dbfs": float(10 * np.log10(np.mean(wave**2))),
                    "high_frequency_fraction": float(
                        spectrum[freq >= 2000].sum() / spectrum.sum()
                    ),
                    "decoded": decoded,
                }
                person_entries.append((e, stem))
            if person in skipped:
                break
        if person in skipped:
            continue
        assert len(person_entries) == LOCK["face_slots"] * 2
        for e, stem in person_entries:
            entries.append(e)
            images.append(stem.with_suffix(".png"))
            waves.append(stem.with_suffix(".wav"))
    del detector
    assert len(entries) >= 24 and not {r["face_word"] for r in entries} & {
        r["voice_word"] for r in entries
    }
    data, errors = encode_media(images, waves, out)
    fp = out / "两种发声特征.npz"
    np.savez_compressed(fp, face=data["face"], voice=data["voice"])
    write(
        out / "双发声编码复核.json",
        {
            "source_sha256": sha(material),
            "lock_sha256": sha(lock),
            "crop": "camera" if frontal_camera else "detected",
            "features_sha256": sha(fp),
            "rows": entries,
            "skipped": skipped,
            "controls": errors,
            "seconds": time.monotonic() - start,
        },
    )
    LOG.info(
        "完成 %d 位人物的 %d 个配平槽位，排除 %d 位",
        len({r["person"] for r in entries}),
        len(entries),
        len(skipped),
    )


def queries(
    rows: list, cohort: str, track: str, face_style: str, voice_style: str
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.array([r["person"] for r in rows])
    sx = np.array([r["gender"] for r in rows])
    style = np.array([r["style"] for r in rows])
    group = np.array([r["cohort"] == cohort for r in rows])
    if track == "same_gender":
        allowed = [
            g for g in set(sx[group]) if len(np.unique(ids[group & (sx == g)])) >= 2
        ]
        group &= np.isin(sx, allowed)
    ix = np.flatnonzero(group & (style == face_style))
    jx = np.flatnonzero(group & (style == voice_style))
    np.testing.assert_array_equal(ids[ix], ids[jx])
    i, j = np.meshgrid(ix, jx, indexing="ij")
    ok = np.ones_like(i, dtype=bool) if track == "standard" else sx[i] == sx[j]
    i, j = i[ok], j[ok]
    y = ids[i] == ids[j]
    g = 2 * i + y
    return np.column_stack((i, j, y)).astype(int), 1 / np.bincount(g)[g]


def ranking(q: np.ndarray, scores: np.ndarray) -> float:
    a, b, y = q.T
    per = []
    for i in np.unique(a):
        where = a == i
        pos = scores[i, b[where & (y == 1)]]
        neg = scores[i, b[where & (y == 0)]]
        v = pos[:, None] - neg[None, :]
        per.append(np.mean((v > 0) + 0.5 * (v == 0)))
    return float(100 * np.mean(per))


def score(encoded: Path, frozen: Path, out: Path) -> None:
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    setup(out, "发声条件逐对评估.log")
    stamp = encoded / "双发声编码复核.json"
    meta = read(stamp)
    fp = encoded / "两种发声特征.npz"
    assert sha(fp) == meta["features_sha256"]
    a = arrays(fp)
    rows = meta["rows"]
    ids = np.array([r["person"] for r in rows])
    sx = {r["person"]: r["gender"] for r in rows}
    ds = Media(a["face"], a["voice"], ids, ids)
    locked = read(frozen / "冻结清单.json")
    packages = read(ROOT / CFG["release_cfg"])
    pp = ROOT / packages["frozen"] / "评分前全部冻结.json"
    prior = read(pp)
    base = read(ROOT / CFG["pretrain_run"] / "同协议成绩.json")["parameters"]
    old = read(out.parent / SPEC["old_external"] / "评分前来源与模型.json")
    assert old["historical_parameters_sha256"] == sha(pp)
    assert (
        locked["primary"] == SPEC["primary"]
        and locked["paired_comparisons"] == SPEC["paired_comparisons"]
    )
    names = locked["methods"] + ["current", "v428"]
    design = out / "评分前来源与参数.json"
    write(
        design,
        {
            "feature_lock": sha(stamp),
            "model_lock": sha(frozen / "冻结清单.json"),
            "packages": old["packages"],
            "historical": sha(pp),
            "arms": names,
            "source_design": SRC,
            "training_design": SPEC,
            "bootstrap": CFG["bootstrap_draws"],
            "seed": CFG["seed"],
        },
    )
    cells, saved, checks, shifts = {}, {}, {}, {}
    for tr in TRACKS:
        sums = {}
        counts = dict.fromkeys(["fixed", *names], 0)
        for name, state, co in banks(tr, frozen, locked, packages, prior, base):
            s, err = grid(ds, state, co, counts[name] == 0)
            if counts[name] == 0:
                sums[name] = s
                checks[tr + "__" + name] = err
            else:
                sums[name] += s
            counts[name] += 1
        for n in counts:
            assert counts[n] == len(locked["folds"]) * len(locked["seeds"]) * (
                2 if n == "current" else 1
            )
            sums[n] /= counts[n]
        for name in names:
            if name in locked["methods"]:
                sums[name] = 0.5 * (sums[name] + sums["fixed"])
            saved[tr + "__" + name] = sums[name]
        for ci, cohort in enumerate(SRC["roles"]):
            group = {r["person"] for r in rows if r["cohort"] == cohort}
            if tr == "same_gender":
                group = {p for p in group if sum(sx[n] == sx[p] for n in group) >= 2}
            if len(group) < 2:
                continue
            draws, point = {}, {}
            signature = None
            for face_style in (LOCK["plain"], LOCK["lombard"]):
                for voice_style in (LOCK["plain"], LOCK["lombard"]):
                    key = f"{cohort}__{tr}__{face_style}{voice_style}"
                    q, w = queries(rows, cohort, tr, face_style, voice_style)
                    i, j, y = q.T
                    subjects = [
                        (
                            rows[f]["person"],
                            rows[f]["slot"],
                            rows[v]["person"],
                            rows[v]["slot"],
                            bool(y0),
                            wt,
                        )
                        for (f, v, y0), wt in zip(q, w, strict=True)
                    ]
                    if signature is None:
                        signature = subjects
                    else:
                        assert signature == subjects
                    vls = [sums[n][i, j] for n in names]
                    pt, b = intervals(q, w, ids, ids, sx, vls, CFG["seed"] + ci)
                    cell = face_style + voice_style
                    draws[cell], point[cell] = b, pt
                    cells[key] = {
                        "people": len(group),
                        "gender_counts": {
                            k: sum(sx[p] == k for p in group) for k in ("m", "f")
                        },
                        "trials": len(q),
                        "positive": int(y.sum()),
                        "eer": dict(zip(names, pt.tolist(), strict=True)),
                        "rank": {n: ranking(q, sums[n]) for n in names},
                        "paired": contrasts(pt, b, names, SPEC["paired_comparisons"]),
                    }
                    saved[key + "__q"], saved[key + "__weights"] = q, w
            for left, right, what in (
                ("pl", "pp", "voice"),
                ("lp", "pp", "face"),
                ("ll", "pp", "both"),
            ):
                vals = {}
                for idx, n in enumerate(names):
                    d = draws[left][:, idx] - draws[right][:, idx]
                    good = d[np.isfinite(d)]
                    vals[n] = {
                        "pp": float(point[left][idx] - point[right][idx]),
                        "ci95": np.quantile(good, [0.025, 0.975]).tolist(),
                        "usable": len(good),
                    }
                shifts[f"{cohort}__{tr}__{what}"] = vals
            LOG.info("完成 %s %s，共 %d 位人物", cohort, tr, len(group))
    target = out / "发声变化逐对预测.npz"
    np.savez_compressed(target, **saved)
    write(
        out / "双发声验证结果.json",
        {
            "cells": cells,
            "within_person_style_change": shifts,
            "single_pair": checks,
            "scores_sha256": sha(target),
            "lock_sha256": sha(design),
            "seconds": time.monotonic() - start,
        },
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="mode", required=True)
    dec = sub.add_parser("encode")
    dec.add_argument("source", type=Path)
    dec.add_argument("out", type=Path)
    dec.add_argument("--frontal-camera", action="store_true")
    ev = sub.add_parser("score")
    for k in ("encoded", "frozen", "out"):
        ev.add_argument(k, type=Path)
    args = cli.parse_args()
    if args.mode == "encode":
        encode(args.source, args.out, args.frontal_camera)
    else:
        score(args.encoded, args.frozen, args.out)
