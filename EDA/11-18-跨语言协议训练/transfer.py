"""全程留出历史英乌、英印来源，比较英语身份扩充和协议条件损失。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_curve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "11-18-Vox英语身份扩充"))
from train import SIDES, Pools, load_sources, read, save, stage

from exps.v480.run import embed, fit

LOG = logging.getLogger(__name__)
Meta = list[list[dict[str, str]]]


def hashfile(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def genders(cfg: dict[str, Any]) -> dict[str, str]:
    with (ROOT / cfg["v4_gender"]).open(encoding="utf-8-sig") as fh:
        official = {"v4_" + n: g.lower() for n, g in csv.reader(fh)}
    with (ROOT / cfg["v1_gender"]).open(encoding="utf-8-sig") as fh:
        native = {"v1_" + r["ids"]: r["gender"].lower()[0] for r in csv.DictReader(fh)}
    with (ROOT / cfg["vox_gender"]).open(encoding="utf-8-sig") as fh:
        rows = csv.reader(fh)
        next(rows)
        vox = {"vox2t_" + r[0].strip(): r[2].strip().lower() for r in rows}
    combined = {**official, **native, **vox}
    assert set(combined.values()) == {"m", "f"}
    return combined


def hindi_trials(
    meta: Meta, cfg: dict[str, Any], ref: dict[str, Any]
) -> dict[str, np.ndarray]:
    people = sorted(
        {r["id"] for r in meta[0] if r["src"] == "v2"} - set(ref["canonical_alias"])
    )
    by_media: list[dict[tuple[str, str], dict[str, list[int]]]] = []
    for records in meta:
        media: dict[tuple[str, str], dict[str, list[int]]] = {}
        for i, r in enumerate(records):
            if r["src"] == "v2" and r["video"]:
                media.setdefault((r["id"], r["lang"]), {}).setdefault(
                    r["video"], []
                ).append(i)
        by_media.append(media)
    rng = np.random.default_rng(cfg["trial_seed"])
    repeats = cfg["v2_repeats"]
    selected = np.empty((len(people), repeats, 3), dtype=np.int64)
    for i, n in enumerate(people):
        used: list[set[str]] = [set(), set(), set()]
        for rep in range(repeats):
            face_video = ""
            for j, (side, lang) in enumerate(
                ((0, "English"), (1, "English"), (1, "Hindi"))
            ):
                groups = by_media[side][(n, lang)]
                allowed = sorted(set(groups) - used[j] - {face_video})
                assert allowed
                vid = str(rng.choice(allowed))
                selected[i, rep, j] = rng.choice(groups[vid])
                used[j].add(vid)
                if j == 0:
                    face_video = vid
    all_pairs = []
    for i, name in enumerate(people):
        for j, other in enumerate(people):
            for rep in range(repeats):
                a = selected[i, rep, 0]
                b, c = selected[j, rep, 1:]
                if meta[0][a]["video"] in (meta[1][b]["video"], meta[1][c]["video"]):
                    assert i != j
                    continue
                all_pairs.append((a, b, c, int(name == other)))
    q = np.array(all_pairs)
    ids = np.array([meta[0][i]["id"] for i in q[:, 0]])
    w = np.zeros(len(q))
    for n in people:
        for y in (0, 1):
            m = (ids == n) & (q[:, 3] == y)
            w[m] = 1 / (2 * len(people) * int(m.sum()))
    return {
        "English_pairs": q[:, [0, 1, 3]],
        "Hindi_pairs": q[:, [0, 2, 3]],
        "weights": w,
    }


def locked_trials(
    historical: Meta,
    cfg: dict[str, Any],
    native: dict[str, str],
    ref: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    trials = {}
    weights = {}
    manifest = read(ROOT / cfg["v1_manifest"])
    assert all(
        hashfile(ROOT / path) == expected
        for path, expected in manifest["source_hashes"].items()
    )
    with np.load(ROOT / cfg["v1_trials"]) as fixed:
        for p in ("standard", "same_gender"):
            q1, q2 = [
                fixed[lang + "_" + p + "_pairs"].astype(int)
                for lang in ("English", "Urdu")
            ]
            np.testing.assert_array_equal(q1[:, [0, 2]], q2[:, [0, 2]])
            assert {historical[0][a]["id"] for a in q1[:, 0]} == set(
                manifest["supported_ids"]
            )
            for lang, q in (("English", q1), ("Urdu", q2)):
                assert all(
                    historical[0][a]["src"] == "v1"
                    and historical[0][a]["lang"] == "English"
                    and historical[1][b]["lang"] == lang
                    and ((historical[0][a]["id"] == historical[1][b]["id"]) == y)
                    and (
                        not y or historical[0][a]["video"] != historical[1][b]["video"]
                    )
                    for a, b, y in q
                )
                if p == "same_gender":
                    assert all(
                        native[historical[0][a]["id"]] == native[historical[1][b]["id"]]
                        for a, b, _ in q
                    )
                key = "v1_" + lang + "_" + p
                trials[key] = q
                weights[key] = fixed[lang + "_" + p + "_weights"]
            assert all(
                historical[1][b]["id"] == historical[1][c]["id"]
                for b, c in zip(q1[:, 1], q2[:, 1], strict=True)
            )
            np.testing.assert_array_equal(
                weights["v1_English_" + p], weights["v1_Urdu_" + p]
            )
    h = hindi_trials(historical, cfg, ref)
    for lang in ("English", "Hindi"):
        q = h[lang + "_pairs"]
        for a, b, y in q:
            assert (
                historical[0][a]["src"] == historical[1][b]["src"] == "v2"
                and historical[0][a]["lang"] == "English"
            )
            assert (
                historical[1][b]["lang"] == lang
                and historical[0][a]["video"] != historical[1][b]["video"]
            )
            assert (historical[0][a]["id"] == historical[1][b]["id"]) == y
        key = "v2_" + lang + "_standard"
        trials[key] = q
        weights[key] = h["weights"]
    return trials, weights


def eer(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> float:
    assert np.isfinite(s).all() and np.all(w > 0)
    fp, tp, _ = roc_curve(y, s, sample_weight=w, drop_intermediate=False)
    return float(100 * np.interp(0, fp + tp - 1, fp))


def gaussian_scores(
    state: dict[str, torch.Tensor],
    raw: list[np.ndarray],
    groups: list[tuple[np.ndarray, np.ndarray]],
    trials: dict[str, np.ndarray],
    ref: dict[str, Any],
) -> dict[str, np.ndarray]:
    scores = {}
    index = [key for _, key in ref["feature_blocks"]["voice_train"]].index("eres")
    dims = ref["voice_block_dimensions"]
    sl = slice(sum(dims[:index]), sum(dims[: index + 1]))
    for protocol in ("standard", "same_gender"):
        qset = {k: q for k, q in trials.items() if k.endswith("_" + protocol)}
        adapted = dict(state)
        if protocol == "standard":
            w = state["voice.net.0.weight"].clone()
            m = np.stack([raw[1][v, sl].mean(axis=0) for _, v in groups]).mean(axis=0)
            adapted["voice.net.0.bias"] = state["voice.net.0.bias"] + w[
                :, sl
            ] @ torch.from_numpy(m)
            w[:, sl] = 0
            adapted["voice.net.0.weight"] = w
        em = [embed(x, adapted, s, ref) for x, s in zip(raw, SIDES, strict=True)]
        stats = fit(em[0], em[1], groups, ref)
        for k, q in qset.items():
            joint = np.column_stack((em[0][q[:, 0]], em[1][q[:, 1]])) - stats["mean"]
            scores[k] = stats["constant"] - 0.5 * np.sum(
                (joint @ stats["precision_delta"]) * joint, axis=1
            )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    mode = parser.parse_args().mode
    cfg = read(HERE / "config.json")
    train_cfg = read(ROOT / cfg["training_config"])
    ref = read(ROOT / train_cfg["reference_config"])
    spec = read(ROOT / train_cfg["protocol_config"])
    assert (
        Path(sys.executable).parent.parent
        == Path(sys.prefix)
        == Path(train_cfg["python_prefix"])
    )
    torch.set_num_threads(train_cfg["resource_cpu_threads"])
    torch.cuda.set_per_process_memory_fraction(train_cfg["resource_gpu_fraction"])
    start = time.monotonic()
    out = HERE / "output" / (datetime.now().strftime("%m-%d-%H%M%S") + "-" + mode)
    out.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "训练与复核.log")],
    )
    LOG.info("运行 %s，解释器 %s，环境 %s", out.name, sys.executable, sys.prefix)
    folds = ref["folds"] if mode == "full" else ref["folds"][:1]
    seeds = ref["seeds"] if mode == "full" else ref["seeds"][:1]
    phases = {k: dict(train_cfg[k]) for k in ("pretrain", "adapt")}
    if mode == "smoke":
        for phase in phases.values():
            phase["steps"] = cfg["smoke_steps"]
    save(
        out / "实际配置.json",
        {
            "study": cfg,
            "original": train_cfg,
            "matcher": ref,
            "folds": folds,
            "seeds": seeds,
            "phases": phases,
            "mode": mode,
            "executable": sys.executable,
            "prefix": sys.prefix,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "argv": sys.argv,
        },
    )
    raw, historical, extra = load_sources(ref, read(ROOT / cfg["screened_vox"]))
    meta = [
        mm + read(ROOT / ref["metadata"][side][1])
        for side, mm in zip(SIDES, historical, strict=True)
    ]
    raw_eval = [x[: len(m)].cpu().numpy() for x, m in zip(raw, historical, strict=True)]
    native = genders(cfg)
    trials, weights = locked_trials(historical, cfg, native, ref)
    official_ids = [
        np.array([r["id"] for r in mm[: ref["official_rows"]]]) for mm in historical
    ]
    alias = ref["canonical_alias"]
    held_rows = [
        r for mm in historical for r in mm if r["src"] in cfg["withheld_sources"]
    ]
    excluded = {alias.get(r["id"], r["id"]) for r in held_rows}
    held_video = {r["video"] for r in held_rows if r["video"]}
    collision = {
        alias.get(r["id"], r["id"])
        for mm in meta
        for r in mm
        if r["src"] not in cfg["withheld_sources"] and r["video"] in held_video
    }
    excluded |= collision
    extra = {n: rows for n, rows in extra.items() if n not in excluded}
    vocab = {
        v: i
        for i, v in enumerate(
            sorted({r["video"] for mm in meta for r in mm if r["video"]})
        )
    }
    video_codes = [np.array([vocab.get(r["video"], -1) for r in mm]) for mm in meta]
    save(
        out / "隔离审计.json",
        {
            "withheld_sources": cfg["withheld_sources"],
            "held_recordings": len(held_video),
            "excluded_ids": sorted(excluded),
            "other_source_ids_sharing_recordings": sorted(collision),
            "verified_extra_ids": sorted(extra),
            "aliases": alias,
            "v1_frozen_sha256": hashfile(ROOT / cfg["v1_trials"]),
            "source_overlap_limit": "按已知身份别名和原录制编号隔离，公开前端预训练身份未知",
        },
    )
    split = ROOT / ref["parent_run"]
    preds: dict[str, dict[str, list[np.ndarray]]] = {
        arm: defaultdict(list) for arm in cfg["arms"]
    }
    loss_history = {}
    members = {}
    with (
        np.load(ROOT / train_cfg["english_pools"]) as arch,
        np.load(ROOT / spec["flag_pairs"]) as flag,
    ):
        for fold in folds:
            protocol = read(split / f"fold_protocols/fold_{fold}.json")
            banned = excluded | set(protocol["outer_ids"])
            names = sorted(
                key.split("__")[1]
                for key in arch.files
                if key.startswith(f"{fold}_final__")
                and key.endswith("__face")
                and key.split("__")[1] not in banned
            )
            original: Pools = {
                n: (arch[f"{fold}_final__{n}__face"], arch[f"{fold}_final__{n}__voice"])
                for n in names
            }
            expanded = {**original, **extra}
            for n, rows in expanded.items():
                for j, indices in enumerate(rows):
                    assert (
                        np.all(np.diff(np.sort(indices)) > 0)
                        and len(indices) >= train_cfg["per_identity"]
                    )
                    assert all(
                        alias.get(meta[j][i]["id"], meta[j][i]["id"]) == n
                        and meta[j][i]["src"] not in cfg["withheld_sources"]
                        and meta[j][i]["video"] not in held_video
                        for i in indices
                    )
                    if n in original:
                        assert all(meta[j][i]["lang"] == "English" for i in indices)
            strata = {n: (n.split("_")[0], native.get(n, "")) for n in expanded}
            fit_ids = [n for n in protocol["official_fit_ids"] if n not in banned]
            official: Pools = {
                n: tuple(x[x < ref["official_rows"]] for x in original[n])
                for n in fit_ids
                if n in original
            }
            groups = [
                (
                    np.flatnonzero(official_ids[0] == n),
                    np.flatnonzero(official_ids[1] == n),
                )
                for n in fit_ids
            ]
            assert (
                set(expanded).isdisjoint(banned)
                and all(n.startswith("v4_") for n in fit_ids)
                and set(official) <= set(fit_ids)
            )
            members[str(fold)] = {
                "original": names,
                "expanded": sorted(expanded),
                "official_adapt": sorted(official),
                "official_calibration": fit_ids,
                "official_held": protocol["outer_ids"],
            }
            local = {k: q for k, q in trials.items() if not k.startswith("fold")}
            for p in ref["protocols"]:
                key = f"fold{fold}_" + p
                q = flag[key + "_pairs"].astype(int)
                assert all(
                    {historical[j][i]["id"] for i in q[:, j]}
                    <= set(protocol["outer_ids"])
                    for j in (0, 1)
                )
                local[key] = q
                weights[key] = np.ones(len(q))
            for seed in seeds:
                allowed = read(split / f"plans/s{seed}_f{fold}_final.json")
                for j, side in enumerate(SIDES):
                    assert all(
                        set(rows[j]) <= set(allowed[side + "_indices_by_class"][n])
                        for n, rows in expanded.items()
                    )
                for arm in cfg["arms"]:
                    pool = original if arm == "original" else expanded
                    cond = arm if arm in ("count_matched", "same_gender") else None
                    cross = arm not in ("original", "vox_all")
                    args = {
                        "condition": cond,
                        "strata": strata,
                        "condition_weight": cfg["auxiliary_weight"] if cond else 0.0,
                    }
                    st, loss, info = stage(
                        pool,
                        len(pool),
                        raw,
                        phases["pretrain"],
                        train_cfg,
                        ref,
                        fold,
                        seed,
                        videos=video_codes,
                        positive_mode="cross_video" if cross else "all",
                        **args,
                    )
                    final, adapt, ainfo = stage(
                        official,
                        len(pool),
                        raw,
                        phases["adapt"],
                        train_cfg,
                        ref,
                        fold,
                        seed,
                        initial=st,
                        **args,
                    )
                    tag = f"{arm}_f{fold}_s{seed}"
                    loss_history[tag] = {
                        "pretrain": loss,
                        "adapt": adapt,
                        "pretrain_diagnostics": info,
                        "adapt_diagnostics": ainfo,
                    }
                    torch.save(final, out / (tag + ".pt"))
                    for k, s in gaussian_scores(
                        final, raw_eval, groups, local, ref
                    ).items():
                        assert np.isfinite(s).all()
                        preds[arm][k].append(s)
                LOG.info("完成第 %d 折第 %d 种子的五个对照", fold, seed)
            for k, q in local.items():
                if k.startswith("fold"):
                    trials[k] = q
            save(out / "训练损失.json", loss_history)
    report = {}
    stored: dict[str, Any] = {}
    for k, q in trials.items():
        expected_members = (
            len(seeds) if k.startswith("fold") else len(folds) * len(seeds)
        )
        assert all(len(pr[k]) == expected_members for pr in preds.values())
        agg = {arm: np.mean(pr[k], axis=0) for arm, pr in preds.items()}
        for arm in cfg["arms"][1:]:
            agg["equal_" + arm] = (agg["original"] + agg[arm]) / 2
        stored["q_" + k] = q
        stored["w_" + k] = weights[k]
        for arm, s in agg.items():
            stored[arm + "__" + k] = s
            report.setdefault(arm, {})[k] = eer(q[:, 2], s, weights[k])
    for r in report.values():
        for p in ref["protocols"]:
            r["flag_" + p] = float(np.mean([r[f"fold{f}_{p}"] for f in folds]))
    save(
        out / "各协议成绩.json",
        {"eer": report, "elapsed": time.monotonic() - start, "mode": mode},
    )
    save(out / "训练与校准身份.json", members)
    np.savez_compressed(out / "冻结试验与逐对预测.npz", allow_pickle=False, **stored)
    for arm, values in report.items():
        LOG.info(
            "%s %s",
            arm,
            {k: round(v, 3) for k, v in values.items() if not k.startswith("fold")},
        )


if __name__ == "__main__":
    main()
