"""在完整英语训练池上核对来源条件损失，并独立对照官方媒体去重。"""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "11-18-Vox英语身份扩充"))
from train import SIDES, Pools, load_sources, read, save, stage
from transfer import HERE, ROOT, Meta, genders, hashfile

LOG = logging.getLogger(__name__)


def official_media(
    out: Path, historical: Meta, cfg: dict[str, Any], ref: dict[str, Any]
) -> list[np.ndarray]:
    manifest = read(ROOT / cfg["official_hash_manifest"])
    assert len(manifest) == ref["official_rows"]
    masks = []
    representatives = []
    details = {}
    for j, side in enumerate(SIDES):
        seen: dict[str, str] = {}
        first: dict[str, int] = {}
        rep = np.zeros(len(manifest), dtype=np.int64)
        counts: Counter[str] = Counter()
        mask = np.zeros(len(manifest), dtype=bool)
        by_id: dict[str, list[str]] = defaultdict(list)
        for i, r in enumerate(manifest):
            old = historical[j][i]
            path = Path(old["path"])
            assert path.resolve() == (ROOT / r[side + "_path"]).resolve()
            n = old["id"]
            assert (
                n == "v4_" + r["identity"]
                and old["src"] == "v4"
                and old["lang"] == "English"
            )
            sha = hashfile(path)
            assert sha == r[side + "_sha256"]
            mask[i] = sha not in seen
            rep[i] = first.setdefault(sha, i)
            assert seen.setdefault(sha, n) == n
            counts[sha] += 1
            by_id[n].append(sha)
        masks.append(mask)
        representatives.append(rep)
        details[side] = {
            "rows": len(mask),
            "unique": int(mask.sum()),
            "max_multiplicity": max(counts.values()),
            "sha256_verified": len(mask),
            "counts_by_identity": {n: [len(v), len(set(v))] for n, v in by_id.items()},
        }
    np.savez_compressed(out / "官方唯一媒体.npz", face=masks[0], voice=masks[1])
    save(out / "官方媒体重复审计.json", details)
    return representatives


def sampled(
    original: Pools,
    representatives: list[np.ndarray],
    mode: str,
    fold: int,
    seed: int,
    base_seed: int,
) -> Pools:
    selected: Pools = {}
    for name, sides in original.items():
        records = []
        for j, ix in enumerate(sides):
            official = ix[ix < len(representatives[j])]
            if len(official) == 0:
                records.append(ix)
                continue
            _, first = np.unique(representatives[j][official], return_index=True)
            unique = official[np.sort(first)]
            keep = unique
            if mode == "size_matched":
                msg = f"{base_seed}/{fold}/{seed}/{name}/{j}"
                rng = np.random.default_rng(
                    int.from_bytes(hashlib.sha256(msg.encode()).digest()[:8], "big")
                )
                keep = rng.choice(official, size=len(unique), replace=False)
            records.append(ix[(ix >= len(representatives[j])) | np.isin(ix, keep)])
        selected[name] = (records[0], records[1])
    return selected


def main() -> None:
    cfg = read(HERE / "config.json")
    tc = read(ROOT / cfg["training_config"])
    ref = read(ROOT / tc["reference_config"])
    assert (
        Path(sys.executable).parent.parent
        == Path(sys.prefix)
        == Path(tc["python_prefix"])
    )
    torch.set_num_threads(tc["resource_cpu_threads"])
    torch.cuda.set_per_process_memory_fraction(tc["resource_gpu_fraction"])
    start = time.monotonic()
    out = HERE / "output" / (datetime.now().strftime("%m-%d-%H%M%S") + "-production")
    out.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "训练.log"), logging.StreamHandler()],
    )
    raw, hist, extra = load_sources(ref, read(ROOT / cfg["screened_vox"]))
    representatives = official_media(out, hist, cfg, ref)
    meta = [
        mm + read(ROOT / ref["metadata"][s][1])
        for s, mm in zip(SIDES, hist, strict=True)
    ]
    all_videos = {
        v: i
        for i, v in enumerate(
            sorted({r["video"] for mm in meta for r in mm if r["video"]})
        )
    }
    videos = [np.array([all_videos.get(r["video"], -1) for r in mm]) for mm in meta]
    native = genders(cfg)
    loss = {a: {} for a in cfg["production_arms"]}
    for arm in cfg["production_arms"]:
        dest = out / arm
        dest.mkdir()
        save(
            dest / "实际配置.json",
            {
                "experiment": tc,
                "matcher": ref,
                "study": cfg,
                "positive_mode": "cross_video",
                "recipe": arm,
                "python": sys.executable,
                "prefix": sys.prefix,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "argv": sys.argv,
            },
        )
    split = ROOT / ref["parent_run"]
    with np.load(ROOT / tc["english_pools"]) as arch:
        for fold in ref["folds"]:
            proto = read(split / f"fold_protocols/fold_{fold}.json")
            names = sorted(
                k.split("__")[1]
                for k in arch.files
                if k.startswith(f"{fold}_final__") and k.endswith("__face")
            )
            historical: Pools = {
                n: (arch[f"{fold}_final__{n}__face"], arch[f"{fold}_final__{n}__voice"])
                for n in names
            }
            assert historical.keys().isdisjoint(extra)
            base = {**historical, **extra}
            assert set(base).isdisjoint(proto["outer_ids"])
            strata = {n: (n.split("_")[0], native.get(n, "")) for n in base}
            for n, rows in historical.items():
                for j, ix in enumerate(rows):
                    assert all(
                        hist[j][i]["lang"] == "English"
                        and ref["canonical_alias"].get(
                            hist[j][i]["id"], hist[j][i]["id"]
                        )
                        == n
                        for i in ix
                    )
            for seed in ref["seeds"]:
                tag = f"fold{fold}_s{seed}"
                plan = read(split / f"plans/s{seed}_f{fold}_final.json")
                for j, side in enumerate(SIDES):
                    assert all(
                        set(rows[j]) <= set(plan[side + "_indices_by_class"][n])
                        for n, rows in base.items()
                    )
                lengths = None
                for arm in cfg["production_arms"]:
                    cond = arm if arm in ("count_matched", "same_gender") else None
                    pool = (
                        base
                        if cond
                        else sampled(
                            base, representatives, arm, fold, seed, cfg["trial_seed"]
                        )
                    )
                    if not cond:
                        sizes = {n: tuple(map(len, v)) for n, v in pool.items()}
                        if lengths is not None:
                            assert lengths == sizes
                        lengths = sizes
                    official: Pools = {
                        n: (
                            pool[n][0][pool[n][0] < ref["official_rows"]],
                            pool[n][1][pool[n][1] < ref["official_rows"]],
                        )
                        for n in sorted(set(proto["official_fit_ids"]) & set(pool))
                    }
                    assert all(
                        min(map(len, v)) >= tc["per_identity"]
                        for v in official.values()
                    )
                    kw = {
                        "condition": cond,
                        "strata": strata,
                        "condition_weight": cfg["auxiliary_weight"] if cond else 0.0,
                    }
                    state, l1, d1 = stage(
                        pool,
                        len(pool),
                        raw,
                        tc["pretrain"],
                        tc,
                        ref,
                        fold,
                        seed,
                        videos=videos,
                        positive_mode="cross_video",
                        **kw,
                    )
                    state, l2, d2 = stage(
                        official,
                        len(pool),
                        raw,
                        tc["adapt"],
                        tc,
                        ref,
                        fold,
                        seed,
                        initial=state,
                        **kw,
                    )
                    torch.save(state, out / arm / (tag + ".pt"))
                    loss[arm][tag] = {
                        "pretrain": l1,
                        "adapt": l2,
                        "pretrain_diagnostics": d1,
                        "adapt_diagnostics": d2,
                    }
                LOG.info("完成 %s 的四组同预算训练", tag)
            for arm, values in loss.items():
                save(out / arm / "训练损失.json", values)
    for arm in cfg["production_arms"]:
        save(
            out / arm / "训练完成.json",
            {
                "heads": len(ref["folds"]) * len(ref["seeds"]),
                "batch_elapsed": time.monotonic() - start,
            },
        )
    LOG.info("完成全部训练，耗时 %.2fs", time.monotonic() - start)


if __name__ == "__main__":
    main()
