"""复用历史训练的抽样顺序与身份隔离规则。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from media_common import CFG, PC, REF, ROOT, N, read
from sampling import Pools, draw_rows, recording_weights
from torch import Tensor


@dataclass
class BatchPlan:
    indices: list[Tensor]
    weights: Tensor | None
    checks: dict


def fingerprint(x: np.ndarray) -> str:
    return hashlib.sha256(x.tobytes()).hexdigest()


def make(indices: list[np.ndarray], w: np.ndarray | None, desc: dict) -> BatchPlan:
    device = REF["device"]
    return BatchPlan(
        [torch.from_numpy(z).to(device) for z in indices],
        torch.from_numpy(w).to(device) if w is not None else None,
        desc,
    )


def ordinary(
    pool: Pools,
    last: str | None,
    phase: str,
    f: int,
    s: int,
    vid: list[np.ndarray],
    steps: int,
) -> tuple[BatchPlan, np.ndarray]:
    st = PC[phase]
    names = sorted(set(pool) - {last}) + [last] if last in pool else sorted(pool)
    offset = (s - 1) * PC["replicate_stride"]
    rng = np.random.default_rng(st["class_seed"] + f + offset)
    classes = np.stack(
        [rng.choice(len(names), st["identities"], replace=False) for _ in range(steps)]
    )
    z = [
        draw_rows(
            [pool[n][j] for n in names],
            classes,
            st["media_seed"] + 2 * f + j + offset,
            PC["per_identity"],
        )
        for j in range(2)
    ]
    w = None
    diag = {}
    if phase == "pretrain":
        w, diag = recording_weights(
            z,
            vid,
            st["identities"],
            PC["per_identity"],
            "cross_video",
            st["media_seed"] + 2 * f + offset + 2,
        )
    return make(
        z,
        w,
        {
            "class_sha256": fingerprint(classes),
            "rows_sha256": [fingerprint(x) for x in z],
            "recordings": diag,
        },
    ), classes


def pools(
    f: int, z: np.lib.npyio.NpzFile, meta: list, extra: Pools, transfer: bool
) -> tuple[Pools, Pools, dict]:
    split = read(ROOT / CFG["split_manifest"])[str(f)]
    original = read(ROOT / CFG["parent_run"] / "训练身份划分.json")[str(f)]
    deny = set(split["deny"])
    names = sorted(set(original["base"]) - deny)
    base = {n: (z[f"{f}_final__{n}__face"], z[f"{f}_final__{n}__voice"]) for n in names}
    pre = {**base, **extra}
    fit = list(split["fit"])
    off = {n: tuple(ix[ix < N] for ix in pre[n]) for n in split["adapt"]}
    blocked_video = set()
    if transfer:
        src = ROOT / CFG["transfer_run"]
        old = read(src / "隔离审计.json")
        assert old["withheld_sources"] == CFG["source_exclusions"]
        excluded = set(old["excluded_ids"]) | set(
            old["other_source_ids_sharing_recordings"]
        )
        blocked_video = {
            r["video"]
            for side in meta
            for r in side
            if r["src"] in CFG["source_exclusions"] and r["video"]
        }
        assert len(blocked_video) == old["held_recordings"]
        former = read(src / "训练与校准身份.json")[str(f)]
        pre = {
            n: rows
            for n, rows in pre.items()
            if n in former["expanded"] and n not in excluded
        }
        off = {n: rows for n, rows in off.items() if n in pre}
        fit = [
            n for n in fit if n in former["official_calibration"] and n not in excluded
        ]
        deny |= excluded
        assert set(pre) == set(former["expanded"]) - deny
    assert not deny.intersection(set(pre) | set(off) | set(fit)) and set(off) <= set(
        pre
    )
    for j in range(2):
        for n, rows in pre.items():
            items = [meta[j][i] for i in rows[j]]
            assert len(items) >= 4 and all(
                REF["canonical_alias"].get(m["id"], m["id"]) == n for m in items
            )
            assert n in extra or all(m["lang"] == "English" for m in items)
            assert all(m["video"] not in blocked_video for m in items)
            if transfer:
                assert all(m["src"] not in CFG["source_exclusions"] for m in items)
    return (
        pre,
        off,
        {
            "pre": sorted(pre),
            "adapt": sorted(off),
            "fit": fit,
            "outer": split["outer"],
            "denied": sorted(deny),
            "excluded_videos": len(blocked_video),
        },
    )


def gender_table() -> dict[str, str]:
    import csv

    c = read(ROOT / CFG["transfer_run"] / "实际配置.json")["study"]
    with (ROOT / c["v4_gender"]).open(encoding="utf-8-sig") as f:
        out = {"v4_" + r[0]: r[1].lower() for r in csv.reader(f)}
    with (ROOT / c["v1_gender"]).open(encoding="utf-8-sig") as f:
        out.update(
            {"v1_" + r["ids"]: r["gender"][0].lower() for r in csv.DictReader(f)}
        )
    with (ROOT / c["vox_gender"]).open(encoding="utf-8-sig") as f:
        out.update(
            {
                "vox2t_" + r[0].strip(): r[2].strip().lower()
                for r in list(csv.reader(f))[1:]
            }
        )
    assert set(out.values()) == {"m", "f"}
    return out
