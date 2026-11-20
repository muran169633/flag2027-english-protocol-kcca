"""从实际参数重算三个英语基线在相同配对上的分数。"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from anchor_model import SPEC
from data_eval import development, transfer
from external_eval import corpus, trials
from grouped_gaussian import KEYS, precise_project, scored, single_pairs
from media_common import CFG, REF, ROOT, SIDES, TRACKS, arrays, read, setup, sha, write
from media_eval_data import Cell, Media
from speech_eval import queries as speech_queries

LOG = logging.getLogger(__name__)


def sources(lock: dict, group: str) -> tuple:
    if group == "transfer":
        members = {
            str(f): lock["entries"][f"transfer/fold{f}"]["split"] for f in lock["folds"]
        }
        ds, cells, sx, _, _ = transfer(members)
        return ds, cells, sx
    ds, cells, sx, _, _ = development()
    ext, ext_sex, info, _, _ = corpus(ROOT / SPEC["actors"])
    sx.update(ext_sex)
    folds = tuple(REF["folds"])
    for dom, conditions in (
        ("actors", ["same_both"]),
        ("second", ["all", "independent_portrait"]),
    ):
        key = "extra_" + dom
        ds[key] = ext[dom]
        for tr in TRACKS:
            for cond in conditions:
                q, w = trials(dom, cond, ds[key], info, sx, tr)
                cells[f"{dom}_{cond}_{tr}"] = Cell(key, tr, q, w, None, folds)
    folder = ROOT / SPEC["lombard"]
    audit = read(folder / "双发声编码复核.json")
    file = folder / "两种发声特征.npz"
    assert sha(file) == audit["features_sha256"]
    rows = audit["rows"]
    a = arrays(file)
    people = np.array([r["person"] for r in rows])
    sx.update({r["person"]: r["gender"] for r in rows})
    ds["lombard"] = Media(a["face"], a["voice"], people, people)
    for tr in TRACKS:
        for style in ("p", "l"):
            q, w = speech_queries(rows, "known24", tr, "p", style)
            cells[f"lombard_p{style}_{tr}"] = Cell("lombard", tr, q, w, None, folds)
    for c in cells.values():
        q = c.q
        bank = ds[c.domain]
        np.testing.assert_array_equal(
            q[:, 2], bank.face_ids[q[:, 0]] == bank.voice_ids[q[:, 1]]
        )
    return ds, cells, sx


def file_map() -> dict:
    record = read(ROOT / CFG["parent_run"] / "候选包验收.json")
    assert sha(ROOT / record["package"]) == record["sha256"]
    result = {}
    for tr, code in TRACKS.items():
        parts = record["frozen_parameters"][code]
        current = ROOT / CFG["parent_run"] / (tr + "_parameters.npz")
        keys = [p for p in parts if ROOT / p != current]
        assert len(keys) == 1 and sha(current) == parts[str(current.relative_to(ROOT))]
        fixed = ROOT / keys[0]
        assert sha(fixed) == parts[keys[0]]
        result[tr] = {
            "fixed": fixed,
            "current": current,
            **{
                n: ROOT / SPEC["production"] / n / (tr + "_parameters.npz")
                for n in ("count_matched", "same_gender")
            },
        }
    return result


def extracted(p: dict, tag: str, tr: str) -> tuple:
    st = {
        f"{s}.net.{k}": torch.from_numpy(p[f"{tag}.{s}.{k}"])
        for s in SIDES
        for k in KEYS
    }
    co = [
        p[tag + ".mean"],
        p[f"{tag}.{TRACKS[tr]}.delta"],
        p[f"{tag}.{TRACKS[tr]}.bias"],
    ]
    return st, co


def run(frozen: Path, out: Path) -> None:
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    setup(out, "基线复核.log")
    lock = read(frozen / "冻结清单.json")
    assert lock["folds"] == REF["folds"] and lock["spec"] == SPEC
    bank, cells, _ = sources(lock, "development")
    files = file_map()
    write(
        out / "评分前输入.json",
        {
            "design": SPEC,
            "training_lock": sha(frozen / "冻结清单.json"),
            "parameters": {
                tr: {
                    n: {"path": str(p.relative_to(ROOT)), "sha256": sha(p)}
                    for n, p in v.items()
                }
                for tr, v in files.items()
            },
        },
    )
    raw = {}
    counts = {}
    errors = {}
    for tr in TRACKS:
        for name, p in files[tr].items():
            params = arrays(p)
            for f in REF["folds"]:
                active = {
                    k: c for k, c in cells.items() if c.track == tr and f in c.folds
                }
                for s in REF["seeds"]:
                    tag = f"s{s}_f{f}"
                    assert tag in params["tags"]
                    st, co = extracted(params, tag, tr)
                    for dom in sorted({c.domain for c in active.values()}):
                        ds = bank[dom]
                        em = precise_project(ds, st)
                        for k, c in active.items():
                            if c.domain != dom:
                                continue
                            key = name + "__" + k
                            v = scored(em, c.q, *co)
                            if key not in raw:
                                ix = np.linspace(0, len(v) - 1, 4, dtype=int)
                                errors[key] = float(
                                    np.max(
                                        abs(v[ix] - single_pairs(st, co, ds, c.q[ix]))
                                    )
                                )
                                assert errors[key] < 2e-9
                                raw[key] = v
                                counts[key] = 1
                            else:
                                raw[key] += v
                                counts[key] += 1
            LOG.info("已重算 %s %s", tr, name)
    pred = {}
    replay = {}
    for k, c in cells.items():
        for n in ("fixed", "current", "count_matched", "same_gender"):
            key = n + "__" + k
            assert counts[key] == len(c.folds) * len(REF["seeds"])
            raw[key] /= counts[key]
        for n in ("current", "count_matched", "same_gender"):
            key = n + "__" + k
            pred[key] = (raw["fixed__" + k] + raw[key]) / 2
        pred["q__" + k], pred["w__" + k] = c.q, c.weight
    for n in ("current", "count_matched", "same_gender"):
        path = (
            ROOT / CFG["parent_run"]
            if n == "current"
            else ROOT / SPEC["production"] / n
        )
        old = arrays(path / "逐对验证分数.npz")
        for k, c in cells.items():
            if c.domain not in {"flag", "crema", "bangla"}:
                continue
            legacy = k.replace("bangla_", "bn_")
            np.testing.assert_array_equal(old["labels_" + legacy], c.q[:, 2])
            diff = float(np.max(abs(pred[n + "__" + k] - old["equal_" + legacy])))
            assert diff < 1e-4, (n, k, diff)
            replay[n + "__" + k] = diff
    fp = out / "冻结评分.npz"
    np.savez_compressed(fp, allow_pickle=False, **pred)
    write(
        out / "完成.json",
        {
            "score_sha256": sha(fp),
            "source_sha256": sha(out / "评分前输入.json"),
            "single_pair_errors": errors,
            "replay_max_abs": replay,
            "seconds": time.monotonic() - start,
        },
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("frozen", "out"):
        cli.add_argument(name, type=Path)
    a = cli.parse_args()
    run(a.frozen, a.out)
