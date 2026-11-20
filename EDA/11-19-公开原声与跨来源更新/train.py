"""在英语训练来源之间执行一次可微更新，直接训练跨来源泛化能力。"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from identity_update import SPEC as FINISH
from identity_update import balanced, logits, trained_objective, views
from media_common import (
    CFG,
    HERE,
    PC,
    REF,
    ROOT,
    SIDES,
    arrays,
    read,
    setup,
    sha,
    write,
)
from media_network import Matcher, Objective
from plans import BatchPlan, fingerprint, gender_table, ordinary, pools
from sampling import Pools, load_sources, positive_loss
from torch import Tensor

LOG = logging.getLogger(__name__)
DESIGN = read(HERE / "训练与原声筛查.json")


def external(source: Path) -> tuple[list[Tensor], list[dict], str]:
    h = source / "训练前来源与角色.json"
    info = read(h)
    fp = source / "冻结图声特征.npz"
    assert sha(fp) == info["feature_sha256"]
    a = arrays(fp)
    assert list(a["qids"]) == [r["qid"] for r in info["rows"]]
    ix = [
        i for i, r in enumerate(info["rows"]) if r["accepted"] and r["role"] == "train"
    ]
    rows = [info["rows"][i] for i in ix]
    assert (
        len(rows) >= 4
        and {r["lang"] for r in rows} == {"en"}
        and len({r["qid"] for r in rows}) == len(rows)
    )
    return (
        [
            torch.tensor(a[k][ix], device=REF["device"])
            for k in ("face", DESIGN["auxiliary_training_audio"])
        ],
        rows,
        sha(h),
    )


def episode(
    wiki: list[dict],
    vox: Pools,
    sex: dict,
    video: list[np.ndarray],
    fold: int,
    seed: int,
    steps: int,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(
        CFG["seed"] + fold + (seed - 1) * PC["replicate_stride"]
    )
    wing = {
        g: np.array([i for i, r in enumerate(wiki) if r["gender"] == g])
        for g in ("m", "f")
    }
    people = {g: sorted(n for n in vox if sex[n] == g) for g in wing}
    total = min(CFG["outside_batch"], CFG["inside_batch"], len(wiki))
    nf = min(len(wing["f"]), len(people["f"]) // 2, total // 2)
    nm = min(len(wing["m"]), len(people["m"]) // 2, total - nf)
    assert min(nf, nm) >= 2
    candidates = {}
    for name, axes in vox.items():
        a, b = axes
        ok = (
            (video[0][a, None] != video[1][b][None, :])
            & (video[0][a, None] >= 0)
            & (video[1][b][None, :] >= 0)
        )
        i, j = np.where(ok)
        assert len(i)
        candidates[name] = np.column_stack((a[i], b[j]))
    chosen = {k: [] for k in ("wiki", "wrong", "voxs", "voxq")}
    supports = set()
    queries = set()
    for _ in range(steps):
        raw = []
        wrong = []
        one = []
        two = []
        for g, n in (("m", nm), ("f", nf)):
            a = rng.choice(wing[g], n, replace=False)
            raw.extend(a)
            wrong.extend(np.roll(a, 1))
            who = rng.choice(people[g], 2 * n, replace=False)
            supports.update(who[:n])
            queries.update(who[n:])
            pairs = [candidates[k][rng.integers(len(candidates[k]))] for k in who]
            one.extend(pairs[:n])
            two.extend(pairs[n:])
        assert set(raw) == set(wrong) and all(
            l != r and wiki[l]["gender"] == wiki[r]["gender"]
            for l, r in zip(raw, wrong, strict=True)
        )
        for k, val in zip(chosen, (raw, wrong, one, two), strict=True):
            chosen[k].append(val)
    npplans = {k: np.asarray(v, dtype=np.int64) for k, v in chosen.items()}
    return {k: torch.from_numpy(v).to(REF["device"]) for k, v in npplans.items()}, {
        "rows_sha256": {k: fingerprint(v) for k, v in npplans.items()},
        "people_per_gender": {"m": nm, "f": nf},
        "wiki_eligible": [r["qid"] for r in wiki],
        "wiki_drawn": sorted({wiki[i]["qid"] for i in np.unique(npplans["wiki"])}),
        "vox_queried": sorted(queries),
        "vox_control": sorted(supports),
        "steps": steps,
    }


def loss(
    params: dict, bn: dict, batch: list[Tensor], scale: Tensor, center: Tensor
) -> Tensor:
    return balanced(logits(params, bn, batch, scale, center))


def objective(
    params: dict,
    bn: dict,
    support: list[Tensor],
    query: list[Tensor],
    scale: Tensor,
    center: Tensor,
    method: str,
    lr: float,
    gender: Tensor | None = None,
) -> tuple[Tensor, list[Tensor]]:
    if method in FINISH["objectives"]:
        if gender is None:
            raise ValueError("身份与排序目标需要训练人物的性别分层")
        return trained_objective(
            params, bn, support, query, scale, center, method, lr, gender
        )
    inner = loss(params, bn, support, scale, center)
    before = loss(params, bn, query, scale, center)
    if method in ("joint", "vox_only"):
        after = before
    else:
        gradients = torch.autograd.grad(
            inner,
            tuple(params.values()),
            create_graph=method != "stop",
            retain_graph=True,
        )
        virtual = {
            k: p - lr * g for (k, p), g in zip(params.items(), gradients, strict=True)
        }
        after = loss(virtual, bn, query, scale, center)
    return 0.5 * (inner + after), [inner.detach(), before.detach(), after.detach()]


def fit(
    raw: list[Tensor],
    plan: BatchPlan,
    nclasses: int,
    initseed: int,
    initial: dict | None,
    mode: str,
    ext: list[Tensor],
    aux: dict,
    terminal: bool = False,
    group: Tensor | None = None,
) -> tuple[dict, dict]:
    pre = initial is None
    optcfg = PC["pretrain" if pre else "adapt"]
    device = REF["device"]
    torch.manual_seed(initseed)
    model = Matcher(raw[0].shape[1], raw[1].shape[1], nclasses, REF).to(device)
    if initial is not None:
        model.load_state_dict(initial)
    obj = Objective(REF).to(device)
    with torch.no_grad():
        obj.center.fill_(PC["initial_center"])
        obj.log_scale.fill_(math.log(PC["initial_scale"]))
    opt = torch.optim.AdamW(
        [*model.parameters(), *obj.parameters()],
        lr=optcfg["learning_rate"],
        weight_decay=PC["weight_decay"],
    )
    ci = torch.arange(optcfg["identities"], device=device)
    labels = ci.repeat_interleave(PC["per_identity"])
    negative = labels[:, None] != labels[None, :]
    trace = []
    for step, (fi, vi) in enumerate(zip(*plan.indices, strict=True)):
        f, _ = model.face(raw[0][fi])
        v, _ = model.voice(raw[1][vi])
        w = plan.weights[step] if plan.weights is not None else None
        main = (
            obj(f, v, labels)
            if w is None
            else positive_loss(obj, f, v, w, ci, negative)
        )
        total = main
        diag = []
        if mode != "baseline":
            supp = (
                [raw[j][aux["voxs"][step, :, j]] for j in range(2)]
                if mode == "vox_only"
                else [
                    ext[0][aux["wiki"][step]],
                    ext[1][aux["wrong" if mode == "permuted" else "wiki"][step]],
                ]
            )
            query = [raw[j][aux["voxq"][step, :, j]] for j in range(2)]
            p, b = views(model)
            extra, diag = objective(
                p,
                b,
                supp,
                query,
                obj.log_scale.detach().exp().clamp(*obj.bounds),
                obj.center.detach(),
                mode,
                CFG["lookahead_lr"],
                group,
            )
            total = main + CFG["meta_fraction"] * extra
            diag.append(CFG["meta_fraction"] * extra.detach())
        opt.zero_grad()
        total.backward()
        opt.step()
        trace.append(torch.stack([main.detach(), total.detach(), *diag]))
    values = torch.stack(trace).cpu().numpy()
    assert np.isfinite(values).all()
    names = ["main", "total"] + (
        ["support", "query_before", "query_after", "weighted_aux"]
        if mode != "baseline"
        else []
    )
    if mode != "baseline":
        np.testing.assert_allclose(
            values[:, 1], values[:, 0] + values[:, -1], atol=1e-7, rtol=0
        )
    record: dict = {
        **dict(zip(names, values.T.tolist(), strict=True)),
        "lr": optcfg["learning_rate"],
    }
    if terminal:
        record["temperature"] = {
            "scale": float(obj.log_scale.detach().exp().clamp(*obj.bounds)),
            "center": float(obj.center.detach()),
        }
    return {k: t.detach().cpu() for k, t in model.state_dict().items()}, record


def training_inputs() -> tuple[list[Tensor], list[list[dict]], Pools, list[np.ndarray]]:
    raw, old, extra = load_sources(REF, read(ROOT / CFG["screened_vox"]))
    meta = [
        axis + read(ROOT / REF["metadata"][side][1])
        for axis, side in zip(old, SIDES, strict=True)
    ]
    codes = {
        v: i
        for i, v in enumerate(
            sorted({r["video"] for axis in meta for r in axis if r["video"]})
        )
    }
    videos = [np.array([codes.get(r["video"], -1) for r in axis]) for axis in meta]
    return raw, meta, extra, videos


def run(source: Path, out: Path, mode: str, transfer: bool) -> None:
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    setup(out, "来源间元训练.log")
    ext, people, digest = external(source)
    raw, meta, extra, videos = training_inputs()
    sex = gender_table()
    groups = read(ROOT / CFG["group_run"] / "评分前分组.json")
    past = ROOT / CFG["pretrain_run"]
    foreign: dict = {}
    ref = read(past / "训练适配与复放.json")["records"]
    if transfer:
        other = ROOT / CFG["existing_paired"] / "11-19-整来源五折"
        h = other / "逐步记录.json"
        assert sha(h) == read(other / "完成.json")["history_sha256"]
        foreign = read(h)
    folds = REF["folds"] if mode == "full" else REF["folds"][:1]
    seeds = REF["seeds"] if mode == "full" else REF["seeds"][:1]
    arms = DESIGN["arms"]
    write(
        out / "训练前配置.json",
        {
            "study": CFG,
            "design": DESIGN,
            "mode": mode,
            "methods": arms,
            "primary": DESIGN["primary"],
            "transfer": transfer,
            "folds": folds,
            "seeds": seeds,
            "external_lock_sha256": digest,
            "argv": sys.argv,
            "interpreter": sys.executable,
            "prefix": sys.prefix,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
        },
    )
    steps = CFG["smoke_steps"] if mode == "smoke" else PC["adapt"]["steps"]
    members = {}
    records = {}
    with np.load(ROOT / PC["english_pools"]) as bank:
        for f in folds:
            pool, off, members[str(f)] = pools(f, bank, meta, extra, transfer)
            allowed = {k: v for k, v in extra.items() if k in pool}
            for s in seeds:
                tag = f"fold{f}_s{s}"
                init = PC["torch_seed"] + f + (s - 1) * PC["replicate_stride"]
                mainplan, _ = ordinary(
                    off, groups[str(f)]["matched"], "adapt", f, s, videos, steps
                )
                plans, checks = episode(people, allowed, sex, videos, f, s, steps)
                if transfer:
                    pre, _ = ordinary(
                        pool,
                        groups[str(f)]["matched"],
                        "pretrain",
                        f,
                        s,
                        videos,
                        PC["pretrain"]["steps"],
                    )
                    state, prelog = fit(
                        raw, pre, len(pool), init, None, "baseline", [], {}
                    )
                    assert prelog["main"] == foreign[tag]["pretrain"]["base"]["main"]
                else:
                    origin = past / "共同初始化" / (tag + ".pt")
                    assert sha(origin) == ref[tag]["pre_sha256"]
                    state = torch.load(origin, map_location="cpu", weights_only=True)
                    prelog = {
                        "source": str(origin.relative_to(ROOT)),
                        "sha256": sha(origin),
                    }
                history = records[tag] = {
                    "aux_sampling": checks,
                    "official": mainplan.checks,
                    "pre": prelog,
                    "arms": {},
                }
                for name in arms:
                    result, metrics = fit(
                        raw, mainplan, len(pool), init, state, name, ext, plans
                    )
                    if name == "baseline" and mode != "smoke":
                        path = (
                            ROOT
                            / CFG["existing_paired"]
                            / "11-19-整来源五折"
                            / "baseline"
                            if transfer
                            else past / "original"
                        ) / (tag + ".pt")
                        expected = (
                            foreign[tag]["arms"]["baseline"]["sha256"]
                            if transfer
                            else ref[tag]["arms"]["original"]["sha256"]
                        )
                        assert sha(path) == expected
                        gold = torch.load(path, weights_only=True, map_location="cpu")
                        assert all(
                            torch.equal(result[k], val) for k, val in gold.items()
                        )
                    dest = out / name / (tag + ".pt")
                    dest.parent.mkdir(exist_ok=True)
                    torch.save(result, dest)
                    history["arms"][name] = {**metrics, "sha256": sha(dest)}
                LOG.info("完成 %s 的 %d 个训练对照", tag, len(arms))
            write(out / "逐步记录.json", records)
            write(out / "人物清单.json", members)
    write(
        out / "完成.json",
        {
            "seconds": time.monotonic() - start,
            "heads": len(records) * len(arms),
            "history_sha256": sha(out / "逐步记录.json"),
            "members_sha256": sha(out / "人物清单.json"),
            "lock_sha256": sha(out / "训练前配置.json"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--mode", choices=["smoke", "first", "full"], default="full")
    parser.add_argument("--transfer", action="store_true")
    a = parser.parse_args()
    run(a.source, a.out, a.mode, a.transfer)
