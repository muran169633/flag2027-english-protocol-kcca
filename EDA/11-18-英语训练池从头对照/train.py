"""同一身份与随机初态下比较英语和多语言匹配头。"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from exps.v480.run import Matcher, Objective, embed, fit


def draw(pools: list[np.ndarray], classes: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = np.empty((*classes.shape, 4), dtype=np.int64)
    for k, pool in enumerate(pools):
        use = classes == k
        # 每个身份的四个训练样本无放回抽取。
        rank = rng.random((int(use.sum()), len(pool)))
        rows[use] = pool[np.argpartition(rank, 3, axis=1)[:, :4]]
    return rows.reshape(len(classes), -1)


def main() -> None:
    cfg = json.loads((ROOT / "exps/v480/config.json").read_text())
    spec = json.loads((ROOT / "exps/v494/config.json").read_text())
    assert (
        Path(sys.executable).parent.parent
        == Path(sys.prefix)
        == Path(cfg["python_prefix"])
    )
    print(sys.executable, sys.prefix, flush=True)
    parent = ROOT / cfg["parent_run"]
    pools = np.load(HERE.parent / "11-18-跨录制协方差迁移/output/仅英语训练池.npz")
    trials = np.load(ROOT / spec["flag_pairs"])
    meta = [
        json.loads((ROOT / cfg["metadata"][s][0]).read_text())
        for s in ("face", "voice")
    ]
    raw = [
        np.column_stack(
            [np.load(ROOT / path)[k] for path, k in spec[s + "_features"]]
        ).astype(np.float32)
        for s in ("face", "voice")
    ]
    device = cfg["device"]
    tensors = [torch.from_numpy(x).to(device) for x in raw]
    labels = torch.arange(64, device=device).repeat_interleave(4)
    steps = 750
    result = {}
    loss_history = {}
    for fold in spec["folds"]:
        protocol = json.loads((parent / f"fold_protocols/fold_{fold}.json").read_text())
        plan = json.loads((parent / f"plans/s1_f{fold}_final.json").read_text())
        names = sorted(
            k.split("__")[1]
            for k in pools.files
            if k.startswith(f"{fold}_final__") and k.endswith("__face")
        )
        assert not set(names) & set(protocol["outer_ids"])
        rng = np.random.default_rng(8500 + fold)
        classes = np.stack(
            [rng.choice(len(names), 64, replace=False) for _ in range(steps)]
        )
        groups = [
            tuple(
                np.array(
                    [
                        i
                        for i, r in enumerate(mm[: cfg["official_rows"]])
                        if r["id"] == n
                    ]
                )
                for mm in meta
            )
            for n in protocol["official_fit_ids"]
        ]
        for arm in ("english", "multilingual"):
            indices = []
            for j, side in enumerate(("face", "voice")):
                source = [
                    pools[f"{fold}_final__{n}__{side}"]
                    if arm == "english"
                    else np.asarray(plan[side + "_indices_by_class"][n])
                    for n in names
                ]
                ix = draw(source, classes, 9600 + fold * 2 + j)
                if arm == "english":
                    assert all(meta[j][i]["lang"] == "English" for i in np.unique(ix))
                indices.append(torch.from_numpy(ix).to(device))
            torch.manual_seed(7500 + fold)
            model = Matcher(raw[0].shape[1], raw[1].shape[1], len(names), cfg).to(
                device
            )
            obj = Objective(cfg).to(device)
            with torch.no_grad():
                obj.log_scale.fill_(math.log(20))
                obj.center.fill_(0.3)
            opt = torch.optim.AdamW(
                [*model.parameters(), *obj.parameters()], lr=0.001, weight_decay=0.001
            )
            losses = []
            for step in range(steps):
                f, _ = model.face(tensors[0][indices[0][step]])
                v, _ = model.voice(tensors[1][indices[1][step]])
                loss = obj(f, v, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.detach())
            tag = f"fold{fold}_{arm}"
            loss_history[tag] = torch.stack(losses).cpu().tolist()
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(state, HERE / "output" / (tag + ".pt"))
            f, v = [
                embed(x[: cfg["official_rows"]], state, s, cfg)
                for s, x in zip(("face", "voice"), raw, strict=True)
            ]
            st = fit(f, v, [(g[0], g[1]) for g in groups], cfg)
            result[tag] = {}
            for p in spec["flag_protocols"]:
                a, b, y = trials[f"fold{fold}_{p}_pairs"].astype(int).T
                joint = np.column_stack((f[a], v[b])) - st["mean"]
                scores = st["constant"] - 0.5 * np.sum(
                    (joint @ st["precision_delta"]) * joint, axis=1
                )
                fp, tp, _ = roc_curve(y, scores, drop_intermediate=False)
                result[tag][p] = 100 * float(np.interp(0, fp + tp - 1, fp))
            print(tag, result[tag], flush=True)
    for arm in ("english", "multilingual"):
        result[arm] = {
            p: float(np.mean([result[f"fold{f}_{arm}"][p] for f in spec["folds"]]))
            for p in spec["flag_protocols"]
        }
    (HERE / "output/五折成绩.json").write_text(json.dumps(result, indent=2))
    (HERE / "output/训练损失.json").write_text(json.dumps(loss_history))
    print(result["english"], result["multilingual"])


if __name__ == "__main__":
    main()
