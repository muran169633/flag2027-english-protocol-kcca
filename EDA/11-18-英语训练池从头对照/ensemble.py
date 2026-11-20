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
    out = HERE / "output/英语三种子"
    out.mkdir(exist_ok=True)
    bn = HERE.parent / "11-18-孟加拉语验证源/output"
    br = json.loads((bn / "七人固定跨录制诊断.json").read_text())["trials"]
    bz = [
        dict(np.load(bn / f"七人原始特征{suffix}.npz")) for suffix in ("", "交换录制")
    ]
    ids = bz[0]["identities"].tolist()
    qbn = np.array(
        [
            (ids.index(r["face_id"]) + 2, ids.index(r["voice_id"]) + 2, r["label"])
            for r in br
        ]
    )
    mask = np.tile([r["same_gender"] for r in br], 2)
    pred = {}
    targets = {"bn": np.tile(qbn[:, 2], 2)}

    for fold in spec["folds"]:
        protocol = json.loads((parent / f"fold_protocols/fold_{fold}.json").read_text())

        names = sorted(
            k.split("__")[1]
            for k in pools.files
            if k.startswith(f"{fold}_final__") and k.endswith("__face")
        )
        assert not set(names) & set(protocol["outer_ids"])
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
        for seed in (1, 2, 3):
            tag = f"fold{fold}_s{seed}"
            offset = 10000 * (seed - 1)
            if seed == 1:
                state = torch.load(
                    HERE / "output" / f"fold{fold}_english.pt",
                    weights_only=True,
                    map_location="cpu",
                )
            else:
                rng = np.random.default_rng(8500 + fold + offset)
                classes = np.stack(
                    [rng.choice(len(names), 64, replace=False) for _ in range(steps)]
                )
                indices = []
                for j, side in enumerate(("face", "voice")):
                    source = [pools[f"{fold}_final__{n}__{side}"] for n in names]
                    ix = draw(source, classes, 9600 + fold * 2 + j + offset)
                    assert all(meta[j][i]["lang"] == "English" for i in np.unique(ix))
                    indices.append(torch.from_numpy(ix).to(device))
                torch.manual_seed(7500 + fold + offset)
                model = Matcher(raw[0].shape[1], raw[1].shape[1], len(names), cfg).to(
                    device
                )
                obj = Objective(cfg).to(device)
                with torch.no_grad():
                    obj.log_scale.fill_(math.log(20))
                    obj.center.fill_(0.3)
                opt = torch.optim.AdamW(
                    [*model.parameters(), *obj.parameters()],
                    lr=0.001,
                    weight_decay=0.001,
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

                loss_history[tag] = torch.stack(losses).cpu().tolist()
                state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(state, out / (tag + ".pt"))
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
                key = f"fold{fold}_{p}"
                targets[key] = y
                pred.setdefault(key, []).append(scores)
            bn_scores = []
            for z in bz:
                a, b = [embed(z[side], state, side, cfg) for side in ("face", "voice")]
                x = np.column_stack((a[qbn[:, 0]], b[qbn[:, 1]])) - st["mean"]
                bn_scores.append(
                    st["constant"]
                    - 0.5 * np.sum((x @ st["precision_delta"]) * x, axis=1)
                )
            pred.setdefault("bn", []).append(np.concatenate(bn_scores))
    previous = json.loads((HERE / "output/五折成绩.json").read_text())
    for fold in spec["folds"]:
        for p in spec["flag_protocols"]:
            np.testing.assert_allclose(
                result[f"fold{fold}_s1"][p],
                previous[f"fold{fold}_english"][p],
                atol=1e-8,
            )
    ensemble = {}
    for key, rows in pred.items():
        s = np.mean(rows, axis=0)
        masks = (
            {"bn_standard": np.ones(len(s), bool), "bn_same_gender": mask}
            if key == "bn"
            else {key: np.ones(len(s), bool)}
        )
        for name, m in masks.items():
            fp, tp, _ = roc_curve(targets[key][m], s[m], drop_intermediate=False)
            ensemble[name] = 100 * float(np.interp(0, fp + tp - 1, fp))
    for p in spec["flag_protocols"]:
        ensemble[p] = float(np.mean([ensemble[f"fold{f}_{p}"] for f in spec["folds"]]))
    (out / "结果.json").write_text(
        json.dumps({"ensemble": ensemble, "members": result}, indent=2)
    )
    (out / "训练损失.json").write_text(json.dumps(loss_history))
    for key, rows in pred.items():
        np.save(out / f"{key}_scores.npy", np.stack(rows))
    print(ensemble)


if __name__ == "__main__":
    main()
