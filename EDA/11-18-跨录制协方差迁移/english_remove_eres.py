"""限制外部典型相关强度至目标训练身份支持的幅度。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import OAS
from sklearn.metrics import roc_curve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from exps.v480.run import embed
from exps.v480.run import fit as fit_gaussian


def symmetric_power(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    assert np.all(values > 0)
    return (vectors * values**power) @ vectors.T


def main() -> None:
    cfg = json.loads((ROOT / "exps/v480/config.json").read_text())
    assert sys.prefix == cfg["python_prefix"]
    spec = json.loads((ROOT / "exps/v493/config.json").read_text())
    loc = ROOT / spec["strong"]
    bn = ROOT / "EDA/11-18-孟加拉语验证源/output"
    trials = json.loads((bn / "七人固定跨录制诊断.json").read_text())["trials"]
    bns = [dict(np.load(bn / f"七人原始特征{s}.npz")) for s in ("", "交换录制")]
    ids = bns[0]["identities"].tolist()
    assert ids == bns[1]["identities"].tolist()
    qbn = np.array(
        [
            (ids.index(r["face_id"]) + 2, ids.index(r["voice_id"]) + 2, r["label"])
            for r in trials
        ]
    )
    official_raw = []
    raw = []
    ext = []
    official = []
    clips = []
    blocks = cfg["feature_blocks"]
    for side in ("face", "voice"):
        meta = json.loads((ROOT / cfg["metadata"][side][0]).read_text())
        ix = np.array(
            [
                i
                for i, r in enumerate(meta)
                if r["src"] == "v1" and r["lang"] == "English" and r["video"]
            ]
        )
        specs = (
            [blocks["face"][0], blocks["face_additional"][0]]
            if side == "face"
            else blocks["voice_train"]
        )
        raw.append(
            np.column_stack(
                [np.load(ROOT / path)[k][ix].astype(np.float32) for path, k in specs]
            )
        )
        official_raw.append(
            np.column_stack(
                [np.load(ROOT / path)[k][: cfg["official_rows"]] for path, k in specs]
            ).astype(np.float32)
        )
        assert all(r["lang"] == "English" for r in meta[: cfg["official_rows"]])
        ext.append(np.array([meta[i]["id"] for i in ix]))
        clips.append(np.array([meta[i]["video"] for i in ix]))
        official.append(np.array([r["id"] for r in meta if r["src"] == "v4"]))
    pairs = np.load(loc / "diagnostic_predictions.npz")
    bundle = {}
    tags = []
    widths = [np.load(ROOT / path)[k].shape[1] for path, k in blocks["voice_train"]]
    index = [k for _, k in blocks["voice_train"]].index("eres")
    sl = slice(sum(widths[:index]), sum(widths[: index + 1]))
    scores = {}
    targets = {}
    counts = []
    for fold in cfg["folds"]:
        base = ROOT / spec["split"]
        fit = json.loads((base / f"fold_protocols/fold_{fold}.json").read_text())[
            "official_fit_ids"
        ]
        for seed in (1, 2, 3):
            allowed = set(
                json.loads((base / f"plans/s{seed}_f{fold}_final.json").read_text())[
                    "classes"
                ]
            )
            donors = sorted(
                n
                for n in set(ext[0]) & set(ext[1]) & allowed
                if all(
                    len(np.unique(v[names == n])) >= 2
                    for names, v in zip(ext, clips, strict=True)
                )
            )
            assert donors
            counts.append(len(donors))
            ck = (
                HERE.parent
                / "11-18-英语训练池从头对照/output/续训FLAG仅官方行"
                / f"fold{fold}_s{seed}.pt"
            )
            state = torch.load(ck, map_location="cpu", weights_only=True)
            groups = [
                (np.flatnonzero(official[0] == n), np.flatnonzero(official[1] == n))
                for n in fit
            ]
            means = np.stack(
                [official_raw[1][ix, sl].mean(0) for _, ix in groups]
            ).mean(0)
            w = state["voice.net.0.weight"]
            state["voice.net.0.bias"] += w[:, sl] @ torch.from_numpy(means)
            w[:, sl] = 0
            em = {
                side: embed(x, state, side, cfg)
                for side, x in zip(("face", "voice"), official_raw, strict=True)
            }
            st = fit_gaussian(em["face"], em["voice"], groups, cfg)
            original_means = []
            donor_means = []
            for j, side in enumerate(("face", "voice")):
                z = embed(raw[j], state, side, cfg)
                new = []
                for n in donors:
                    ix = np.flatnonzero(ext[j] == n)
                    unique, inv, count = np.unique(
                        clips[j][ix], return_inverse=True, return_counts=True
                    )
                    new.append((1 / (len(unique) * count[inv])) @ z[ix])
                donor_means.append(np.stack(new))
                original_means.append(
                    np.stack([em[side][official[j] == n].mean(0) for n in fit])
                )
            original = np.column_stack(original_means)
            donor = np.column_stack(donor_means)
            np.testing.assert_allclose(original.mean(0), st["mean"], atol=1e-10)
            prior = OAS(store_precision=False).fit(original).covariance_
            np.testing.assert_allclose(
                prior[:128, 128:], st["same"][:128, 128:], atol=1e-12
            )
            external = OAS(store_precision=False).fit(donor).covariance_
            d = original_means[0].shape[1]
            transforms = []
            for idx in (slice(0, d), slice(d, None)):
                a = prior[idx, idx]
                b = external[idx, idx]
                transform = symmetric_power(a, 0.5) @ symmetric_power(b, -0.5)
                np.testing.assert_allclose(transform @ b @ transform.T, a, atol=1e-10)
                transforms.append(transform)
            cross = transforms[0] @ external[:d, d:] @ transforms[1].T
            lf = np.linalg.cholesky(prior[:d, :d])
            lv = np.linalg.cholesky(prior[d:, d:])
            source_white = np.linalg.solve(lv, np.linalg.solve(lf, cross).T).T
            target_white = np.linalg.solve(lv, np.linalg.solve(lf, prior[:d, d:]).T).T
            u, rho, vt = np.linalg.svd(source_white, full_matrices=False)
            evidence = np.diag(u.T @ target_white @ vt.T)
            # 同一对外部典型方向，仅保留两组训练身份均支持的正向相关量。
            shared = lf @ (u * np.clip(evidence, 0, rho)) @ vt @ lv.T
            np.testing.assert_allclose(lf @ (u * rho) @ vt @ lv.T, cross, atol=1e-10)
            weight = len(donor) / (len(donor) + len(original))
            variants = {"base": st["same"]}
            for arm, extra in (
                ("aligned", cross),
                ("shared", shared),
                ("shrink_only", np.zeros_like(cross)),
            ):
                cov = st["same"].copy()
                cov[:d, d:] = (1 - weight) * prior[:d, d:] + weight * extra
                cov[d:, :d] = cov[:d, d:].T
                variants[arm] = cov
            tag = f"s{seed}_f{fold}"
            tags.append(tag)
            for side in ("face", "voice"):
                prefix = side + ".net."
                for k in (
                    "0.weight",
                    "0.bias",
                    "1.weight",
                    "1.bias",
                    "1.running_mean",
                    "1.running_var",
                ):
                    bundle[f"{tag}.{side}.{k}"] = state[prefix + k].cpu().numpy()
            bundle[tag + ".mean"] = st["mean"]
            for protocol, cov in (
                ("no_gender", variants["base"]),
                ("gender", variants["aligned"]),
            ):
                null = st["different"]
                bundle[tag + "." + protocol + ".delta"] = np.linalg.inv(
                    null
                ) - np.linalg.inv(cov)
                bundle[tag + "." + protocol + ".bias"] = np.array(
                    (np.linalg.slogdet(null)[1] - np.linalg.slogdet(cov)[1]) / 2
                )
            joint = {}
            for prot in cfg["protocols"]:
                q = pairs[f"fold{fold}_{prot}_pairs"].astype(int)
                tag = f"{fold}_{prot}"
                joint[tag] = np.column_stack(
                    (em["face"][q[:, 0]], em["voice"][q[:, 1]])
                )
                targets[tag] = q[:, 2]
            collected = []
            for b in bns:
                a, v = [embed(b[s], state, s, cfg) for s in ("face", "voice")]
                collected.append(np.column_stack((a[qbn[:, 0]], v[qbn[:, 1]])))
            full = np.concatenate(collected)
            for name, mask in (
                ("bn_standard", np.ones(len(trials), bool)),
                ("bn_same_gender", np.array([r["same_gender"] for r in trials])),
            ):
                joint[name] = full[np.tile(mask, 2)]
                targets[name] = np.tile(qbn[mask, 2], 2)
            for arm, sc in variants.items():
                dc = st["different"]
                np.linalg.cholesky(sc)
                np.linalg.cholesky(dc)
                delta = np.linalg.inv(dc) - np.linalg.inv(sc)
                const = (np.linalg.slogdet(dc)[1] - np.linalg.slogdet(sc)[1]) / 2
                for tag, value in joint.items():
                    x = value - st["mean"]
                    s = 0.5 * np.sum((x @ delta) * x, axis=1) + const
                    scores.setdefault((tag, arm), []).append(s)
    result = {}
    for (tag, arm), ls in scores.items():
        fp, tp, _ = roc_curve(
            targets[tag], np.mean(ls, axis=0), drop_intermediate=False
        )
        result.setdefault(tag, {})[arm] = float(100 * np.interp(0, fp + tp - 1, fp))
    for p in cfg["protocols"]:
        result[p] = {
            arm: float(np.mean([result[f"{f}_{p}"][arm] for f in cfg["folds"]]))
            for arm in ("base", "aligned", "shrink_only", "shared")
        }
    np.savez_compressed(
        HERE / "output/纯英语移除eres评分参数.npz",
        tags=tags,
        epsilon=cfg["bn_epsilon"],
        **bundle,
    )
    archived = json.loads(
        (
            HERE.parent / "11-18-英语训练池从头对照/output/语音逐路移除/eres/结果.json"
        ).read_text()
    )["ensemble"]
    for key, arms in result.items():
        archived_key = "fold" + key if key[0].isdigit() else key
        np.testing.assert_allclose(arms["base"], archived[archived_key], atol=1e-8)
    (HERE / "output/纯英语移除eres共同方向.json").write_text(
        json.dumps(result, indent=2)
    )
    print(
        counts,
        {p: result[p] for p in ["bn_standard", "bn_same_gender", *cfg["protocols"]]},
    )


if __name__ == "__main__":
    main()
