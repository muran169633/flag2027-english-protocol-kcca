"""检验新增训练身份对跨模态身份均值协方差的贡献。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import OAS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from exps.v480.run import embed


def symmetric_power(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    assert np.all(values > 0)
    return (vectors * values**power) @ vectors.T


def main() -> None:
    cfg = json.loads((ROOT / "exps/v480/config.json").read_text())
    assert sys.prefix == cfg["python_prefix"]
    spec = json.loads((ROOT / "exps/v493/config.json").read_text())
    loc = ROOT / spec["strong"]
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
        ext.append(np.array([meta[i]["id"] for i in ix]))
        clips.append(np.array([meta[i]["video"] for i in ix]))
        official.append(np.array([r["id"] for r in meta if r["src"] == "v4"]))
    bundle = {}
    tags = []
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
            assert len(donors) == 70 and len(fit) == 56
            key = f"B_s{seed}_f{fold}_hard_labels"
            state = torch.load(
                loc / (key + ".pt"), map_location="cpu", weights_only=True
            )["state"]
            em = dict(np.load(loc / (key + "_embeddings.npz")))
            st = dict(np.load(loc / (key + "_stats.npz")))
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
            weight = len(donor) / (len(donor) + len(original))
            variants = {"base": st["same"]}
            for arm, extra in (
                ("aligned", cross),
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
    np.savez_compressed(
        HERE / "output/逐对评分参数.npz", tags=tags, epsilon=cfg["bn_epsilon"], **bundle
    )


if __name__ == "__main__":
    main()
