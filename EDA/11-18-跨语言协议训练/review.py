"""复核辅助损失、固定换语试验，以及训练集重复媒体对英语留出成绩的影响。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "11-18-Vox英语身份扩充"))
from diagnose import leave_one
from transfer import HERE, ROOT, SIDES, eer, read, save

LOG = logging.getLogger(__name__)


def loss_check(
    random: dict[str, Any], native: dict[str, Any], weight: float
) -> dict[str, Any]:
    assert random.keys() == native.keys()
    totals = {"anchor_groups": 0, "negative_identity_pairs": 0}
    steps = 0
    native_negatives = [0, 0]
    for tag in random:
        for stage in ("pretrain", "adapt"):
            both = [d[tag][stage + "_diagnostics"] for d in (random, native)]
            for item, losses in zip(both, (random, native), strict=True):
                assert item["conditional_weight"] == weight
                parts = np.asarray(item["standard_conditional_loss"])
                np.testing.assert_allclose(
                    parts @ np.array([1 - weight, weight]),
                    losses[tag][stage],
                    rtol=1e-6,
                    atol=1e-7,
                )
            for key in totals:
                a, b = [d["condition_counts"][key] for d in both]
                assert a == b
                totals[key] += a
            for j, item in enumerate(both):
                native_negatives[j] += item["condition_counts"][
                    "true_same_gender_negative_pairs"
                ]
            steps += len(random[tag][stage])
    assert native_negatives[1] == totals["negative_identity_pairs"]
    return {
        "steps_per_arm": steps,
        "verified_model_pairs": len(random),
        **totals,
        "same_gender_fraction_count_matched": native_negatives[0] / native_negatives[1],
    }


def bilingual(run: Path, historical: list[list[dict[str, str]]]) -> dict[str, Any]:
    output = {}
    with np.load(run / "冻结试验与逐对预测.npz") as z:
        for lang in ("English", "Urdu"):
            key = "v1_" + lang + "_same_gender"
            q, w = z["q_" + key], z["w_" + key]
            y = q[:, 2]
            face, voice = [
                np.array([historical[j][i]["id"] for i in q[:, j]]) for j in (0, 1)
            ]
            np.testing.assert_array_equal(face == voice, y)
            np.testing.assert_array_equal(q[::2, 0], q[1::2, 0])
            assert np.all(y[::2] == 1) and np.all(y[1::2] == 0)
            names, fi = np.unique(face, return_inverse=True)
            strata = 2 * fi + y
            by_arm: dict[str, dict[str, Any]] = {}
            erasures = {}
            for arm in ("vox_cross", "count_matched", "same_gender"):
                s = z[arm + "__" + key]
                by_arm[arm] = {
                    "eer": eer(y, s, w),
                    "paired_choice_percent": float(100 * np.mean(s[::2] > s[1::2])),
                    "ties": int((s[::2] == s[1::2]).sum()),
                }
                deleted = []
                for n in names:
                    take = (face != n) & (voice != n)
                    counts = np.bincount(
                        strata[take], weights=w[take], minlength=2 * len(names)
                    )
                    assert np.count_nonzero(counts) == 2 * (len(names) - 1)
                    # 其余各脸端身份的正、负两类仍分别保留相同的总权重。
                    deleted.append(
                        eer(y[take], s[take], w[take] / counts[strata[take]])
                    )
                erasures[arm] = np.array(deleted)
            for first, second in (
                ("vox_cross", "count_matched"),
                ("count_matched", "same_gender"),
            ):
                diff = erasures[second] - erasures[first]
                by_arm[second]["paired_deletion_vs_" + first] = {
                    "better_tied_worse": [
                        int((diff < -1e-8).sum()),
                        int((abs(diff) <= 1e-8).sum()),
                        int((diff > 1e-8).sum()),
                    ],
                    "min_max_pp": [float(diff.min()), float(diff.max())],
                }
            output[key] = by_arm
    return output


def official_repeats(
    runs: dict[str, Path],
    ref: dict[str, Any],
    source: Path,
    manifest: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ids = [
        np.unique([r[side + "_sha256"] for r in manifest], return_inverse=True)[1]
        for side in SIDES
    ]
    inventory = {}
    weights = {}
    with np.load(source) as pairs:
        for fold in ref["folds"]:
            for protocol in ref["protocols"]:
                key = f"fold{fold}_{protocol}"
                q = pairs[key + "_pairs"].astype(int)
                y = q[:, 2]
                f, v = [codes[q[:, j]] for j, codes in enumerate(ids)]
                per_hash = np.column_stack([y, f, v])
                codes = [per_hash, per_hash[:, [0, 2]]]
                ws = []
                for group in codes:
                    _, inv, counts = np.unique(
                        group, axis=0, return_inverse=True, return_counts=True
                    )
                    ws.append(1.0 / counts[inv])
                weights[key] = (y, *ws)
                inventory[key] = {
                    "trials": len(q),
                    "distinct_byte_pairs": len(np.unique(per_hash, axis=0)),
                    "distinct_faces": len(np.unique(f)),
                    "distinct_voices": len(np.unique(v)),
                }
    variants = {}
    for name, run in runs.items():
        r = read(run / "同协议复核.json")["metrics"]["equal"]
        bn = read(run / "身份与独立孟语来源复核.json")["benav"]["ranking"]["equal"]
        summary = {k: v for k, v in r.items() if not k.startswith("fold")}
        summary["benav_rank1"] = [bn[side]["rank1"] for side in SIDES]
        with np.load(run / "逐对验证分数.npz") as z:
            for protocol in ref["protocols"]:
                by_fold = []
                for fold in ref["folds"]:
                    key = f"fold{fold}_{protocol}"
                    y, by_pair, by_voice = weights[key]
                    np.testing.assert_array_equal(z["labels_" + key], y)
                    s = z["equal_" + key]
                    row = [eer(y, s, w) for w in (np.ones(len(y)), by_pair, by_voice)]
                    np.testing.assert_allclose(row[0], r[key], atol=1e-8)
                    by_fold.append(row)
                summary[protocol + "_sensitivity"] = {
                    "columns": [
                        "original",
                        "equal_byte_pair_within_label",
                        "equal_voice_within_label",
                    ],
                    "per_fold": by_fold,
                    "mean": np.mean(by_fold, axis=0).tolist(),
                }
        variants[name] = summary
    return inventory, variants


def external_deletions(runs: dict[str, Path]) -> dict[str, Any]:
    corpus = ROOT / "outputs/research/11-13-msm-native-transfer/crema_stress"
    crema = [np.array([r["id"] for r in read(corpus / (s + "s.json"))]) for s in SIDES]
    seven = read(HERE.parent / "11-18-孟加拉语验证源/output/七人固定跨录制诊断.json")[
        "trials"
    ]
    domains = {}
    for protocol in ("standard", "same_gender"):
        with np.load(corpus / f"cross_video_trials_{protocol}.npz") as z:
            q = z["triples"].astype(int)
        domains["crema_" + protocol] = [v[q[:, j]] for j, v in enumerate(crema)]
        rows = [r for r in seven if protocol == "standard" or r["same_gender"]] * 2
        domains["bn_" + protocol] = [
            np.array([r[s + "_id"] for r in rows]) for s in SIDES
        ]
    output = {}
    with np.load(runs["previous"] / "逐对验证分数.npz") as baseline:
        for name, run in runs.items():
            if name == "previous":
                continue
            by_domain = {}
            with np.load(run / "逐对验证分数.npz") as current:
                for key, (face, voice) in domains.items():
                    y = baseline["labels_" + key]
                    np.testing.assert_array_equal(y, current["labels_" + key])
                    np.testing.assert_array_equal(face == voice, y)
                    sk = "equal_" + key
                    by_domain[key] = leave_one(
                        face, voice, y, baseline[sk], current[sk]
                    )
            output[name] = by_domain
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("transfer", "production", "baseline"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    target = args.production.resolve()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(target / "成对复核.log"),
        ],
    )
    cfg = read(HERE / "config.json")
    tc = read(ROOT / cfg["training_config"])
    ref = read(ROOT / tc["reference_config"])
    spec = read(ROOT / tc["protocol_config"])
    tr = read(args.transfer / "训练损失.json")
    loss_arms = ("count_matched", "same_gender")
    transfer_loss = [
        {k.removeprefix(a + "_"): v for k, v in tr.items() if k.startswith(a + "_")}
        for a in loss_arms
    ]
    production_loss = [read(target / arm / "训练损失.json") for arm in loss_arms]
    history = [read(ROOT / ref["metadata"][s][0]) for s in SIDES]
    runs = {
        "previous": args.baseline,
        **{
            a: target / a
            for a in [*cfg["production_arms"], "calibration_only", "unique_both"]
        },
    }
    inventory, variants = official_repeats(
        runs, ref, ROOT / spec["flag_pairs"], read(ROOT / cfg["official_hash_manifest"])
    )
    answer = {
        "loss_replay": {
            k: loss_check(rows[0], rows[1], cfg["auxiliary_weight"])
            for k, rows in (
                ("transfer", transfer_loss),
                ("production", production_loss),
            )
        },
        "bilingual": bilingual(args.transfer, history),
        "official_folds": inventory,
        "full_training_equal": variants,
        "external_equal_vs_previous": external_deletions(runs),
        "sources": {k: str(p.resolve().relative_to(ROOT)) for k, p in runs.items()},
    }
    save(target / "成对复核.json", answer)
    LOG.info(
        "已复核 %d 组冻结预测、%d 个留出协议及两批训练损失", len(runs), len(inventory)
    )


if __name__ == "__main__":
    main()
