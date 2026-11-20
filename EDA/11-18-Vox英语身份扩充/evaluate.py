"""在冻结的匹配协议上复核原训练池、新增身份及二者等权平均。"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import OAS
from sklearn.metrics import roc_curve
from train import HERE, ROOT, SIDES, feature_matrix, read, save

from exps.v480.run import embed, fit

ArrayPair = tuple[np.ndarray, np.ndarray]
LOG = logging.getLogger(__name__)


def eer(labels: np.ndarray, scores: np.ndarray) -> float:
    assert labels.shape == scores.shape and np.isfinite(scores).all()
    fp, tp, _ = roc_curve(labels, scores, drop_intermediate=False)
    return float(100 * np.interp(0, fp + tp - 1, fp))


def symmetric_power(matrix: np.ndarray, power: float) -> np.ndarray:
    val, vec = np.linalg.eigh(matrix)
    assert np.all(val > 0)
    return (vec * val**power) @ vec.T


def align(
    stats: dict[str, np.ndarray],
    em: list[np.ndarray],
    donors: list[np.ndarray],
    groups: list[ArrayPair],
    donor_groups: list[list[ArrayPair]],
) -> tuple[np.ndarray, np.ndarray]:
    original = np.column_stack(
        [np.stack([em[j][g[j]].mean(0) for g in groups]) for j in (0, 1)]
    )
    external = np.column_stack(
        [
            np.stack([w @ donors[j][ix] for ix, w in side_groups])
            for j, side_groups in enumerate(donor_groups)
        ]
    )
    prior = OAS(store_precision=False).fit(original).covariance_
    ext = OAS(store_precision=False).fit(external).covariance_
    dim = em[0].shape[1]
    std = []
    for sl in (slice(0, dim), slice(dim, None)):
        transform = symmetric_power(prior[sl, sl], 0.5) @ symmetric_power(
            ext[sl, sl], -0.5
        )
        np.testing.assert_allclose(
            transform @ ext[sl, sl] @ transform.T, prior[sl, sl], atol=1e-10
        )
        std.append(transform)
    cross = std[0] @ ext[:dim, dim:] @ std[1].T
    weight = len(external) / (len(original) + len(external))
    same = stats["same"].copy()
    np.testing.assert_allclose(same[:dim, dim:], prior[:dim, dim:], atol=1e-12)
    same[:dim, dim:] = (1 - weight) * prior[:dim, dim:] + weight * cross
    same[dim:, :dim] = same[:dim, dim:].T
    np.linalg.cholesky(same)
    null = stats["different"]
    return np.linalg.inv(null) - np.linalg.inv(same), np.asarray(
        (np.linalg.slogdet(null)[1] - np.linalg.slogdet(same)[1]) / 2
    )


def external_trials() -> tuple[dict[str, list[ArrayPair]], dict[str, np.ndarray]]:
    bn = HERE.parent / "11-18-孟加拉语验证源/output"
    rows = read(bn / "七人固定跨录制诊断.json")["trials"]
    matrices = []
    ids = []
    for suffix in ("", "交换录制"):
        with np.load(bn / f"七人原始特征{suffix}.npz") as z:
            if ids:
                assert z["identities"].tolist() == ids
            ids = z["identities"].tolist()
            # 两个开头的特征行来自提取器质检录音，以后才是七位验证身份。
            assert len(z["face"]) == len(z["voice"]) == len(ids) + 2
            matrices.append((z["face"][2:], z["voice"][2:]))
    data = {"bn": matrices}
    by_protocol = {}
    for protocol in ("standard", "same_gender"):
        selected = [r for r in rows if protocol == "standard" or r["same_gender"]]
        by_protocol["bn_" + protocol] = np.array(
            [
                (ids.index(r["face_id"]), ids.index(r["voice_id"]), r["label"])
                for r in selected
            ]
        )
    crema = ROOT / "outputs/research/11-13-msm-native-transfer/crema_stress"
    arr = feature_matrix([[str(crema / "face_v399.npz"), k] for k in ("clip", "clipL")])
    voice = feature_matrix(
        [
            [str(crema / "voice_core_v399.npz"), k]
            for k in ("wavlm", "ecapa", "eres", "resnet")
        ]
        + [[str(crema / "voice_attrs_v399.npz"), k] for k in ("whisper", "age6")]
    )
    data["crema"] = [(arr, voice)]
    m = [read(crema / (s + "s.json")) for s in SIDES]
    for protocol in ("standard", "same_gender"):
        with np.load(crema / f"cross_video_trials_{protocol}.npz") as z:
            q = z["triples"].astype(int)
        assert all(
            (m[0][a]["id"] == m[1][b]["id"]) == y
            and m[0][a]["video"] != m[1][b]["video"]
            for a, b, y in q
        )
        by_protocol["crema_" + protocol] = q
    return data, by_protocol


def score(
    em: list[np.ndarray],
    q: np.ndarray,
    mu: np.ndarray,
    delta: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    x = np.column_stack((em[0][q[:, 0]], em[1][q[:, 1]])) - mu
    return 0.5 * np.sum((x @ delta) * x, axis=1) + bias


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="本次训练的输出目录")
    parser.add_argument(
        "--unique-calibration",
        type=Path,
        help="已核实官方训练媒体字节指纹的去重掩码，仅用于候选分支",
    )
    parser.add_argument("--result-dir", type=Path, help="另行归档候选校准对照")
    args = parser.parse_args()
    run = args.run.resolve()
    dest = args.result_dir.resolve() if args.result_dir is not None else run
    if args.unique_calibration is not None and dest == run:
        parser.error("去重校准需要与原评分结果不同的 --result-dir")
    dest.mkdir(parents=True, exist_ok=True)
    cfg = read(HERE / "config.json")
    assert (
        Path(sys.executable).parent.parent
        == Path(sys.prefix)
        == Path(cfg["python_prefix"])
    )
    ref = read(ROOT / cfg["reference_config"])
    spec = read(ROOT / cfg["protocol_config"])
    assert read(run / "训练完成.json")["heads"] == len(ref["seeds"]) * len(ref["folds"])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(dest / "验证.log"), logging.StreamHandler()],
    )
    start = time.monotonic()
    torch.set_num_threads(cfg["resource_cpu_threads"])
    torch.cuda.set_per_process_memory_fraction(cfg["resource_gpu_fraction"])
    ext_raw, qext = external_trials()
    official = []
    donors = []
    official_ids = []
    donor_ids = []
    donor_videos = []
    for side in SIDES:
        raw = feature_matrix(spec[side + "_features"])
        meta = read(ROOT / ref["metadata"][side][0])
        n = ref["official_rows"]
        assert all(r["src"] == "v4" and r["lang"] == "English" for r in meta[:n])
        ix = np.array(
            [
                i
                for i, r in enumerate(meta)
                if r["src"] == "v1" and r["lang"] == "English" and r["video"]
            ]
        )
        official.append(raw[:n].copy())
        donors.append(raw[ix])
        official_ids.append(np.array([r["id"] for r in meta[:n]]))
        donor_ids.append(np.array([meta[i]["id"] for i in ix]))
        donor_videos.append(np.array([meta[i]["video"] for i in ix]))
    index = [key for _, key in spec["voice_features"]].index("eres")
    dims = ref["voice_block_dimensions"]
    sl = slice(sum(dims[:index]), sum(dims[: index + 1]))
    parent = ROOT / ref["parent_run"]
    unique = None
    if args.unique_calibration is not None:
        with np.load(args.unique_calibration, allow_pickle=False) as masks:
            unique = [masks[k] for k in SIDES]
        assert all(
            m.dtype == bool and m.shape == (ref["official_rows"],) for m in unique
        )
    if dest != run:
        effective = read(run / "实际配置.json")
        effective["heads_source"] = str(run.relative_to(ROOT))
        effective["calibration_mask"] = (
            str(args.unique_calibration.resolve().relative_to(ROOT))
            if args.unique_calibration is not None
            else None
        )
        save(dest / "实际配置.json", effective)
    arms = {"old": ROOT / cfg["baseline_heads"], "expanded": run}
    results: dict[str, dict[str, list[np.ndarray]]] = {
        a: defaultdict(list) for a in arms
    }
    bundles: dict[str, dict[str, np.ndarray]] = {p: {} for p in ref["protocols"]}
    labels: dict[str, np.ndarray] = {}
    tags = []
    with np.load(ROOT / spec["flag_pairs"]) as pairs:
        for fold in ref["folds"]:
            split = read(parent / f"fold_protocols/fold_{fold}.json")
            all_groups = [
                (
                    np.flatnonzero(official_ids[0] == n),
                    np.flatnonzero(official_ids[1] == n),
                )
                for n in split["official_fit_ids"]
            ]
            candidate_groups = (
                all_groups
                if unique is None
                else [(f[unique[0][f]], v[unique[1][v]]) for f, v in all_groups]
            )
            assert all(len(f) and len(v) for f, v in candidate_groups)
            for seed in ref["seeds"]:
                tag = f"s{seed}_f{fold}"
                tags.append(tag)
                allow = set(
                    read(parent / f"plans/s{seed}_f{fold}_final.json")["classes"]
                )
                names = sorted(
                    n
                    for n in set(donor_ids[0]) & set(donor_ids[1]) & allow
                    if all(
                        len(np.unique(v[ids == n])) >= 2
                        for ids, v in zip(donor_ids, donor_videos, strict=True)
                    )
                )
                donor_groups: list[list[ArrayPair]] = []
                for j in (0, 1):
                    per_id = []
                    for name in names:
                        ix = np.flatnonzero(donor_ids[j] == name)
                        video, inv, counts = np.unique(
                            donor_videos[j][ix], return_inverse=True, return_counts=True
                        )
                        per_id.append((ix, 1 / (len(video) * counts[inv])))
                    donor_groups.append(per_id)
                for arm, directory in arms.items():
                    groups = candidate_groups if arm == "expanded" else all_groups
                    # 一个唯一观测具有零类内方差。以两个完全相同的观测表示它，保留原拟合器接口及单位身份权重。
                    fit_groups = [
                        (
                            np.repeat(f, 2) if len(f) == 1 else f,
                            np.repeat(v, 2) if len(v) == 1 else v,
                        )
                        for f, v in groups
                    ]
                    full = torch.load(
                        directory / f"fold{fold}_s{seed}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    for protocol in ref["protocols"]:
                        track = "no_gender" if protocol == "standard" else "gender"
                        state = dict(full)
                        if protocol == "standard":
                            w = state["voice.net.0.weight"].clone()
                            m = np.stack(
                                [official[1][ix, sl].mean(0) for _, ix in groups]
                            ).mean(0)
                            state["voice.net.0.bias"] = state["voice.net.0.bias"] + w[
                                :, sl
                            ] @ torch.from_numpy(m)
                            w[:, sl] = 0
                            state["voice.net.0.weight"] = w
                        em = [
                            embed(x, state, s, ref)
                            for s, x in zip(SIDES, official, strict=True)
                        ]
                        stats = fit(em[0], em[1], fit_groups, ref)
                        delta, bias = -stats["precision_delta"], stats["constant"]
                        if protocol == "same_gender":
                            ez = [
                                embed(x, state, s, ref)
                                for s, x in zip(SIDES, donors, strict=True)
                            ]
                            delta, bias = align(stats, em, ez, groups, donor_groups)
                        if arm == "expanded":
                            bundle = bundles[protocol]
                            for side in SIDES:
                                for key in (
                                    "0.weight",
                                    "0.bias",
                                    "1.weight",
                                    "1.bias",
                                    "1.running_mean",
                                    "1.running_var",
                                ):
                                    bundle[f"{tag}.{side}.{key}"] = state[
                                        f"{side}.net.{key}"
                                    ].numpy()
                            bundle[tag + ".mean"] = stats["mean"]
                            bundle[f"{tag}.{track}.delta"] = delta
                            bundle[f"{tag}.{track}.bias"] = bias
                        q = pairs[f"fold{fold}_{protocol}_pairs"].astype(int)
                        assert all(
                            set(official_ids[j][q[:, j]]) <= set(split["outer_ids"])
                            for j in (0, 1)
                        )
                        np.testing.assert_array_equal(
                            official_ids[0][q[:, 0]] == official_ids[1][q[:, 1]],
                            q[:, 2],
                        )
                        key = f"fold{fold}_{protocol}"
                        labels[key] = q[:, 2]
                        results[arm][key].append(
                            score(em, q, stats["mean"], delta, bias)
                        )
                        for domain, sources in ext_raw.items():
                            key = domain + "_" + protocol
                            q = qext[key]
                            labels[key] = np.tile(q[:, 2], len(sources))
                            pieces = []
                            for matr in sources:
                                projected = [
                                    embed(x, state, s, ref)
                                    for s, x in zip(SIDES, matr, strict=True)
                                ]
                                pieces.append(
                                    score(projected, q, stats["mean"], delta, bias)
                                )
                            results[arm][key].append(np.concatenate(pieces))
            LOG.info("完成第 %d 折的三种子离线对照", fold)
    save_pred = {}
    report: dict[str, dict[str, float]] = {a: {} for a in (*arms, "equal")}
    for key, y in labels.items():
        save_pred["labels_" + key] = y
        sa, sb = [np.mean(results[arm][key], axis=0) for arm in arms]
        for arm, v in (("old", sa), ("expanded", sb), ("equal", (sa + sb) / 2)):
            report[arm][key] = eer(y, v)
            save_pred[f"{arm}_{key}"] = v
    for by_key in report.values():
        for p in ref["protocols"]:
            by_key["flag_" + p] = float(
                np.mean([by_key[f"fold{f}_{p}"] for f in ref["folds"]])
            )
    previous = HERE.parent / "11-18-跨录制协方差迁移/output"
    crema = read(previous / "纯英语移除eres外部复核.json")
    for p, file, variant, crema_arm in (
        ("standard", "纯英语移除eres共同方向", "base", "remove"),
        ("same_gender", "纯英语投影共同方向", "aligned", "full"),
    ):
        old = read(previous / (file + ".json"))
        for k, prev in [
            (f"fold{f}_{p}", old[f"{f}_{p}"][variant]) for f in ref["folds"]
        ] + [("bn_" + p, old["bn_" + p][variant]), ("crema_" + p, crema[p][crema_arm])]:
            np.testing.assert_allclose(report["old"][k], prev, atol=1e-8)
        np.savez_compressed(
            dest / (p + "_parameters.npz"),
            allow_pickle=False,
            tags=tags,
            epsilon=ref["bn_epsilon"],
            **bundles[p],
        )
    np.savez_compressed(dest / "逐对验证分数.npz", allow_pickle=False, **save_pred)
    save(
        dest / "同协议复核.json",
        {
            "metrics": report,
            "calibration_source": "official_unique_media"
            if unique is not None
            else "official_trial_rows",
            "candidate_positive_mode": read(run / "实际配置.json").get(
                "positive_mode", "all"
            ),
            "trials": {
                k: {"count": len(y), "positive": int(y.sum())}
                for k, y in labels.items()
            },
            "flag_aggregation": "五折分别对三种子分数取均值，计算 EER 后再平均五折",
            "external_aggregation": "十五个模型分数等权平均，再计算 EER",
            "elapsed": time.monotonic() - start,
        },
    )
    for arm, res in report.items():
        LOG.info(
            "%s %s",
            arm,
            {k: round(v, 4) for k, v in res.items() if not k.startswith("fold")},
        )


if __name__ == "__main__":
    main()
