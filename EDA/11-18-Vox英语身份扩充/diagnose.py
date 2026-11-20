"""对相同配对成对比较，检查结论是否由个别身份或录制方向驱动。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from evaluate import eer
from train import HERE, ROOT, feature_matrix, read, save

SCORER_DIR = HERE.parent / "11-18-跨录制协方差迁移"
sys.path.insert(0, str(SCORER_DIR))
from pair_score import PairScorer

LOG = logging.getLogger(__name__)


def leave_one(
    f: np.ndarray, v: np.ndarray, y: np.ndarray, left: np.ndarray, right: np.ndarray
) -> dict[str, Any]:
    ids = np.union1d(f, v)
    diffs = []
    for n in ids:
        keep = (f != n) & (v != n)
        diffs.append(eer(y[keep], right[keep]) - eer(y[keep], left[keep]))
    diff = np.array(diffs)
    return {
        "identities": len(ids),
        "full_delta_pp": eer(y, right) - eer(y, left),
        "range_pp": [float(diff.min()), float(diff.max())],
        "better_tied_worse": [
            int(np.sum(diff < -1e-8)),
            int(np.sum(abs(diff) <= 1e-8)),
            int(np.sum(diff > 1e-8)),
        ],
        "per_identity": dict(zip(ids.tolist(), diff.tolist(), strict=True)),
    }


def benav(run: Path) -> dict[str, Any]:
    corpus = ROOT / "outputs/research/11-14-bengali-benav-pilot"
    lock = read(corpus / "locked_selection.json")
    names = [Path(r["repo_path"]).stem for r in lock["samples"]]
    words = np.asarray([Path(r["repo_path"]).parent.name for r in lock["samples"]])
    ids = np.asarray([r["speaker"] for r in lock["samples"]])
    for side in ("face", "voice"):
        records = read(corpus / (side + "s.json"))
        assert [r["id"] for r in records] == list(ids) and [
            Path(r["path"]).stem for r in records
        ] == names
    a, b = np.where(~np.eye(len(ids), dtype=bool))
    assert np.all(words[a] != words[b])
    target = ids[a] == ids[b]
    face = feature_matrix(
        [[str(corpus / "face_v399.npz"), key] for key in ("clip", "clipL")]
    )
    voice = feature_matrix(
        [
            [str(corpus / "voice_core_v399.npz"), key]
            for key in ("wavlm", "ecapa", "eres", "resnet")
        ]
        + [
            [
                str(
                    HERE.parent / "11-18-孟加拉语验证源/output/benav_voices_generic.npz"
                ),
                key,
            ]
            for key in ("whisper", "age6")
        ]
    )
    scores = {}
    for arm, path in (
        ("old", SCORER_DIR / "output/纯英语移除eres评分参数.npz"),
        ("expanded", run / "standard_parameters.npz"),
    ):
        model = PairScorer(path)
        s = -model.distance(face[a], voice[b], "no_gender")
        np.testing.assert_allclose(
            -model.distance(face[a[:1]], voice[b[:1]], "no_gender"),
            s[:1],
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            -model.distance(face[a[::-1]], voice[b[::-1]], "no_gender"),
            s[::-1],
            rtol=1e-10,
            atol=1e-10,
        )
        scores[arm] = s
    scores["equal"] = (scores["old"] + scores["expanded"]) / 2
    output = {}
    for arm, s in scores.items():
        ranks = {}
        for side, axis in (("face", a), ("voice", b)):
            rr = []
            for i in range(len(ids)):
                pos = s[(axis == i) & target].item()
                rr.append(int(1 + np.sum(s[(axis == i) & ~target] >= pos)))
            ranks[side] = {
                "rank1": rr.count(1),
                "total": len(rr),
                "mean_rank": float(np.mean(rr)),
            }
        output[arm] = ranks
    return {
        "identities": len(np.unique(ids)),
        "trials": len(target),
        "positive": int(target.sum()),
        "ranking": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    run = parser.parse_args().run.resolve()
    assert (
        Path(sys.executable).parent.parent
        == Path(sys.prefix)
        == Path(read(HERE / "config.json")["python_prefix"])
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(run / "身份复核.log"), logging.StreamHandler()],
    )
    gender_dir = HERE.parent / "11-18-孟加拉语验证源/output"
    trials = read(gender_dir / "七人固定跨录制诊断.json")["trials"]
    selected = read(HERE / "output/英语片段筛选.json")
    external = read(ROOT / "outputs/v380/features/vox2t_voices.json")
    extra_ids = {
        external[i]["id"].removeprefix("vox2t_")
        for i in selected["selected_voice_rows"]
    }
    extra_videos = {external[i]["video"] for i in selected["selected_voice_rows"]}
    if read(run / "实际配置.json").get("training_pool") == "original":
        extra_ids.clear()
        extra_videos.clear()
    bn_meta = read(gender_dir / "孟语候选独立语言复核.json")
    bn_ids = {r["identity"] for r in bn_meta}
    bn_videos = {Path(r["path"]).parent.name for r in bn_meta}
    assert extra_ids.isdisjoint(bn_ids) and extra_videos.isdisjoint(bn_videos)
    crema = ROOT / "outputs/research/11-13-msm-native-transfer/crema_stress"
    crema_ids = [
        np.asarray([r["id"] for r in read(crema / (side + "s.json"))])
        for side in ("face", "voice")
    ]
    result = {}
    with np.load(run / "逐对验证分数.npz") as pred:
        for protocol in ("standard", "same_gender"):
            rows = [r for r in trials if protocol == "standard" or r["same_gender"]] * 2
            with np.load(crema / f"cross_video_trials_{protocol}.npz") as z:
                q = z["triples"].astype(int)
            for domain, f, v in (
                (
                    "bn",
                    np.array([r["face_id"] for r in rows]),
                    np.array([r["voice_id"] for r in rows]),
                ),
                ("crema", crema_ids[0][q[:, 0]], crema_ids[1][q[:, 1]]),
            ):
                key = domain + "_" + protocol
                y = pred["labels_" + key]
                np.testing.assert_array_equal(f == v, y)
                left = pred["old_" + key]
                right = pred["expanded_" + key]
                item = leave_one(f, v, y, left, right)
                item["equal_vs_old"] = leave_one(f, v, y, left, pred["equal_" + key])
                if domain == "bn":
                    item["direction_delta_pp"] = [
                        eer(ys, rs) - eer(ys, ls)
                        for ys, ls, rs in zip(
                            np.split(y, 2),
                            np.split(left, 2),
                            np.split(right, 2),
                            strict=True,
                        )
                    ]
                result[key] = item
                LOG.info(
                    "%s 增量 %.3f 个百分点，逐人剔除改善/持平/退化 %s",
                    key,
                    item["full_delta_pp"],
                    item["better_tied_worse"],
                )
    result["benav"] = benav(run)
    result["vox_id_video_intersection"] = {
        "new_train_ids": len(extra_ids),
        "bn_ids": len(bn_ids),
        "shared_ids": 0,
        "shared_videos": 0,
    }
    save(run / "身份与独立孟语来源复核.json", result)
    LOG.info("BenAV 双向排名 %s", result["benav"]["ranking"])


if __name__ == "__main__":
    main()
