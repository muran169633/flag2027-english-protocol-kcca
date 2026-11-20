"""装载冻结英语评分器，逐对输出距离，并对原 Evaluation 清单生成本地研究包。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from anchor_model import SPEC, normalized, project
from media_common import HERE, ROOT, SIDES, TRACKS, arrays, read, setup, sha, write

sys.path.insert(0, str(HERE.parent / "11-18-跨录制协方差迁移"))
from package_english_standard_no_eres import write_package
from pair_score import PairScorer

PACK = read(HERE / "核方法打包配置.json")
LOG = logging.getLogger(__name__)


class FrozenPairMatcher:
    def __init__(self) -> None:
        trained = ROOT / PACK["training"]
        native = read(trained / "冻结清单.json")
        base = ROOT / PACK["baselines"]
        spec = read(base / "评分前输入.json")
        assert (
            sha(base / "评分前输入.json") == read(base / "完成.json")["source_sha256"]
        )
        assert (
            spec["training_lock"] == sha(trained / "冻结清单.json")
            and native["spec"] == SPEC
        )
        self.fingerprint = {}
        self.gaussian = {}
        for tr, method in (
            ("standard", "count_matched"),
            ("same_gender", "same_gender"),
        ):
            banks = []
            for entry in (
                spec["parameters"][tr]["fixed"],
                spec["parameters"][tr][method],
            ):
                p = ROOT / entry["path"]
                assert sha(p) == entry["sha256"]
                banks.append(PairScorer(p))
                self.fingerprint[entry["path"]] = entry["sha256"]
            self.gaussian[TRACKS[tr]] = banks
        self.kernels = []
        for f in native["folds"]:
            name = f"development/fold{f}"
            p = trained / name / "rbf.npz"
            digest = native["entries"][name]["fit"]["models"]["rbf"]["sha256"]
            assert sha(p) == digest
            self.fingerprint[str(p.relative_to(ROOT))] = digest
            self.kernels.append(arrays(p))

    def distance(self, face: np.ndarray, voice: np.ndarray, track: str) -> np.ndarray:
        assert len(face) == len(voice)
        base = np.mean(
            [bank.distance(face, voice, track) for bank in self.gaussian[track]], axis=0
        )
        norm = [
            normalized(x, side) for side, x in zip(SIDES, [face, voice], strict=True)
        ]
        logits = []
        protocol = {v: k for k, v in TRACKS.items()}[track]
        for p in self.kernels:
            a, b = [project(x, s, p) for s, x in zip(SIDES, norm, strict=True)]
            scale, shift = p["calibration_" + protocol]
            logits.append(scale * np.einsum("ij,ij->i", a, b) + shift)
        return base - np.mean(logits, axis=0)


def run(out: Path) -> None:
    from anchor_baselines import sources

    out.mkdir(parents=True, exist_ok=False)
    setup(out, "研究包核验.log")
    frozen = ROOT / PACK["training"]
    val = ROOT / PACK["validation"]
    result = read(val / "核方向与分协议结果.json")
    assert (
        sha(val / "评分前锁定.json") == result["lock_sha256"]
        and sha(val / "所有独立配对分数.npz") == result["score_sha256"]
    )
    assert (
        PACK["selected"]
        == SPEC["primary"]
        == read(val / "评分前锁定.json")["design"]["primary"]
    )
    matcher = FrozenPairMatcher()
    ds, cells, _ = sources(read(frozen / "冻结清单.json"), "development")
    checks = {}
    with np.load(val / "所有独立配对分数.npz", allow_pickle=False) as data:
        for k in PACK["validation_cells"]:
            c = cells[k]
            media = ds[c.domain]
            ix = np.unique(
                np.linspace(0, len(c.q) - 1, PACK["samples_per_cell"], dtype=int)
            )
            q = c.q[ix]
            face, voice = media.face[q[:, 0]], media.voice[q[:, 1]]
            code = TRACKS[c.track]
            s = matcher.distance(face, voice, code)
            ref = -data["development/" + PACK["selected"] + "__" + k][ix]
            error = float(np.max(np.abs(s - ref)))
            single = matcher.distance(face[:1], voice[:1], code)
            reversed_distance = matcher.distance(face[::-1], voice[::-1], code)[::-1]
            assert error < 1e-4 and abs(float(single[0]) - s[0]) < 1e-4
            np.testing.assert_allclose(reversed_distance, s, atol=1e-4, rtol=0)
            checks[k] = {
                "pairs": len(q),
                "max_abs": error,
                "single": abs(float(single[0]) - s[0]),
            }
    package = ROOT / PACK["zip"]
    report = write_package(
        package, {code: matcher.distance for code in TRACKS.values()}
    )
    original = read(ROOT / PACK["reference_package"])
    assert report["pairs"] == original["pairs"] and report["sha256"] == sha(package)
    write(
        out / "研究候选验收.json",
        {
            **report,
            "parameters": matcher.fingerprint,
            "validation_sha256": sha(val / "核方向与分协议结果.json"),
            "config": PACK,
            "validation_replay": checks,
            "research_only": True,
            "platform_submission": None,
        },
    )
    LOG.info(
        "研究包 %s，%d 个配对，SHA256 %s",
        PACK["zip"],
        sum(report["pairs"].values()),
        report["sha256"],
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("out", type=Path)
    run(cli.parse_args().out)
