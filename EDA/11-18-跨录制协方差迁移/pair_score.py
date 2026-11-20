"""读取冻结的投影及高斯参数，输出每对脸声的距离，越小越匹配。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


class PairScorer:
    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as archive:
            self.params = dict(archive)

    def distance(self, face: np.ndarray, voice: np.ndarray, track: str) -> np.ndarray:
        if track not in ("no_gender", "gender"):
            raise ValueError("协议应为 no_gender 或 gender")
        if (
            face.ndim != 2
            or voice.ndim != 2
            or face.shape != (len(voice), 1280)
            or voice.shape[1] != 2944
        ):
            raise ValueError("输入应为行数相同的 [N,1280] 与 [N,2944] 特征")
        if not (np.isfinite(face).all() and np.isfinite(voice).all()):
            raise ValueError("输入包含非有限特征")
        p = self.params
        total = np.zeros(len(face))
        for tag in p["tags"]:
            embeddings = []
            for side, x in (("face", face), ("voice", voice)):
                key = f"{tag}.{side}."
                h = (
                    np.asarray(x, dtype=np.float64) @ p[key + "0.weight"].T
                    + p[key + "0.bias"]
                )
                h = (h - p[key + "1.running_mean"]) / np.sqrt(
                    p[key + "1.running_var"].astype(np.float64) + float(p["epsilon"])
                )
                h = np.maximum(h * p[key + "1.weight"] + p[key + "1.bias"], 0)
                embeddings.append(
                    h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
                )
            joint = np.column_stack(embeddings) - p[tag + ".mean"]
            total -= (
                0.5 * np.sum((joint @ p[f"{tag}.{track}.delta"]) * joint, axis=1)
                + p[f"{tag}.{track}.bias"]
            )
        return total / len(p["tags"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "features", type=Path, help="含逐对对齐的 face 与 voice 矩阵的 NPZ"
    )
    parser.add_argument("track", choices=("no_gender", "gender"))
    args = parser.parse_args()
    scorer = PairScorer(HERE / "output/逐对评分参数.npz")
    with np.load(args.features) as z:
        for x in scorer.distance(z["face"], z["voice"], args.track):
            print(f"{x:.12g}")


if __name__ == "__main__":
    main()
