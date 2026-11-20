"""Frozen feature projection and Gaussian pair scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

Array = NDArray[Any]


@torch.inference_mode()
def embed(x: Array, state: dict[str, Any], side: str, cfg: dict[str, Any]) -> Array:
    """按冻结 Linear、BN、ReLU、L2 前向，Dropout 在推断关闭。"""
    prefix = side + ".net."
    p = {
        k[len(prefix) :]: v.to(cfg["device"])
        for k, v in state.items()
        if k.startswith(prefix)
    }
    chunks = []
    for start in range(0, len(x), cfg["batch_size"]):
        a = torch.from_numpy(x[start : start + cfg["batch_size"]]).to(cfg["device"])
        a = F.linear(a, p["0.weight"], p["0.bias"])
        a = F.batch_norm(
            a,
            p["1.running_mean"],
            p["1.running_var"],
            p["1.weight"],
            p["1.bias"],
            training=False,
            eps=cfg["bn_epsilon"],
        )
        # [媒体数, 128]：每条媒体独立归一化。
        chunks.append(F.normalize(F.relu(a), dim=1).cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def score(face: Array, voice: Array, pairs: Array, stats: dict[str, Array]) -> Array:
    """逐配对 LLR，同时用独立 Cholesky 二次型验证。"""
    x = np.concatenate([face[pairs[:, 0]], voice[pairs[:, 1]]], axis=1) - stats["mean"]
    result = (
        -0.5 * np.sum((x @ stats["precision_delta"]) * x, axis=1) + stats["constant"]
    )
    a = np.linalg.solve(np.linalg.cholesky(stats["same"]), x.T)
    b = np.linalg.solve(np.linalg.cholesky(stats["different"]), x.T)
    direct = -0.5 * (np.sum(a * a, axis=0) - np.sum(b * b, axis=0)) + stats["constant"]
    assert np.max(abs(result - direct)) < 1e-8
    return result


def load_checkpoint(path: Path) -> dict[str, Any]:
    """仅允许原本地权重必要的NumPy字符串元数据。"""
    with torch.serialization.safe_globals(
        [np._core.multiarray.scalar, np.dtype, type(np.dtype("U1"))]
    ):
        return torch.load(path, map_location="cpu", weights_only=True)
