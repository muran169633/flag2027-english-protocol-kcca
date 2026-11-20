"""复用既有投影器、正则化高斯拟合及跨录制协方差变换的数学实现。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.covariance import OAS

Array = NDArray[Any]
ArrayPair = tuple[np.ndarray, np.ndarray]


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


def fit(
    face: Array, voice: Array, groups: list[tuple[Array, Array]], cfg: dict[str, Any]
) -> dict[str, Array]:
    """同一身份的人脸、声音分别统计，避免假设媒体行一一配对。"""
    dim = face.shape[1]
    wf = np.zeros((dim, dim))
    wv = np.zeros_like(wf)
    means = []
    for fi, vi in groups:
        assert len(fi) >= 2 and len(vi) >= 2
        f, v = face[fi], voice[vi]
        mf, mv = f.mean(0), v.mean(0)
        rf, rv = f - mf, v - mv
        wf += rf.T @ rf / len(fi)
        wv += rv.T @ rv / len(vi)
        means.append(np.concatenate([mf, mv]))
    within = np.zeros((2 * dim, 2 * dim))
    within[:dim, :dim] = wf / len(groups)
    within[dim:, dim:] = wv / len(groups)
    model = OAS(store_precision=False).fit(np.stack(means))
    same = model.covariance_ + within
    same += np.eye(len(same)) * np.trace(same) / len(same) * cfg["jitter_relative"]
    different = same.copy()
    different[:dim, dim:] = 0
    different[dim:, :dim] = 0
    ls, ld = np.linalg.cholesky(same), np.linalg.cholesky(different)
    identity = np.eye(len(same))
    ps = np.linalg.solve(ls.T, np.linalg.solve(ls, identity))
    pd = np.linalg.solve(ld.T, np.linalg.solve(ld, identity))
    delta = ps - pd
    return {
        "mean": model.location_,
        "same": same,
        "different": different,
        "precision_delta": (delta + delta.T) / 2,
        "constant": np.asarray(np.log(np.diag(ld)).sum() - np.log(np.diag(ls)).sum()),
        "shrinkage": np.asarray(model.shrinkage_),
    }


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


def score(
    em: list[np.ndarray],
    q: np.ndarray,
    mu: np.ndarray,
    delta: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    x = np.column_stack((em[0][q[:, 0]], em[1][q[:, 1]])) - mu
    return 0.5 * np.sum((x @ delta) * x, axis=1) + bias
