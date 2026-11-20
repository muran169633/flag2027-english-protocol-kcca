"""用训练身份两端的核矩阵拟合正则化典型相关方向。"""

from __future__ import annotations

import numpy as np
from media_common import HERE, REF, SIDES, read

SPEC = read(HERE / "英语训练人物核对齐配置.json")


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def normalized(x: np.ndarray, side: str) -> np.ndarray:
    """每个编码器特征块等权，所得每行长度至多为一。"""
    sizes = REF[side + "_block_dimensions"]
    assert x.ndim == 2 and x.shape[1] == sum(sizes)
    blocks = np.split(np.asarray(x, dtype=np.float32), np.cumsum(sizes)[:-1], axis=1)
    return np.column_stack([unit(t) for t in blocks]) / np.float32(np.sqrt(len(sizes)))


def kernel(x: np.ndarray, anchors: np.ndarray, width: float) -> np.ndarray:
    dot = np.asarray(x, dtype=np.float64) @ anchors.T
    if width == 0:
        return dot
    d = (
        np.sum(np.square(x, dtype=np.float64), axis=1)[:, None]
        + np.sum(anchors**2, axis=1)
        - 2 * dot
    )
    return np.exp(-np.maximum(d, 0) / width)


def shuffled(strata: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = np.arange(len(strata))
    for cohort in np.unique(strata):
        ix = rng.permutation(np.flatnonzero(strata == cohort))
        order[ix] = np.roll(ix, 1)
    assert np.array_equal(strata[order], strata) and len(np.unique(order)) == len(order)
    return order


def remove_means(gram: np.ndarray, groups: np.ndarray) -> tuple:
    """在训练核空间中去除各组中心与总中心之差，返回可用于新输入的系数。"""
    levels = np.unique(groups)
    member = (groups[:, None] == levels).astype(float)
    contrasts = member / member.sum(0) - 1 / len(groups)
    cov = contrasts.T @ gram @ contrasts
    e, u = np.linalg.eigh((cov + cov.T) / 2)
    ok = e > max(1e-12, float(e[-1]) * 1e-10)
    z = contrasts @ (u[:, ok] / np.sqrt(e[ok]))
    k = gram @ z
    residual = gram - k @ k.T
    np.testing.assert_allclose(contrasts.T @ residual, 0, rtol=0, atol=1e-9)
    return residual, z, k, contrasts


def fit(
    centers: list[np.ndarray], strata: np.ndarray, method: str, seed: int
) -> dict[str, np.ndarray]:
    n = len(strata)
    order = shuffled(strata, seed) if method == "permuted" else np.arange(n)
    groups = None
    if method == "remove_source":
        groups = np.array([g.split("/")[0] for g in strata])
    elif method == "remove_source_gender":
        groups = strata
    elif method == "remove_random":
        groups = np.random.default_rng(seed).permutation(strata)
    elif method not in ("rbf", "linear", "permuted"):
        raise ValueError(method)
    vals = {"permutation": order}
    whitened, bases = [], []
    for side, x in zip(SIDES, centers, strict=True):
        anchors = x.astype(np.float64)
        ix, jx = np.triu_indices(n, 1)
        dot = anchors @ anchors.T
        ds = dot.diagonal()[:, None] + dot.diagonal()[None, :] - 2 * dot
        width = float(np.median(ds[ix, jx])) if method != "linear" else 0.0
        assert width > 1e-12 or method == "linear"
        gram = kernel(anchors, anchors, width)
        means = gram.mean(0)
        centered = gram - means[:, None] - means[None, :] + means.mean()
        original = centered
        removal = remove_means(centered, groups) if groups is not None else None
        if removal is not None:
            centered = removal[0]
        e, u = np.linalg.eigh(centered)
        ok = e > max(1e-12, float(e[-1]) * 1e-10)
        e, u = e[ok], u[:, ok]
        ridge = SPEC["ridge_ratio"] * np.trace(centered)
        assert ridge > 0 and len(e) > 1
        # 白化后的训练特征 [人物, 核主成分]，对应新样本的基为 [训练人物, 核主成分]。
        whitened.append(u * np.sqrt(e / (e + ridge)))
        b = u / np.sqrt(e * (e + ridge))
        if removal is not None:
            _, z, k, contrasts = removal
            # 将训练正交投影并入导出系数；新输入无需提供性别或来源。
            effective = b - z @ (k.T @ b)
            np.testing.assert_allclose(original @ effective, centered @ b, atol=1e-9)
            np.testing.assert_allclose(contrasts.T @ original @ effective, 0, atol=1e-9)
            vals[side + "_removed_rank"] = np.array(z.shape[1])
            b = effective
        bases.append(b)
        vals[side + "_anchors"], vals[side + "_width"], vals[side + "_mean"] = (
            anchors,
            np.array(width),
            means,
        )
    ux, cor, vy = np.linalg.svd(whitened[0].T @ whitened[1][order], full_matrices=False)
    d = min(SPEC["dimensions"], len(cor))
    assert np.all(cor >= 0) and cor[0] <= 1 + 1e-9
    vals["correlations"] = cor[:d]
    for side, b, directions in zip(SIDES, bases, (ux, vy.T), strict=True):
        vals[side + "_projection"] = (b @ directions[:, :d]) * np.sqrt(cor[:d])
    return vals


def project(x: np.ndarray, side: str, params: dict[str, np.ndarray]) -> np.ndarray:
    k = kernel(x, params[side + "_anchors"], float(params[side + "_width"]))
    m = params[side + "_mean"]
    centered = k - k.mean(1, keepdims=True) - m + m.mean()
    return unit(centered @ params[side + "_projection"])


def scores(em: list[np.ndarray], q: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", em[0][q[:, 0]], em[1][q[:, 1]])
