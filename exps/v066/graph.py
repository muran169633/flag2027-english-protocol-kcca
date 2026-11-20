"""冻结三路声纹的完整留一边图推断，不接受开发集身份标签。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering

Array = NDArray[Any]


def normalize(x: Array) -> Array:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def unique_media(values: Array, hashes: Array, rows: Array) -> tuple[Array, Array]:
    selected = np.unique(rows)
    _, inverse = np.unique(hashes[selected], return_inverse=True)
    pooled = np.zeros((int(inverse.max()) + 1, values.shape[1]))
    np.add.at(pooled, inverse, normalize(values[selected].astype(float)))
    lookup = np.full(len(values), -1, dtype=np.int64)
    lookup[selected] = inverse
    return normalize(pooled), lookup


def cluster(values: Array, threshold: float) -> Array:
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.int64)
    distance = np.clip(1 - values @ values.T, 0, 2)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    return AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=threshold,
    ).fit_predict(distance)


def infer(
    queries: dict[str, Array],
    face: Array,
    reps: dict[str, Array],
    face_hash: Array,
    voice_hash: Array,
    thresholds: dict[str, float],
    cfg: dict[str, Any],
) -> tuple[dict[str, Array], dict[str, Any], dict[str, Array]]:
    if any(q.ndim != 2 or q.shape[1] != 2 for q in queries.values()):
        raise ValueError("图推断仅接受 Nx2 无标签行号")
    joined = np.vstack(list(queries.values()))
    fv, fi = unique_media(face, face_hash, joined[:, 0])
    voices = {}
    vi = None
    for method in cfg["methods"]:
        voices[method], lookup = unique_media(reps[method], voice_hash, joined[:, 1])
        if vi is not None and not np.array_equal(vi, lookup):
            raise ValueError("三路声纹实体顺序不同")
        vi = lookup
    if vi is None:
        raise ValueError("声纹为空")
    mapped = np.c_[fi[joined[:, 0]], vi[joined[:, 1]]]
    edges, inverse = np.unique(mapped, axis=0, return_inverse=True)
    buffers = {m: [] for m in cfg["methods"]}
    diagnostics = []
    storage = {
        "face_lookup": fi,
        "voice_lookup": vi,
        "edges": edges,
        "inverse": inverse,
    }
    for index, factor in enumerate(cfg["factors"]):
        fc = cluster(fv, thresholds["face"] * factor)
        storage[f"face_{index}"] = fc
        for method in cfg["methods"]:
            vc = cluster(voices[method], thresholds[method] * factor)
            counts = np.zeros((int(fc.max()) + 1, int(vc.max()) + 1))
            np.add.at(counts, (fc[edges[:, 0]], vc[edges[:, 1]]), 1)
            a, b = fc[mapped[:, 0]], vc[mapped[:, 1]]
            observed = counts[a, b] - 1
            expected = (
                (counts.sum(axis=1)[a] - 1)
                * (counts.sum(axis=0)[b] - 1)
                / max(len(edges) - 1, 1)
            )
            if (observed < 0).any() or (expected < 0).any():
                raise AssertionError("查询边扣除错误")
            scores = np.log(
                (observed + cfg["smoothing"]) / (expected + cfg["smoothing"])
            )
            buffers[method].append(scores)
            storage[f"voice_{method}_{index}"] = vc
            diagnostics.append(
                {
                    "factor": factor,
                    "method": method,
                    "face_clusters": len(np.unique(fc)),
                    "voice_clusters": len(np.unique(vc)),
                }
            )
    averaged = {m: np.mean(values, axis=0) for m, values in buffers.items()}
    output = {}
    offset = 0
    for cell, q in queries.items():
        parts = [averaged[m][offset : offset + len(q)] for m in cfg["methods"]]
        output[cell] = np.maximum.reduce(
            [(v - v.mean()) / max(float(v.std()), 1e-8) for v in parts]
        )
        offset += len(q)
    return (
        output,
        {
            "face_entities": len(fv),
            "voice_entities": len(voices[cfg["methods"][0]]),
            "unique_edges": len(edges),
            "clusters": diagnostics,
            "query_edge_removed": True,
        },
        storage,
    )
