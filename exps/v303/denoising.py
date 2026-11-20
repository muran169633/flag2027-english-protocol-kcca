"""一次连续软边去噪；每条外层查询完整排除于中间后验。"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit


def edge_kernels(
    pf: np.ndarray, pv: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(np.unique(edges, axis=0)) != len(edges):
        raise AssertionError("去噪图需要唯一实体边")
    left = pf[np.ix_(edges[:, 0], edges[:, 0])]
    right = pv[np.ix_(edges[:, 1], edges[:, 1])]
    return left, right, left * right


def first_pass(
    pf: np.ndarray, pv: np.ndarray, edges: np.ndarray, cfg: dict[str, Any]
) -> np.ndarray:
    left, right, joint = edge_kernels(pf, pv, edges)
    remaining = len(edges) - 1
    if remaining <= 0:
        return np.zeros(len(edges))
    support = np.maximum(joint.sum(axis=1) - 1, 0)
    background = (
        np.maximum(left.sum(axis=1) - 1, 0)
        * np.maximum(right.sum(axis=1) - 1, 0)
        / remaining
    )
    alpha = cfg["graph_pseudocount"]
    return np.log(support + alpha) - np.log(background + alpha)


def infer(
    pf: np.ndarray,
    pv: np.ndarray,
    edges: np.ndarray,
    queries: np.ndarray,
    model: dict[str, float],
    cfg: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    left, right, joint = edge_kernels(pf, pv, edges)
    location = {tuple(q): i for i, q in enumerate(edges.tolist())}
    totals = [matrix.sum(axis=1) for matrix in [left, right, joint]]
    alpha = cfg["graph_pseudocount"]
    predictions = {name: [] for name in ["dense", "constant_prior", "denoised"]}
    mean_weight, effective_mass = [], []
    for query in queries:
        index = location[tuple(query.tolist())]
        if len(edges) <= 1:
            for value in predictions.values():
                value.append(0.0)
            mean_weight.append(0.0)
            effective_mass.append(0.0)
            continue
        if len(edges) <= 2:
            inner_score = np.zeros(len(edges))
        else:
            l = np.maximum(totals[0] - 1 - left[:, index], 0)
            r = np.maximum(totals[1] - 1 - right[:, index], 0)
            support = np.maximum(totals[2] - 1 - joint[:, index], 0)
            background = l * r / (len(edges) - 2)
            inner_score = np.log(support + alpha) - np.log(background + alpha)
        weights = expit(model["coefficient"] * inner_score + model["intercept"])
        mean_weight.append(float(np.mean(np.delete(weights, index))))
        for name, weight in [
            ("dense", np.ones(len(edges))),
            ("constant_prior", np.full(len(edges), model["positive_prior"])),
            ("denoised", weights),
        ]:
            weight[index] = 0
            mass = weight.sum()
            count = float(weight @ joint[index])
            background = (
                float((weight @ left[index]) * (weight @ right[index]) / mass)
                if mass > 0
                else 0.0
            )
            predictions[name].append(np.log(count + alpha) - np.log(background + alpha))
            if name == "denoised":
                effective_mass.append(float(mass))
    return {name: np.asarray(value) for name, value in predictions.items()}, {
        "mean_inner_edge_probability": np.asarray(mean_weight),
        "effective_positive_edge_mass": np.asarray(effective_mass),
    }


def physical(
    pf: np.ndarray,
    pv: np.ndarray,
    edges: np.ndarray,
    query: np.ndarray,
    model: dict[str, float],
    cfg: dict[str, Any],
) -> float:
    remaining = edges[~np.all(edges == query, axis=1)]
    if len(remaining) != len(edges) - 1:
        raise AssertionError("去噪查询边未唯一排除")
    values = first_pass(pf, pv, remaining, cfg)
    weight = expit(model["coefficient"] * values + model["intercept"])
    left = pf[query[0], remaining[:, 0]]
    right = pv[query[1], remaining[:, 1]]
    count = weight @ (left * right)
    background = (
        (weight @ left) * (weight @ right) / weight.sum() if weight.sum() > 0 else 0.0
    )
    return float(
        np.log(count + cfg["graph_pseudocount"])
        - np.log(background + cfg["graph_pseudocount"])
    )
