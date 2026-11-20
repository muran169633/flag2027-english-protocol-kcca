"""完整双重留边的正关联与排他身份消息。"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit

from .denoising import edge_kernels, first_pass


def messages(
    pf: np.ndarray,
    pv: np.ndarray,
    edges: np.ndarray,
    queries: np.ndarray,
    model: dict[str, float],
    cfg: dict[str, Any],
) -> np.ndarray:
    left, right, joint = edge_kernels(pf, pv, edges)
    conflict = left * (1 - right) + right * (1 - left)
    totals = [x.sum(axis=1) for x in [left, right, joint]]
    locations = {tuple(edge): i for i, edge in enumerate(edges.tolist())}
    alpha = cfg["graph_pseudocount"]
    rows = []
    for query in queries:
        index = locations[tuple(query.tolist())]
        support = np.maximum(totals[2] - 1 - joint[:, index], 0)
        background = (
            np.maximum(totals[0] - 1 - left[:, index], 0)
            * np.maximum(totals[1] - 1 - right[:, index], 0)
            / (len(edges) - 2)
        )
        p = expit(
            model["coefficient"]
            * (np.log(support + alpha) - np.log(background + alpha))
            + model["intercept"]
        )
        p[index] = 0
        agreement = joint[index].copy()
        agreement[index] = 0
        negative = conflict[index]
        rows.append(
            [
                float(agreement @ p / max(agreement.sum(), cfg["probability_floor"])),
                float(negative @ p / max(negative.sum(), cfg["probability_floor"])),
            ]
        )
    return np.asarray(rows)


def physical(
    pf: np.ndarray,
    pv: np.ndarray,
    edges: np.ndarray,
    query: np.ndarray,
    model: dict[str, float],
    cfg: dict[str, Any],
) -> np.ndarray:
    context = edges[~np.all(edges == query, axis=1)]
    inner = first_pass(pf, pv, context, cfg)
    p = expit(model["coefficient"] * inner + model["intercept"])
    left, right = pf[query[0], context[:, 0]], pv[query[1], context[:, 1]]
    agreement = left * right
    negative = left * (1 - right) + right * (1 - left)
    return np.array(
        [
            agreement @ p / max(agreement.sum(), cfg["probability_floor"]),
            negative @ p / max(negative.sum(), cfg["probability_floor"]),
        ]
    )
