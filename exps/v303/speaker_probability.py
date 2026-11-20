"""冻结共分配声纹头的全矩阵同人概率，不接受人物或配对标签。"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit, logit

from .local_connectivity import matrix


def probability(
    s: dict[str, np.ndarray], head: dict[str, Any], cfg: dict[str, Any]
) -> np.ndarray:
    k = len(np.unique(s["face"]))
    prior_log_odds = -np.log(max(k - 1, 1))
    original = s["voice_probability"]
    acoustic = (
        logit(np.clip(original, cfg["probability_floor"], 1 - cfg["probability_floor"]))
        - prior_log_odds
    )
    value = np.full(original.shape, head["intercept"] + prior_log_odds, dtype=float)
    for column, beta in zip(head["columns"], head["coefficient"], strict=True):
        if column == 0:
            feature = acoustic
        elif column == 1:
            vote = np.full(original.shape, cfg["vote_pseudocount"])
            for labels in s["voice_labels"]:
                vote += labels[:, None] == labels[None, :]
            vote /= len(s["voice_labels"]) + 2 * cfg["vote_pseudocount"]
            feature = logit(vote)
        elif column == 2:
            specific = np.full(original.shape, cfg["partition_pseudocount"])
            for labels in s["voice_labels"]:
                _, canonical = np.unique(labels, return_inverse=True)
                sizes = np.bincount(canonical)
                specific += (canonical[:, None] == canonical[None, :]) * (
                    (len(labels) - 1) / np.maximum(sizes[canonical, None] - 1, 1)
                )
            feature = np.log(
                specific / (len(s["voice_labels"]) + 2 * cfg["partition_pseudocount"])
            )
        elif column == 3:
            feature = matrix(s, cfg)
        else:
            raise ValueError("未知共分配声纹特征")
        value += beta * feature
    result = np.clip(
        expit((value + value.T) / 2),
        cfg["probability_floor"],
        1 - cfg["probability_floor"],
    )
    np.fill_diagonal(result, 1)
    return result
