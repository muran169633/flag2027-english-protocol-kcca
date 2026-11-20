"""全矩阵构建同簇条件下、排除两端见证的局部支持一致度。"""

from __future__ import annotations

from typing import Any

import numpy as np


def matrix(s: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    p = s["voice_probability"].copy()
    np.fill_diagonal(p, 0)
    numerator = np.full(p.shape, cfg["profile_pseudocount"])
    denominator = numerator.copy()
    memo: dict[tuple[int, ...], np.ndarray] = {}
    for labels in s["voice_labels"]:
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            if len(members) < 2:
                continue
            key = tuple(members.tolist())
            if key not in memo:
                profile = p[np.ix_(members, members)]
                square = np.sum(profile**2, axis=1)
                gram = profile @ profile.T
                norm = np.sqrt(
                    np.maximum(square[:, None] - profile**2, 0)
                    * np.maximum(square[None, :] - profile.T**2, 0)
                )
                result = np.ones(norm.shape)
                valid = norm > cfg["profile_norm_floor"]
                result[valid] = np.clip(gram[valid] / norm[valid], 0, 1)
                memo[key] = result
            index = np.ix_(members, members)
            weight = 1 / (len(members) - 1)
            numerator[index] += weight * memo[key]
            denominator[index] += weight
    return np.log(numerator / denominator)


def independent_matrix(s: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    raw = s["voice_probability"]
    n = len(raw)
    out = np.full((n, n), cfg["profile_pseudocount"])
    count = out.copy()
    for row in s["voice_labels"]:
        for label in np.unique(row):
            group = np.flatnonzero(row == label)
            if len(group) < 2:
                continue
            # 用逐行内积与端点显式扣除核验批量Gram路径。
            profiles = raw[np.ix_(group, group)].copy()
            np.fill_diagonal(profiles, 0)
            for i, a in enumerate(group):
                squared = np.sum(profiles**2, axis=1) - profiles[:, i] ** 2
                own = np.sum(profiles[i] ** 2) - profiles[i] ** 2
                norm = np.sqrt(np.maximum(own * squared, 0))
                value = np.ones(len(group))
                valid = norm > cfg["profile_norm_floor"]
                value[valid] = np.clip(
                    (profiles @ profiles[i])[valid] / norm[valid], 0, 1
                )
                out[a, group] += value / (len(group) - 1)
                count[a, group] += 1 / (len(group) - 1)
    return np.log(out / count)
