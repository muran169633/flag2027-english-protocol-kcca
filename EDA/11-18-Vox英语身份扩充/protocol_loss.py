"""在相同训练来源内构造条件负例，保持每个脸端身份的权重一致。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


def negative_masks(
    names: list[str],
    classes: np.ndarray,
    strata: dict[str, tuple[str, str]],
    mode: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """真实条件和随机对照保留相同锚点、来源及逐锚点负例数。"""
    assert mode in ("same_gender", "count_matched")
    sources = np.asarray([strata[n][0] for n in names])[classes]
    genders = np.asarray([strata[n][1] for n in names])[classes]
    same_source = sources[:, :, None] == sources[:, None, :]
    known = genders != ""
    available = same_source & known[:, :, None] & known[:, None, :]
    available &= ~np.eye(classes.shape[1], dtype=bool)[None, :, :]
    target = available & (genders[:, :, None] == genders[:, None, :])
    count = target.sum(axis=-1)
    mask = target
    if mode == "count_matched":
        rng = np.random.default_rng(seed)
        priority = np.where(available, rng.random(available.shape), np.inf)
        order = priority.argsort(axis=-1)
        mask = np.zeros_like(available)
        np.put_along_axis(
            mask,
            order,
            np.arange(classes.shape[1])[None, None, :] < count[:, :, None],
            axis=-1,
        )
    np.testing.assert_array_equal(mask.sum(axis=-1), count)
    assert not (mask & ~available).any() and np.all((count > 0).sum(1) > 0)
    return mask, {
        "anchor_groups": np.count_nonzero(count),
        "negative_identity_pairs": int(mask.sum()),
        "true_same_gender_negative_pairs": int((mask & target).sum()),
    }


def condition_loss(
    calibration: Any,
    f: torch.Tensor,
    v: torch.Tensor,
    classes: torch.Tensor,
    count: int,
    weights: torch.Tensor | None,
    mask: torch.Tensor,
) -> torch.Tensor:
    n = len(classes)
    sim = calibration.log_scale.exp().clamp(*calibration.bounds) * (
        f @ v.T - calibration.center
    )
    # [身份, 人脸样本, 身份, 语音样本]，每个已核实身份各贡献一份正例及负例损失。
    blocks = sim.reshape(n, count, n, count)
    positives = F.softplus(-blocks[classes, :, classes, :])
    per_pos = (
        positives.mean(dim=(1, 2))
        if weights is None
        else (positives * weights).sum(dim=(1, 2))
    )
    pair_neg = F.softplus(blocks).mean(dim=(1, 3))
    counts = mask.sum(1)
    active = counts > 0
    per_neg = (pair_neg * mask).sum(1) / counts.clamp_min(1)
    return (per_pos[active].mean() + per_neg[active].mean()) / 2
