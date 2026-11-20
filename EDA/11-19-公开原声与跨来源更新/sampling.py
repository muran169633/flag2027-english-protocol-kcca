"""复用逐行抽样、平衡身份监督及跨录制训练的数学实现。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from media_common import ROOT, SIDES, read
from media_network import Objective
from torch.nn import functional as F

LOG = logging.getLogger(__name__)
Pools = dict[str, tuple[np.ndarray, np.ndarray]]


def feature_matrix(blocks: list[list[str]]) -> np.ndarray:
    columns = []
    for path, key in blocks:
        with np.load(ROOT / path, allow_pickle=False) as z:
            columns.append(z[key])
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def load_sources(
    ref: dict[str, Any], screened: dict[str, Any] | None
) -> tuple[list[torch.Tensor], list[list[dict[str, str]]], Pools]:
    raw = []
    historical = []
    newly_selected = []
    blocks = ref["feature_blocks"]
    for side in SIDES:
        old = read(ROOT / ref["metadata"][side][0])
        offset = len(old)
        historical.append(old)
        parts = (
            [[blocks["face"][i], blocks["face_additional"][i]] for i in (0, 1)]
            if side == "face"
            else [blocks["voice_train"], blocks["voice_extra"]]
        )
        first = feature_matrix(parts[0])
        assert len(first) == offset and first.shape[1] == sum(
            ref[f"{side}_block_dimensions"]
        )
        if screened is None:
            newly_selected.append({})
            raw.append(torch.from_numpy(first).to(ref["device"]))
            continue
        new = read(ROOT / ref["metadata"][side][1])
        sel = np.array(screened[f"selected_{side}_rows"], dtype=np.int64)
        assert np.array_equal(np.unique(sel), sel) and all(
            new[i]["src"] == "vox2t" for i in sel
        )
        grouped: dict[str, list[int]] = {}
        for i in sel:
            grouped.setdefault(new[i]["id"], []).append(int(i) + offset)
        newly_selected.append(grouped)
        second = feature_matrix(parts[1])
        assert len(second) == len(new) and first.shape[1] == second.shape[1]
        tensor = torch.from_numpy(np.concatenate((first, second))).to(ref["device"])
        raw.append(tensor)
        LOG.info("载入 %s，历史行 %d，外部行 %d", side, offset, len(new))
    assert newly_selected[0].keys() == newly_selected[1].keys()
    extra = {
        n: (np.asarray(v), np.asarray(newly_selected[1][n]))
        for n, v in newly_selected[0].items()
    }
    assert len(extra) == (0 if screened is None else screened["identities"])
    return raw, historical, extra


def draw_rows(
    pools: list[np.ndarray], classes: np.ndarray, seed: int, count: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = np.empty((*classes.shape, count), dtype=np.int64)
    for k, pool in enumerate(pools):
        use = classes == k
        ranks = rng.random((int(use.sum()), len(pool)))
        rows[use] = pool[np.argpartition(ranks, count - 1, axis=1)[:, :count]]
    return rows.reshape(len(classes), -1)


def recording_weights(
    indices: list[np.ndarray],
    videos: list[np.ndarray],
    identities: int,
    count: int,
    mode: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    # [步, 身份, 人脸样本, 语音样本]，两侧录制字段均存在时才判为同视频。
    f, v = [
        video[ix].reshape(len(ix), identities, count)
        for ix, video in zip(indices, videos, strict=True)
    ]
    same = (f[:, :, :, None] >= 0) & (f[:, :, :, None] == v[:, :, None, :])
    eligible = ~same.reshape(len(f), identities, count * count)
    fallback = ~eligible.any(axis=-1)
    eligible[fallback] = True
    kept = eligible.sum(axis=-1)
    if mode == "shuffled":
        perm = np.random.default_rng(seed).random(eligible.shape).argsort(axis=-1)
        eligible = np.take_along_axis(eligible, perm, axis=-1)
        np.testing.assert_array_equal(eligible.sum(axis=-1), kept)
    weights = (eligible / kept[:, :, None]).reshape(same.shape).astype(np.float32)
    np.testing.assert_allclose(weights.sum(axis=(2, 3)), 1, atol=1e-6)
    return weights, {
        "all_positive_pairs": same.size,
        "known_same_video_pairs": int(same.sum()),
        "excluded_pairs": int(eligible.size - eligible.sum()),
        "remaining_same_video_pairs": int(np.sum(same & (weights > 0))),
        "all_same_fallback_groups": int(fallback.sum()),
        "identity_groups": fallback.size,
    }


def positive_loss(
    obj: Objective,
    f: torch.Tensor,
    v: torch.Tensor,
    weights: torch.Tensor,
    classes: torch.Tensor,
    negative: torch.Tensor,
) -> torch.Tensor:
    n, k, _ = weights.shape
    logits = obj.log_scale.exp().clamp(*obj.bounds) * (f @ v.T - obj.center)
    positives = logits.reshape(n, k, n, k)[classes, :, classes, :]
    return 0.5 * (
        (F.softplus(-positives) * weights).sum(dim=(1, 2)).mean()
        + F.softplus(logits[negative]).mean()
    )
