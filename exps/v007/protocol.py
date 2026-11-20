"""固定身份配对与插值 EER，用于冻结模型的评分对照。"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_curve

from .config import Config
from .data import Features

Groups = dict[str, NDArray[np.int64]]


def groups(labels: list[str], allowed: list[str]) -> Groups:
    result: dict[str, list[int]] = defaultdict(list)
    keep = set(allowed)
    for index, value in enumerate(labels):
        if value in keep:
            result[value].append(index)
    return {key: np.asarray(value, dtype=np.int64) for key, value in result.items()}


def eer(labels: NDArray[np.int64], scores: NDArray[np.float32]) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    difference = fpr - (1 - tpr)
    right = int(np.flatnonzero(difference >= 0)[0])
    if right == 0:
        return float(fpr[0])
    left = right - 1
    weight = -difference[left] / (difference[right] - difference[left])
    return float(fpr[left] + weight * (fpr[right] - fpr[left]))


def validation_pairs(
    features: Features, speakers: list[str], config: Config, same_gender: bool
) -> NDArray[np.int64]:
    """严格保持父实验的 RNG、调用顺序和正负样本抽样规则。"""
    generator = np.random.default_rng(config.seed)
    face_groups = groups(features.face_speakers, speakers)
    voice_groups = groups(features.voice_speakers, speakers)
    eligible = speakers
    if same_gender:
        eligible = [
            speaker
            for speaker in speakers
            if speaker in features.genders
            and any(
                other != speaker
                and features.genders.get(other) == features.genders[speaker]
                for other in speakers
            )
        ]
        if len(eligible) < 2:
            return np.empty((0, 3), dtype=np.int64)
    pairs = []
    for _ in range(config.validation_pairs):
        speaker = str(generator.choice(eligible))
        negatives = [
            other
            for other in eligible
            if other != speaker
            and (
                not same_gender or features.genders[other] == features.genders[speaker]
            )
        ]
        negative = str(generator.choice(negatives))
        face_index = int(generator.choice(face_groups[speaker]))
        for target, label in ((speaker, 1), (negative, 0)):
            voice_index = int(generator.choice(voice_groups[target]))
            pairs.append((face_index, voice_index, label))
    return np.asarray(pairs, dtype=np.int64)
