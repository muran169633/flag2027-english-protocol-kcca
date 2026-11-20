"""逐项核对同序全局与局部缓存，只读复用，不复制大型特征。"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import Config

LOGGER = logging.getLogger("data")


@dataclass
class Features:
    name: str
    faces: NDArray[np.float16]
    voices: NDArray[np.float16]
    voice_masks: NDArray[np.bool_]
    face_globals: NDArray[np.float32]
    voice_globals: NDArray[np.float32]
    face_speakers: list[str]
    voice_speakers: list[str]
    genders: dict[str, str]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def split_speakers(features: Features, config: Config) -> tuple[list[str], list[str]]:
    speakers = sorted(set(features.face_speakers) & set(features.voice_speakers))
    namespaces: dict[str, list[str]] = defaultdict(list)
    for speaker in speakers:
        namespaces[speaker.split("/")[0]].append(speaker)
    rng = np.random.default_rng(config.seed)
    train, valid = [], []
    for values in namespaces.values():
        order = rng.permutation(values).tolist()
        count = max(2, round(len(values) * config.validation_fraction))
        if len(values) < count + 2:
            raise ValueError("每个来源至少需要四个双模态身份。")
        valid.extend(order[:count])
        train.extend(order[count:])
    assert not set(train) & set(valid)
    return sorted(train), sorted(valid)


def subset(features: Features, config: Config) -> Features:
    keep = set(
        sorted(set(features.face_speakers) & set(features.voice_speakers))[
            : config.smoke_speakers
        ]
    )

    def select(labels: list[str]) -> list[int]:
        counts: dict[str, int] = defaultdict(int)
        indices = []
        for index, label in enumerate(labels):
            if label in keep and counts[label] < config.smoke_samples_per_speaker:
                indices.append(index)
                counts[label] += 1
        return indices

    face, voice = select(features.face_speakers), select(features.voice_speakers)
    return Features(
        features.name,
        features.faces[face].copy(),
        features.voices[voice].copy(),
        features.voice_masks[voice].copy(),
        features.face_globals[face].copy(),
        features.voice_globals[voice].copy(),
        [features.face_speakers[i] for i in face],
        [features.voice_speakers[i] for i in voice],
        {key: value for key, value in features.genders.items() if key in keep},
    )


def verify_manifest(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    fingerprint = result.pop("fingerprint")
    if (
        hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        != fingerprint
    ):
        raise ValueError(f"特征清单指纹不符：{path}")
    return result


def load_cached_features(config: Config, run_dir: Path) -> Features:
    global_root = config.project_root / config.feature_source_run
    token_root = config.project_root / config.token_source_run
    baseline_root = config.project_root / config.comparison_run
    baseline_config = json.loads((baseline_root / "config.json").read_text())
    unchanged = [
        "seed",
        "cpu_threads",
        "allow_tf32",
        "face_global_dimension",
        "voice_global_dimension",
        "global_mix_initial",
        "layernorm_epsilon",
        "face_grid_size",
        "face_dimension",
        "voice_dimension",
        "voice_tokens",
        "embedding_dimension",
        "attention_dimension",
        "position_base",
        "dropout",
        "attention_dropout",
        "batchnorm_momentum",
        "batchnorm_epsilon",
        "normalization_epsilon",
        "linear_init_gain",
        "linear_init_bias",
        "validation_fraction",
        "evaluation_batch_size",
    ]
    if config.mode == "full":
        unchanged += ["validation_pairs"]
    for field in unchanged:
        if baseline_config[field] != getattr(config, field):
            raise ValueError(f"与父实验受控参数不同：{field}")
    global_config = json.loads((global_root / "config.json").read_text())
    with (config.project_root / global_config["v4_gender"]).open(
        newline="", encoding="utf-8"
    ) as stream:
        genders = {
            f"target/{speaker}": gender for speaker, gender in csv.reader(stream)
        }
    provenance: dict[str, Any] = {
        "global_source_run": config.feature_source_run,
        "token_source_run": config.token_source_run,
        "comparison_run": config.comparison_run,
        "gender_metadata_sha256": file_hash(
            config.project_root / global_config["v4_gender"]
        ),
        "gender_codes": config.gender_codes,
        "unchanged_parameters": unchanged,
        "effective_config": asdict(config),
        "datasets": {},
    }
    results = []
    for name, stage in (("target_english", "finetune"),):
        global_dir, token_dir = (
            global_root / "features" / name,
            token_root / "tokens" / name,
        )
        global_path, token_path = (
            global_dir / "manifest.json",
            token_dir / "manifest.json",
        )
        original, local = verify_manifest(global_path), verify_manifest(token_path)
        if local["source_manifest_sha256"] != file_hash(global_path) or local[
            "source_manifest"
        ] != str(global_path.relative_to(config.project_root)):
            raise ValueError("全局与局部缓存不是来自同一份有序媒体清单。")
        for field in (
            "face_weights_sha256",
            "voice_weights_sha256",
            "face_size",
            "face_bgr_mean",
            "sample_rate",
            "audio_seconds",
            "allow_tf32",
        ):
            if original[field] != local[field]:
                raise ValueError(f"全局与局部编码条件不同：{field}")
        faces = np.load(token_dir / "faces.npy", mmap_mode="r", allow_pickle=False)
        voices = np.load(token_dir / "voices.npy", mmap_mode="r", allow_pickle=False)
        mask = np.load(token_dir / "voice_masks.npy", mmap_mode="r", allow_pickle=False)
        face_globals = np.load(
            global_dir / "faces.npy", mmap_mode="r", allow_pickle=False
        )
        voice_globals = np.load(
            global_dir / "voices.npy", mmap_mode="r", allow_pickle=False
        )
        face_labels = [item["speaker"] for item in original["records"]["faces"]]
        voice_labels = [item["speaker"] for item in original["records"]["voices"]]
        expected = (
            (
                faces,
                (len(face_labels), config.face_grid_size**2, config.face_dimension),
                np.float16,
            ),
            (
                voices,
                (len(voice_labels), config.voice_tokens, config.voice_dimension),
                np.float16,
            ),
            (mask, (len(voice_labels), config.voice_tokens), np.bool_),
            (
                face_globals,
                (len(face_labels), config.face_global_dimension),
                np.float32,
            ),
            (
                voice_globals,
                (len(voice_labels), config.voice_global_dimension),
                np.float32,
            ),
        )
        for array, shape, dtype in expected:
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"缓存形状或类型不符：{name}")
        for directory in (global_dir, token_dir):
            for kind, labels in (("faces", face_labels), ("voices", voice_labels)):
                done = np.load(directory / f"{kind}_done.npy", allow_pickle=False)
                if done.shape != (len(labels),) or not bool(done.all()):
                    raise ValueError("特征来源缓存未完成。")
        if not bool(mask.any(1).all()):
            raise ValueError("存在没有有效语音 token 的样本。")
        features = Features(
            name,
            faces,
            voices,
            mask,
            face_globals,
            voice_globals,
            face_labels,
            voice_labels,
            genders if stage == "finetune" else {},
        )
        train, valid = split_speakers(features, config)
        if any(
            features.genders.get(speaker) not in config.gender_codes
            for speaker in train + valid
        ):
            raise ValueError("目标训练/验证身份缺少合法性别元数据。")
        old_split = json.loads((baseline_root / stage / "split.json").read_text())
        if (
            train != old_split["train_speakers"]
            or valid != old_split["validation_speakers"]
        ):
            raise ValueError("身份划分与父实验不一致。")
        provenance["datasets"][name] = {
            "global_manifest_sha256": file_hash(global_path),
            "token_manifest_sha256": file_hash(token_path),
            "same_media_order": True,
            "same_parent_split": True,
            "shapes": [list(array.shape) for array, _, _ in expected],
        }
        results.append(subset(features, config) if config.mode == "smoke" else features)
        LOGGER.info(
            "缓存同序核对通过 dataset=%s faces=%d voices=%d train_ids=%d val_ids=%d",
            name,
            len(face_labels),
            len(voice_labels),
            len(train),
            len(valid),
        )
    write_json(run_dir / "feature_sources.json", provenance)
    return results[0]
