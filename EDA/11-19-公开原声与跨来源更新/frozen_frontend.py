"""使用项目已有的冻结骨干，在相同权重和预处理下提取两路图像及六路声音特征。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from media_common import CFG as STUDY
from media_common import ROOT, read, sha
from PIL import Image
from transformers import (
    AutoImageProcessor,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    Wav2Vec2FeatureExtractor,
    WhisperFeatureExtractor,
    WhisperModel,
)

sys.path.insert(0, str(ROOT))
from exps.v066.audio import extract
from exps.v318 import extract as resnet
from exps.v372.submit import FrozenVoice
from exps.v377.traits import AgeGender
from exps.v383.extract_voice import AGE, SR, wave

CFG = read(ROOT / STUDY["featurizer_config"])


def normalized(values: np.ndarray) -> np.ndarray:
    return (
        values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    ).astype(np.float32)


def faces(paths: list[Path]) -> dict[str, np.ndarray]:
    result = {}
    for key, factory in (("clip", CLIPImageProcessor), ("clipL", AutoImageProcessor)):
        repo = CFG["face_models"][key]
        processor: Any = factory.from_pretrained(repo, local_files_only=True)
        model = (
            CLIPVisionModelWithProjection.from_pretrained(repo, local_files_only=True)
            .eval()
            .cuda()
        )
        parts = []
        for start in range(0, len(paths), CFG["clip_batch"]):
            images = []
            for path in paths[start : start + CFG["clip_batch"]]:
                with Image.open(path) as im:
                    images.append(im.convert("RGB"))
            x = processor(images=images, return_tensors="pt")["pixel_values"].cuda()
            parts.append(
                torch.nn.functional.normalize(model(pixel_values=x).image_embeds, dim=1)
                .cpu()
                .numpy()
            )
        result[key] = np.concatenate(parts)
        del model
    return result


def speakers(paths: list[Path], temp: Path) -> dict[str, np.ndarray]:
    c = json.loads((ROOT / "exps/v066/config.json").read_bytes())
    assert c["gpu_memory_fraction"] == CFG["gpu_fraction"]
    records = [{"voice_path": str(p), "voice_sha256": sha(p)} for p in paths]
    ecapa, eres = extract(records, c, temp, "cross_day")
    model = resnet.SpeakerResNet(resnet.CFG88).eval().requires_grad_(False)
    params = torch.load(
        Path(resnet.CFG88["weights_root"]) / resnet.CFG88["checkpoint_file"],
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(
        {k: v for k, v in params.items() if k in model.state_dict()}, strict=True
    )
    del params
    model = model.cuda()
    vs = []
    for path in paths:
        w = resnet.center(resnet.wave(str(path)), 6.0)
        x = model(resnet.fb(w).unsqueeze(0).cuda()).cpu().numpy()[0]
        vs.append(x / np.linalg.norm(x))
    del model
    cfg = json.loads((ROOT / "exps/v372/config.json").read_bytes())
    cfg["device"] = "cuda:0"
    wavlm = FrozenVoice(cfg)
    return {
        "ecapa": normalized(ecapa),
        "eres": normalized(eres),
        "resnet": normalized(np.stack(vs)),
        "wavlm": np.stack([wavlm.encode(p) for p in paths]),
    }


def generic(paths: list[Path]) -> dict[str, np.ndarray]:
    age = AgeGender.from_pretrained(str(AGE), local_files_only=True).cuda().eval()
    aproc: Any = Wav2Vec2FeatureExtractor.from_pretrained(
        str(AGE), local_files_only=True
    )
    wproc: Any = WhisperFeatureExtractor.from_pretrained(
        "openai/whisper-small", local_files_only=True
    )
    wenc = (
        WhisperModel.from_pretrained("openai/whisper-small", local_files_only=True)
        .encoder.cuda()
        .eval()
    )
    ages, voices = [], []
    for path in paths:
        w = wave(str(path))
        x = aproc(w, sampling_rate=SR, return_tensors="pt")["input_values"].cuda()
        ages.append(age(x)[0][0].float().cpu().numpy())
        f = wproc(w, sampling_rate=SR, return_tensors="pt")["input_features"].cuda()
        h = wenc(f).last_hidden_state[0]  # [T, 768]
        valid = max(1, min(len(h), int(np.ceil(len(w) / SR * 50))))
        voices.append(h[:valid].mean(0).float().cpu().numpy())
    return {"age6": normalized(np.stack(ages)), "whisper": normalized(np.stack(voices))}
