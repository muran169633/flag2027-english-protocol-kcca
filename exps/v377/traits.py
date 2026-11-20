"""公开冻结的年龄模型及与单条媒体有关的确定性预处理。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
from numpy.typing import NDArray
from torch import Tensor, nn
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

ROOT = Path(__file__).resolve().parents[2]
Array = NDArray[Any]


class ModelHead(nn.Module):
    def __init__(self, config: Any, labels: int) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, labels)

    def forward(self, features: Tensor) -> Tensor:
        return self.out_proj(
            self.dropout(torch.tanh(self.dense(self.dropout(features))))
        )


class AgeGender(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.post_init()

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.wav2vec2(values).last_hidden_state.mean(1)  # [B, T, D] -> [B, D]
        return hidden, self.age(hidden), self.gender(hidden).softmax(-1)


class FrozenTraits:
    def __init__(self, config: dict[str, Any]) -> None:
        from insightface.model_zoo.attribute import Attribute

        self.config = config
        for path, expected in (
            (ROOT / config["face_resource"]["path"], config["face_resource"]["sha256"]),
            (
                ROOT / config["resource"]["directory"] / config["resource"]["weights"],
                config["resource"]["weights_sha256"],
            ),
        ):
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError("冻结属性模型指纹不符")
        source = ROOT / config["resource"]["directory"]
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            source, local_files_only=True
        )
        loaded, loading = AgeGender.from_pretrained(
            source,
            local_files_only=True,
            use_safetensors=True,
            output_loading_info=True,
        )
        if (
            loading["missing_keys"]
            or loading["unexpected_keys"]
            or loading["mismatched_keys"]
        ):
            raise ValueError(f"公开语音属性模型未严格加载：{loading}")
        self.audio = loaded.to(config["device"]).requires_grad_(False).eval()
        options = ort.SessionOptions()
        options.intra_op_num_threads = config["cpu_threads"]
        options.inter_op_num_threads = config["face_resource"]["inter_threads"]
        options.log_severity_level = config["face_resource"]["log_severity"]
        ort.disable_telemetry_events()
        path = ROOT / config["face_resource"]["path"]
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=config["face_resource"]["providers"],
        )
        self.face: Any = Attribute(str(path), session=session)
        cv2.setNumThreads(config["cpu_threads"])

    def image(self, path: Path, alignment: dict[str, Any]) -> Array:
        from insightface.utils.face_align import transform

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取训练图像：{path}")
        size = self.face.input_size
        box = alignment["bbox"]
        if box is None:
            crop = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        else:
            center = ((box[2] + box[0]) / 2, (box[3] + box[1]) / 2)
            scale = size[0] / (
                max(box[2] - box[0], box[3] - box[1])
                * self.config["face_resource"]["bounding_box_margin"]
            )
            crop, _ = transform(image, center, size[0], scale, 0)
        blob = cv2.dnn.blobFromImage(
            crop,
            1 / self.face.input_std,
            size,
            (self.face.input_mean,) * 3,
            swapRB=True,
        )
        result = self.face.session.run(
            self.face.output_names, {self.face.input_name: blob}
        )[0][0]
        if result.shape != (3,) or not np.isfinite(result).all():
            raise ValueError("冻结人脸年龄模型输出无效")
        return result.astype(np.float32)

    @torch.inference_mode()
    def voice(self, path: Path) -> tuple[Array, Array]:
        data = self.config["data"]
        values, rate = sf.read(path, dtype="float32", always_2d=True)
        wave = torch.from_numpy(values.mean(1))
        if rate != data["sample_rate"]:
            wave = torchaudio.functional.resample(wave, rate, data["sample_rate"])
        count = int(data["max_audio_seconds"] * data["sample_rate"])
        if len(wave) > count:
            start = (len(wave) - count) // 2
            wave = wave[start : start + count]
        if len(wave) < data["minimum_audio_samples"]:
            wave = torch.nn.functional.pad(
                wave, (0, data["minimum_audio_samples"] - len(wave))
            )
        current = self.processor(
            wave.numpy(), sampling_rate=data["sample_rate"], return_tensors="pt"
        )["input_values"].to(self.config["device"])
        with torch.autocast(device_type="cuda", dtype=getattr(torch, data["autocast"])):
            hidden, age, gender = self.audio(current)
        prediction = torch.cat([age, gender], dim=1)[0].float().cpu().numpy()
        embedding = hidden[0].float().cpu().numpy()
        if (
            prediction.shape != (4,)
            or not np.isfinite(prediction).all()
            or not np.isfinite(embedding).all()
        ):
            raise ValueError("冻结语音年龄模型输出无效")
        return prediction, embedding
