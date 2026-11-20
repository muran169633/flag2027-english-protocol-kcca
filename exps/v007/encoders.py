"""冻结编码器同时提取全局向量和局部 token，保持训练时的预处理。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F

from .config import Config


class FaceEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        convolutions: list[nn.Conv2d] = []
        channels = 3
        # VGG-Face 固定架构：pool5 空间特征与 fc7 全局特征共享卷积计算。
        for block in (
            (64, 64),
            (128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (512, 512, 512),
        ):
            for output in block:
                layer = nn.Conv2d(channels, output, 3, padding=1)
                convolutions.append(layer)
                layers.extend([layer, nn.ReLU()])
                channels = output
            layers.append(nn.MaxPool2d(2, 2))
        self.spatial = nn.Sequential(*layers)
        global_layers: list[nn.Module] = []
        for output, kernel in (
            (config.face_global_dimension, 7),
            (config.face_global_dimension, 1),
        ):
            layer = nn.Conv2d(channels, output, kernel)
            convolutions.append(layer)
            global_layers.extend([layer, nn.ReLU()])
            channels = output
        self.global_head = nn.Sequential(*global_layers)
        with h5py.File(config.project_root / config.face_weights, "r") as handle:
            root: Any = handle.get("model_weights", handle)
            loaded = 0
            for raw_name in root.attrs["layer_names"]:
                if loaded == len(convolutions):
                    break
                name = (
                    raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
                )
                arrays: list[np.ndarray] = []

                def visit(
                    _name: str, value: Any, collected: list[np.ndarray] = arrays
                ) -> None:
                    if isinstance(value, h5py.Dataset):
                        collected.append(np.asarray(value))

                root[name].visititems(visit)
                kernels = [value for value in arrays if value.ndim == 4]
                biases = [value for value in arrays if value.ndim == 1]
                if not kernels or not biases:
                    continue
                layer = convolutions[loaded]
                kernel = torch.from_numpy(kernels[0].transpose(3, 2, 0, 1).copy())
                bias = torch.from_numpy(biases[0])
                if (
                    kernel.shape != layer.weight.shape
                    or layer.bias is None
                    or bias.shape != layer.bias.shape
                ):
                    raise ValueError("VGG-Face 权重形状与固定架构不符。")
                with torch.no_grad():
                    layer.weight.copy_(kernel)
                    layer.bias.copy_(bias)
                loaded += 1
            if loaded != len(convolutions):
                raise ValueError("VGG-Face 权重不完整。")
        self.requires_grad_(False).eval()

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        spatial = self.spatial(images)  # [B, 512, 7, 7]
        expected = (
            self.config.face_dimension,
            self.config.face_grid_size,
            self.config.face_grid_size,
        )
        if spatial.shape[1:] != expected:
            raise ValueError("VGG-Face 空间特征尺寸与配置不符。")
        tokens = spatial.flatten(2).transpose(1, 2)  # [B, 49, 512]
        global_values = self.global_head(spatial).flatten(1)  # [B, 4096]
        return tokens, global_values


class VoiceEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.processing.features import InputNormalization

        self.config = config
        self.features = Fbank(n_mels=config.audio_mels, sample_rate=config.sample_rate)
        self.normalization = InputNormalization(norm_type="sentence", std_norm=False)
        self.embedding = ECAPA_TDNN(
            input_size=config.audio_mels,
            channels=list(config.ecapa_channels),
            kernel_sizes=list(config.ecapa_kernels),
            dilations=list(config.ecapa_dilations),
            attention_channels=config.ecapa_attention_channels,
            lin_neurons=config.voice_global_dimension,
        )
        self.embedding.load_state_dict(
            torch.load(
                config.project_root / config.voice_weights,
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        # SpeechBrain 导入有全局副作用，恢复训练时的数值设置。
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32
        self.requires_grad_(False).eval()

    def forward(
        self, waveforms: Tensor, lengths: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        features = self.features(waveforms)  # [B, T, 80]
        normalized = self.normalization(features, lengths)
        values = normalized.transpose(1, 2)  # [B, 80, T]
        outputs: list[Tensor] = []
        for block in self.embedding.blocks:
            try:
                values = block(values, lengths=lengths)
            except TypeError:
                values = block(values)
            outputs.append(values)
        frames = self.embedding.mfa(torch.cat(outputs[1:], dim=1))  # [B, 1536, T]
        pooled = self.embedding.asp_bn(self.embedding.asp(frames, lengths=lengths))
        global_values = self.embedding.fc(pooled).transpose(1, 2).flatten(1)  # [B, 192]
        tokens = frames.new_zeros(
            (len(frames), self.config.voice_tokens, self.config.voice_dimension)
        )
        mask = torch.zeros(
            (len(frames), self.config.voice_tokens),
            dtype=torch.bool,
            device=frames.device,
        )
        valid_frames = (
            (lengths * frames.shape[-1]).round().long().clamp(1, frames.shape[-1])
        )
        for index in range(len(frames)):
            count = int(valid_frames[index].item())
            valid = frames[index, :, :count]
            if count >= self.config.voice_tokens:
                tokens[index] = F.adaptive_avg_pool1d(valid, self.config.voice_tokens).T
                mask[index] = True
            else:
                tokens[index, :count] = valid.T
                mask[index, :count] = True
        return tokens, mask, global_values


def load_face(path: Path, config: Config) -> Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (config.face_size, config.face_size), Image.Resampling.BICUBIC
        )
        value = np.asarray(image, dtype=np.float32)[..., ::-1].copy()
    value -= np.asarray(config.face_bgr_mean, dtype=np.float32)
    return torch.from_numpy(value.transpose(2, 0, 1).copy())  # [3, H, W]


def load_voice(path: Path, config: Config) -> Tensor:
    import librosa

    signal, rate = sf.read(path, dtype="float32", always_2d=True)
    signal = signal.mean(axis=1)
    if rate != config.sample_rate:
        signal = librosa.resample(signal, orig_sr=rate, target_sr=config.sample_rate)
    maximum = int(config.audio_seconds * config.sample_rate)
    if signal.size > maximum:
        start = (signal.size - maximum) // 2
        signal = signal[start : start + maximum]
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError(f"音频为空或含非有限值：{path}")
    return torch.from_numpy(np.ascontiguousarray(signal))  # [T]
