"""冻结的全局/局部融合模型，仅保留自包含的重建与编码实现。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import Config


def sinusoidal(length: int, dimension: int, base: float) -> Tensor:
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    frequency = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32)
        * (-math.log(base) / dimension)
    )
    angles = positions * frequency[None, :]
    encoding = torch.zeros(length, dimension)
    encoding[:, 0::2], encoding[:, 1::2] = angles.sin(), angles.cos()
    return encoding


class TokenBranch(nn.Module):
    def __init__(self, dimension: int, spatial: bool, config: Config) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(dimension, eps=config.layernorm_epsilon),
            nn.Linear(dimension, config.embedding_dimension),
        )
        if spatial:
            coordinate = sinusoidal(
                config.face_grid_size,
                config.embedding_dimension // 2,
                config.position_base,
            )
            rows = coordinate[:, None, :].expand(-1, config.face_grid_size, -1)
            cols = coordinate[None, :, :].expand(config.face_grid_size, -1, -1)
            position = torch.cat((rows, cols), dim=-1).reshape(
                config.face_grid_size**2, config.embedding_dimension
            )
        else:
            position = sinusoidal(
                config.voice_tokens, config.embedding_dimension, config.position_base
            )
        self.register_buffer("position", position)
        self.output = nn.Sequential(
            nn.LayerNorm(config.embedding_dimension, eps=config.layernorm_epsilon),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, values: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is None:
            mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        if not bool(mask.any(1).all()):
            raise ValueError("每条样本至少需要一个有效 token。")
        projected = self.projection(values) + self.position[None, :, :]  # [B, L, D]
        pooled = (projected * mask[:, :, None]).sum(1) / mask.sum(
            1, keepdim=True
        )  # [B, D]
        return self.output(pooled)


class GlobalLocalBranch(nn.Module):
    def __init__(
        self, dimension: int, global_dimension: int, spatial: bool, config: Config
    ) -> None:
        super().__init__()
        self.config = config
        self.local = TokenBranch(dimension, spatial, config)
        self.global_projection = nn.Sequential(
            nn.LayerNorm(global_dimension, eps=config.layernorm_epsilon),
            nn.Linear(global_dimension, config.embedding_dimension),
            nn.LayerNorm(config.embedding_dimension, eps=config.layernorm_epsilon),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.global_gate = nn.Linear(
            2 * config.embedding_dimension, config.embedding_dimension
        )
        nn.init.zeros_(self.global_gate.weight)
        nn.init.constant_(
            self.global_gate.bias,
            math.log(config.global_mix_initial / (1 - config.global_mix_initial)),
        )

    def forward(
        self, tokens: Tensor, global_values: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        local = F.normalize(
            self.local(tokens, mask), dim=-1, eps=self.config.normalization_epsilon
        )
        global_embedding = F.normalize(
            self.global_projection(global_values),
            dim=-1,
            eps=self.config.normalization_epsilon,
        )
        gate = torch.sigmoid(
            self.global_gate(torch.cat((local, global_embedding), dim=-1))
        )  # [B, D]
        return (1 - gate) * local + gate * global_embedding, gate


class MatchingModel(nn.Module):
    def __init__(self, config: Config, num_classes: int) -> None:
        super().__init__()
        self.config = config
        self.face = GlobalLocalBranch(
            config.face_dimension, config.face_global_dimension, True, config
        )
        self.voice = GlobalLocalBranch(
            config.voice_dimension, config.voice_global_dimension, False, config
        )
        self.attention = nn.Sequential(
            self.block(
                2 * config.embedding_dimension,
                config.attention_dimension,
                config.attention_dropout,
            ),
            nn.Linear(config.attention_dimension, config.embedding_dimension),
        )
        self.classifier = nn.Linear(config.embedding_dimension, num_classes)
        # 两个对照组的共有投影、门控和分类参数使用同一个独立随机序列。
        generator = torch.Generator().manual_seed(config.seed)
        for component in (
            self.face.local.projection,
            self.voice.local.projection,
            self.attention,
            self.classifier,
        ):
            for module in component.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(
                        module.weight, gain=config.linear_init_gain, generator=generator
                    )
                    nn.init.constant_(module.bias, config.linear_init_bias)
        # 共有层先使用父实验相同的随机序列，再初始化新增全局投影，保持局部初始条件。
        for component in (self.face.global_projection, self.voice.global_projection):
            for module in component.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(
                        module.weight, gain=config.linear_init_gain, generator=generator
                    )
                    nn.init.constant_(module.bias, config.linear_init_bias)

    def block(self, source: int, target: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(source, target),
            nn.BatchNorm1d(
                target,
                eps=self.config.batchnorm_epsilon,
                momentum=self.config.batchnorm_momentum,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def normalize(self, values: Tensor) -> Tensor:
        return F.normalize(values, dim=-1, eps=self.config.normalization_epsilon)

    def encode_face(self, values: Tensor, global_values: Tensor) -> Tensor:
        embedding, _ = self.face(values, global_values)
        return self.normalize(torch.tanh(self.normalize(embedding)))

    def encode_voice(
        self, values: Tensor, mask: Tensor, global_values: Tensor
    ) -> Tensor:
        embedding, _ = self.voice(values, global_values, mask)
        return self.normalize(torch.tanh(self.normalize(embedding)))

    def forward_details(
        self,
        face: Tensor,
        voice: Tensor,
        voice_mask: Tensor,
        face_global: Tensor,
        voice_global: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        face_mixed, face_gate = self.face(face, face_global)  # [B, D]
        voice_mixed, voice_gate = self.voice(voice, voice_global, voice_mask)  # [B, D]
        face_unit, voice_unit = self.normalize(face_mixed), self.normalize(voice_mixed)
        face_trans, voice_trans = torch.tanh(face_unit), torch.tanh(voice_unit)
        gate = torch.sigmoid(self.attention(torch.cat((face_unit, voice_unit), dim=1)))
        fused = gate * face_trans + (1 - gate) * voice_trans
        return (
            fused,
            self.classifier(fused),
            self.normalize(face_trans),
            self.normalize(voice_trans),
            face_gate,
            voice_gate,
        )

    def forward(
        self,
        face: Tensor,
        voice: Tensor,
        voice_mask: Tensor,
        face_global: Tensor,
        voice_global: Tensor,
    ) -> tuple[Tensor, Tensor]:
        fused, logits, _, _, _, _ = self.forward_details(
            face, voice, voice_mask, face_global, voice_global
        )
        return fused, logits
