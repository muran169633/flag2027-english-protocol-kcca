"""自包含的双分支匹配头，兼容既有冻结权重并用于内层训练。"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Projection(nn.Module):
    def __init__(self, config: dict[str, Any], classes: int, seed: int) -> None:
        super().__init__()
        self.config = config
        dimension = config["fixed_embedding_dimension"]

        def branch(input_dimension: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dimension, dimension),
                nn.BatchNorm1d(
                    dimension,
                    eps=config["batchnorm_epsilon"],
                    momentum=config["batchnorm_momentum"],
                ),
                nn.ReLU()
                if config["projection_activation"] == "relu"
                else nn.Identity(),
                nn.Dropout(config["dropout"]),
            )

        self.face = branch(config["face_dimension"])
        self.voice = branch(config["voice_dimension"])
        self.fusion_weights = nn.Parameter(
            torch.full((2,), config["fusion_initial_weight"])
        )
        self.classifier = nn.Linear(dimension, classes)
        generator = torch.Generator().manual_seed(seed)
        for component in (self.face, self.voice, self.classifier):
            for module in component.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(
                        module.weight,
                        gain=config["initialization_gain"],
                        generator=generator,
                    )
                    nn.init.constant_(module.bias, config["initialization_bias"])

    def encode(self, faces: Tensor, voices: Tensor) -> tuple[Tensor, Tensor]:
        return (
            F.normalize(
                self.face(faces), dim=-1, eps=self.config["normalization_epsilon"]
            ),
            F.normalize(
                self.voice(voices), dim=-1, eps=self.config["normalization_epsilon"]
            ),
        )

    def forward(self, faces: Tensor, voices: Tensor) -> Tensor:
        face, voice = self.encode(faces, voices)
        return (face * voice).sum(-1)  # [B, D] -> [B]，只比较当前配对。

    def loss_terms(
        self, faces: Tensor, voices: Tensor, labels: Tensor
    ) -> dict[str, Tensor]:
        face, voice = self.encode(faces, voices)
        fused = self.fusion_weights[0] * face + self.fusion_weights[1] * voice
        ce_face = F.cross_entropy(
            self.classifier(face) * self.config["classification_scale"], labels
        )
        ce_voice = F.cross_entropy(
            self.classifier(voice) * self.config["classification_scale"], labels
        )
        ce = (ce_face + ce_voice) / 2
        normalized = F.normalize(
            fused, dim=-1, eps=self.config["normalization_epsilon"]
        )
        scores = normalized @ normalized.T  # [B, B]，仅在训练批中计算损失。
        same = labels[:, None].eq(labels[None, :])
        positive = same & ~torch.eye(
            len(labels), dtype=torch.bool, device=labels.device
        )
        negative = ~same
        opl = (
            1
            - scores.masked_fill(~positive, 0).sum() / positive.sum().clamp_min(1)
            + self.config["orthogonality_negative_weight"]
            * scores.masked_fill(~negative, 0).abs().sum()
            / negative.sum().clamp_min(1)
        )
        logits = (face @ voice.T) / self.config["multi_positive_temperature"]
        targets = same.to(logits.dtype) / same.sum(1, keepdim=True)
        multi = (
            -(
                (targets * F.log_softmax(logits, dim=1)).sum(1).mean()
                + (targets.T * F.log_softmax(logits.T, dim=1)).sum(1).mean()
            )
            / 2
        )
        weighted_ce = ce * self.config["classification_weight"]
        weighted_opl = opl * self.config["orthogonality_weight"]
        weighted_multi = multi * self.config["multi_positive_weight"]
        return {
            "loss_ce": ce,
            "loss_ce_face": ce_face,
            "loss_ce_voice": ce_voice,
            "loss_opl": opl,
            "loss_multi": multi,
            "loss_weighted_ce": weighted_ce,
            "loss_weighted_opl": weighted_opl,
            "loss_weighted_multi": weighted_multi,
            "loss_total": weighted_ce + weighted_opl + weighted_multi,
        }
