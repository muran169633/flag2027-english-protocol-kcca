"""作者AdaFace IR101推理骨干的带类型实现，保留上游参数名称。

来源：mk-minchul/AdaFace，c60eaa786a42c03444f3df7096dbaf9d57ae010d/net.py。
Copyright (c) 2022 Minchul Kim，MIT许可见LICENSE.AdaFace。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, incoming: int, depth: int, stride: int) -> None:
        super().__init__()
        self.shortcut_layer = (
            nn.MaxPool2d(1, stride)
            if incoming == depth
            else nn.Sequential(
                nn.Conv2d(incoming, depth, 1, stride, bias=False), nn.BatchNorm2d(depth)
            )
        )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(incoming),
            nn.Conv2d(incoming, depth, 3, 1, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, 3, stride, 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.res_layer(value) + self.shortcut_layer(value)


class FaceBackbone(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        first = cfg["model_channels"][0]
        self.input_layer = nn.Sequential(
            nn.Conv2d(cfg["input_channels"], first, 3, 1, 1, bias=False),
            nn.BatchNorm2d(first),
            nn.PReLU(first),
        )
        units, incoming = [], first
        for depth, count in zip(
            cfg["model_channels"], cfg["model_blocks"], strict=True
        ):
            for index in range(count):
                units.append(
                    Block(incoming, depth, cfg["stage_stride"] if index == 0 else 1)
                )
                incoming = depth
        self.body = nn.Sequential(*units)
        side = cfg["recognition_size"] // cfg["stage_stride"] ** len(
            cfg["model_blocks"]
        )
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(incoming),
            nn.Dropout(cfg["model_dropout"]),
            nn.Flatten(),
            nn.Linear(incoming * side * side, cfg["face_dimension"]),
            nn.BatchNorm1d(cfg["face_dimension"], affine=False),
        )

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [B, 3, 112, 112] -> [B, 512]，输出单位身份向量和归一化前模长。
        value = self.output_layer(self.body(self.input_layer(value)))
        norm = torch.linalg.vector_norm(value, dim=1, keepdim=True)
        return value / norm, norm
