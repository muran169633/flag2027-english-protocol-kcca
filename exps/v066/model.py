"""ERes2NetV2 的带类型实现，与固定版本官方参考逐值核对。

改写自 3D-Speaker，保留参数名称及网络运算顺序。
Copyright 3D-Speaker. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AFF(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = channels // reduction
        self.local_att = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        attention = 1.0 + torch.tanh(self.local_att(torch.cat((first, second), dim=1)))
        return first * attention + second * (2.0 - attention)


class Block(nn.Module):
    def __init__(
        self, in_planes: int, planes: int, stride: int, fuse: bool, cfg: dict[str, Any]
    ) -> None:
        super().__init__()
        self.width = int(planes * (cfg["base_width"] / 64.0))
        self.scale = cfg["scale"]
        self.fuse = fuse
        width = self.width
        scale = self.scale
        expansion = cfg["expansion"]
        self.conv1 = nn.Conv2d(in_planes, width * scale, 1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.convs = nn.ModuleList(
            [nn.Conv2d(width, width, 3, padding=1, bias=False) for _ in range(scale)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(scale)])
        if fuse:
            self.fuse_models = nn.ModuleList(
                [AFF(width, cfg["fusion_reduction"]) for _ in range(scale - 1)]
            )
        self.relu = nn.Hardtanh(0, cfg["relu_max"], inplace=True)
        self.conv3 = nn.Conv2d(width * scale, planes * expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )

    def forward(self, values: Tensor) -> Tensor:
        chunks = torch.split(self.relu(self.bn1(self.conv1(values))), self.width, dim=1)
        outputs = []
        current = chunks[0]
        for index in range(self.scale):
            if index == 0:
                current = chunks[index]
            elif self.fuse:
                current = self.fuse_models[index - 1](current, chunks[index])
            else:
                current = current + chunks[index]
            current = self.relu(self.bns[index](self.convs[index](current)))
            outputs.append(current)
        out = self.bn3(self.conv3(torch.cat(outputs, dim=1)))
        return self.relu(out + self.shortcut(values))


class StatisticsPooling(nn.Module):
    def __init__(self, epsilon: float) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, values: Tensor) -> Tensor:
        mean = values.mean(dim=-1).flatten(start_dim=1)
        std = torch.sqrt(torch.var(values, dim=-1) + self.epsilon).flatten(start_dim=1)
        return torch.cat((mean, std), dim=1)


class ERes2NetV2(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        channels = cfg["channels"]
        expansion = cfg["expansion"]
        self.in_planes = channels
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer1 = self.make_layer(channels, cfg["blocks"][0], 1, False, cfg)
        self.layer2 = self.make_layer(channels * 2, cfg["blocks"][1], 2, False, cfg)
        self.layer3 = self.make_layer(channels * 4, cfg["blocks"][2], 2, True, cfg)
        self.layer4 = self.make_layer(channels * 8, cfg["blocks"][3], 2, True, cfg)
        self.layer3_ds = nn.Conv2d(
            channels * 4 * expansion,
            channels * 8 * expansion,
            3,
            padding=1,
            stride=2,
            bias=False,
        )
        self.fuse34 = AFF(channels * 8 * expansion, cfg["fusion_reduction"])
        self.pool = StatisticsPooling(cfg["pooling_epsilon"])
        stats_dim = int(cfg["mel_bins"] / 8) * channels * 8
        self.seg_1 = nn.Linear(stats_dim * expansion * 2, cfg["embedding_size"])
        self.seg_bn_1 = nn.Identity()
        self.seg_2 = nn.Identity()

    def make_layer(
        self, planes: int, count: int, stride: int, fuse: bool, cfg: dict[str, Any]
    ) -> nn.Sequential:
        layers = []
        for current_stride in [stride] + [1] * (count - 1):
            layers.append(Block(self.in_planes, planes, current_stride, fuse, cfg))
            self.in_planes = planes * cfg["expansion"]
        return nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        values = values.permute(0, 2, 1).unsqueeze(1)  # [B, T, F] → [B, 1, F, T]
        first = self.layer1(F.relu(self.bn1(self.conv1(values))))
        second = self.layer2(first)
        third = self.layer3(second)
        fourth = self.layer4(third)
        return self.seg_1(
            self.pool(self.fuse34(fourth, self.layer3_ds(third)))
        )  # [B, 192]
