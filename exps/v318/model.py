"""WeSpeaker ResNet293结构的带类型实现；Apache-2.0来源见docs与refs。

原结构版权：2021 Shuai Wang，2022/2024 Zhengyang Chen，2023 Bing Han。
保留卷积与状态字典名称，以官方PyTorch/ONNX双格式核对前向。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    def __init__(
        self, incoming: int, channels: int, stride: int, expansion: int
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(
            channels, channels * expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(channels * expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or incoming != channels * expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    incoming,
                    channels * expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channels * expansion),
            )

    def forward(self, values: Tensor) -> Tensor:
        result = F.relu(self.bn1(self.conv1(values)))
        result = F.relu(self.bn2(self.conv2(result)))
        result = self.bn3(self.conv3(result))
        return F.relu(result + self.shortcut(values))


class SpeakerResNet(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        if cfg["pooling"] != "TSTP" or cfg["two_embedding_layers"]:
            raise ValueError("冻结权重仅支持单嵌入层TSTP")
        channels = cfg["model_channels"]
        self.in_planes = channels
        self.expansion = cfg["bottleneck_expansion"]
        self.variance_epsilon = cfg["pooling_variance_epsilon"]
        self.variance_correction = cfg["pooling_variance_correction"]
        self.conv1 = nn.Conv2d(
            1, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer1 = self.make_layer(
            channels, cfg["model_blocks"][0], cfg["stage_strides"][0]
        )
        self.layer2 = self.make_layer(
            channels * 2, cfg["model_blocks"][1], cfg["stage_strides"][1]
        )
        self.layer3 = self.make_layer(
            channels * 4, cfg["model_blocks"][2], cfg["stage_strides"][2]
        )
        self.layer4 = self.make_layer(
            channels * 8, cfg["model_blocks"][3], cfg["stage_strides"][3]
        )
        stats_dim = (cfg["mel_bins"] // 8) * channels * 8 * self.expansion * 2
        self.seg_1 = nn.Linear(stats_dim, cfg["embedding_size"])
        self.seg_bn_1 = nn.Identity()
        self.seg_2 = nn.Identity()

    def make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for step in [stride] + [1] * (blocks - 1):
            layers.append(Bottleneck(self.in_planes, channels, step, self.expansion))
            self.in_planes = channels * self.expansion
        return nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        # [B,T,F] -> [B,1,F,T]
        result = values.permute(0, 2, 1).unsqueeze(1)
        result = F.relu(self.bn1(self.conv1(result)))
        result = self.layer4(self.layer3(self.layer2(self.layer1(result))))
        # [B,C,F,T] -> [B,2*C*F]
        mean = result.mean(dim=-1).flatten(start_dim=1)
        std = torch.sqrt(
            torch.var(result, dim=-1, correction=self.variance_correction)
            + self.variance_epsilon
        ).flatten(start_dim=1)
        return self.seg_1(torch.cat([mean, std], dim=1))
