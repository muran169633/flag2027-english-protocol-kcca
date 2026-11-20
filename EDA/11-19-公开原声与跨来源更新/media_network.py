"""保留已发布双塔的模块顺序与前向，使未调权的适配可以逐张量核实。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class Tower(nn.Module):
    """保留原投影，额外返回未施加Dropout的同一次激活。"""

    def __init__(self, dimension: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, cfg["model_dim"]),
            nn.BatchNorm1d(cfg["model_dim"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
        )
        self.skip = nn.Linear(dimension, 1, bias=False)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.net[0](x)
        if residual is not None:
            projected = projected + residual
        clean = self.net[2](self.net[1](projected))
        return F.normalize(self.net[3](clean), dim=-1), F.normalize(clean, dim=-1)


class Matcher(nn.Module):
    """参数注册顺序与原头相同；参数从已保存初态精确加载。"""

    def __init__(self, df: int, dv: int, classes: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.face, self.voice = Tower(df, cfg), Tower(dv, cfg)
        self.face.skip = nn.Linear(cfg["face_base_dim"], 1, bias=False)
        self.proto = nn.Parameter(torch.zeros(classes, cfg["model_dim"]))


class Objective(nn.Module):
    """原正负平衡 BCE，可学习尺度与中心。"""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.center = nn.Parameter(torch.zeros(()))
        self.bounds = cfg["scale_bounds"]

    def forward(
        self, f: torch.Tensor, v: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        logits = self.log_scale.exp().clamp(*self.bounds) * (f @ v.T - self.center)
        positive = labels[:, None] == labels[None, :]
        return 0.5 * (
            F.softplus(-logits[positive]).mean() + F.softplus(logits[~positive]).mean()
        )
