"""冻结的既有 ECAPA 基线，用于新来源录音。"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class ECAPA(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.processing.features import InputNormalization

        self.features = Fbank(n_mels=cfg["audio_mels"], sample_rate=cfg["sample_rate"])
        self.norm = InputNormalization(
            norm_type=cfg["normalization_type"], std_norm=cfg["normalization_std"]
        )
        self.model = ECAPA_TDNN(
            input_size=cfg["audio_mels"],
            channels=cfg["ecapa_channels"],
            kernel_sizes=cfg["ecapa_kernels"],
            dilations=cfg["ecapa_dilations"],
            attention_channels=cfg["ecapa_attention_channels"],
            lin_neurons=cfg["embedding_size"],
        )
        self.model.load_state_dict(
            torch.load(cfg["ecapa_weights"], map_location="cpu", weights_only=True),
            strict=True,
        )
        self.requires_grad_(False).eval()

    def forward(self, waveform: Tensor) -> Tensor:
        length = torch.ones(1, device=waveform.device)
        values = self.features(waveform.unsqueeze(0))  # [1, T, 80]
        return self.model(self.norm(values, length), length).flatten(1)  # [1, 192]
