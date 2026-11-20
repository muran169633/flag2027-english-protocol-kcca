"""冻结 ERes2NetV2 的开发身份隔离对照，以及预先锁定的新来源声纹验证。"""

from __future__ import annotations

import gc
import importlib
import logging
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    roc_curve,
)
from torch import Tensor
from torchaudio.compliance import kaldi

from .common import Array, digest, normalize, save
from .ecapa import ECAPA
from .model import ERes2NetV2

logger = logging.getLogger(__name__)


def load_wave(path: Path, cfg: dict[str, Any]) -> Tensor:
    values, rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(values.mean(axis=1).copy())
    if rate != cfg["sample_rate"]:
        waveform = audio_functional.resample(waveform, rate, cfg["sample_rate"])
    limit = int(cfg["max_audio_seconds"] * cfg["sample_rate"])
    if len(waveform) > limit:
        start = (len(waveform) - limit) // 2
        waveform = waveform[start : start + limit]
    if (
        len(waveform) < cfg["minimum_audio_samples"]
        or not torch.isfinite(waveform).all()
    ):
        raise ValueError("音频过短或无效")
    return waveform


def reference_check(
    network: ERes2NetV2, state: dict[str, Tensor], cfg: dict[str, Any]
) -> dict[str, Any]:
    sys.path.insert(0, str(Path(cfg["reference_root"]).resolve()))
    try:
        module = importlib.import_module("speakerlab.models.eres2net.ERes2NetV2")
        reference = module.ERes2NetV2(
            feat_dim=cfg["mel_bins"],
            embedding_size=cfg["embedding_size"],
            m_channels=cfg["channels"],
            baseWidth=cfg["base_width"],
            scale=cfg["scale"],
            expansion=cfg["expansion"],
        )
        reference.load_state_dict(state, strict=True)
        reference.eval()
        torch.manual_seed(cfg["seed"])
        values = torch.randn(1, cfg["reference_frames"], cfg["mel_bins"])
        with torch.inference_mode():
            first = network(values.clone())
            second = reference(values.clone())
        difference = float(torch.max(torch.abs(first - second)))
        if difference > cfg["reference_tolerance"]:
            raise AssertionError("本地带类型模型未复现官方前向")
        return {
            "max_absolute_difference": difference,
            "parameters": sum(p.numel() for p in network.parameters()),
            "source_commit": cfg["source_commit"],
        }
    finally:
        sys.path.pop(0)


def extract(
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    output: Path,
    mode: str,
) -> tuple[Array, Array]:
    start = time.monotonic()
    weight = Path(cfg["weights_root"]) / cfg["weights_file"]
    if digest(weight) != cfg["weights_sha256"]:
        raise ValueError("ERes2NetV2 权重 SHA256 不符")
    state = torch.load(weight, map_location="cpu", weights_only=True)
    network = ERes2NetV2(cfg).eval()
    network.load_state_dict(state, strict=True)
    network.requires_grad_(False)
    save(output / "reference_check.json", reference_check(network, state, cfg))
    del state
    gc.collect()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    torch.cuda.set_per_process_memory_fraction(
        cfg["gpu_memory_fraction"], device=cfg["device"]
    )
    torch.backends.cuda.matmul.allow_tf32 = cfg["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = cfg["allow_tf32"]
    network = network.to(cfg["device"])
    ecapa = ECAPA(cfg).to(cfg["device"])
    # 第三方初始化可能改写 TF32；两种骨干推理前按主配置重新固定。
    torch.backends.cuda.matmul.allow_tf32 = cfg["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = cfg["allow_tf32"]
    values = []
    baselines = []
    for i, row in enumerate(records):
        if digest(Path(row["voice_path"])) != row["voice_sha256"]:
            raise ValueError("音频哈希改变")
        waveform = load_wave(Path(row["voice_path"]), cfg)
        fbank = kaldi.fbank(
            waveform.unsqueeze(0),
            num_mel_bins=cfg["mel_bins"],
            sample_frequency=cfg["sample_rate"],
            dither=cfg["fbank_dither"],
        )
        fbank -= fbank.mean(dim=0, keepdim=True)
        with torch.inference_mode():
            embedding = network(fbank.unsqueeze(0).to(cfg["device"])).cpu().numpy()[0]
            baselines.append(ecapa(waveform.to(cfg["device"])).cpu().numpy()[0])
        values.append(embedding)
        if (i + 1) % cfg["feature_log_interval"] == 0 or i + 1 == len(records):
            logger.info("声纹进度 %s/%s", i + 1, len(records))
        if time.monotonic() - start > cfg["time_limit_seconds"]:
            raise TimeoutError("提取超过预算")
    eres = normalize(np.stack(values))
    baseline = normalize(np.stack(baselines))
    if (
        eres.shape != (len(records), cfg["embedding_size"])
        or not np.isfinite(eres).all()
    ):
        raise ValueError("ERes 特征形状或数值无效")
    np.savez_compressed(output / "features.npz", eres=eres, ecapa=baseline)
    save(
        output / "extraction.json",
        {
            "seconds": time.monotonic() - start,
            "rows": len(records),
            "mode": mode,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
    )
    return baseline, eres


def measured(truth: Array, values: Array) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(cast(Any, truth), cast(Any, values))
    error = float(brentq(lambda x: 1 - x - interp1d(fpr, tpr)(x), 0, 1))
    threshold = float(thresholds[int(np.argmin(np.abs(fpr - (1 - tpr))))])
    return error, threshold
