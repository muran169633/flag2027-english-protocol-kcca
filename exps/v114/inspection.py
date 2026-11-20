"""安全读取作者检查点，核对独立骨干与原作者数值前向。"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import digest, save
from .model import FaceBackbone


def weights(cfg: dict[str, Any]) -> tuple[FaceBackbone, dict[str, Any]]:
    path = Path(cfg["weight_root"]) / cfg["weight_filename"]
    if digest(path) != cfg["expected_weight_sha256"]:
        raise AssertionError("作者公开文件首次指纹改变")
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    if any(name != "argparse.Namespace" for name in unsafe):
        raise TypeError("检查点含未许可元数据类型：" + repr(unsafe))
    with torch.serialization.safe_globals([argparse.Namespace]):
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not isinstance(state.get("state_dict"), dict):
        raise TypeError("检查点不是作者所述state_dict结构")
    prefix = cfg["model_state_prefix"]
    backbone = {
        k[len(prefix) :]: v
        for k, v in state["state_dict"].items()
        if k.startswith(prefix)
    }
    if not backbone or not all(isinstance(v, torch.Tensor) for v in backbone.values()):
        raise TypeError("身份表示权重不完整")
    model = FaceBackbone(cfg).eval().requires_grad_(False)
    model.load_state_dict(backbone, strict=True)
    metadata = {
        "checkpoint_sha256": digest(path),
        "checkpoint_unsafe_metadata_types": unsafe,
        "loaded_backbone_keys": len(backbone),
        "parameters": sum(p.numel() for p in model.parameters()),
        "discarded_training_state_keys": [
            k for k in state["state_dict"] if not k.startswith(prefix)
        ],
    }
    return model, metadata


def run(cfg: dict[str, Any], output: Path) -> None:
    torch.set_num_threads(cfg["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(cfg["gpu_memory_fraction"])
    torch.backends.cuda.matmul.allow_tf32 = cfg["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = cfg["allow_tf32"]
    torch.manual_seed(cfg["seed"])
    model, metadata = weights(cfg)
    path = Path(cfg["upstream_root"]) / "net.py"
    spec = importlib.util.spec_from_file_location("pinned_adaface_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("作者参考实现不能加载")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = (
        module.build_model(cfg["reference_model_name"]).eval().requires_grad_(False)
    )
    reference.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(
        cfg["reference_batch_size"],
        cfg["input_channels"],
        cfg["recognition_size"],
        cfg["recognition_size"],
    )
    with torch.inference_mode():
        expected, expected_norm = reference(x)
        actual, actual_norm = model(x)
        cpu_error = float((actual - expected).abs().max())
        cpu_norm_error = float(
            ((actual_norm - expected_norm) / expected_norm).abs().max()
        )
        actual, actual_norm = model.cuda()(x.cuda())
        gpu_error = float(np.max(np.abs(actual.cpu().numpy() - expected.numpy())))
        gpu_norm_error = float(
            np.max(np.abs(actual_norm.cpu().numpy() / expected_norm.numpy() - 1))
        )
    save(
        output / "reference_diagnostics.json",
        {
            "cpu_feature_error": cpu_error,
            "cpu_norm_relative_error": cpu_norm_error,
            "gpu_feature_error": gpu_error,
            "gpu_norm_relative_error": gpu_norm_error,
        },
    )
    if (
        max(cpu_error, cpu_norm_error, gpu_error, gpu_norm_error)
        > cfg["reference_tolerance"]
    ):
        raise AssertionError("本地实现与作者随机输入前向不符")
    save(
        output / "result.json",
        {
            **metadata,
            "cpu_feature_error": cpu_error,
            "cpu_norm_relative_error": cpu_norm_error,
            "gpu_feature_error": gpu_error,
            "gpu_norm_relative_error": gpu_norm_error,
            "upstream_net_sha256": digest(path),
            "quality_metrics_computed": False,
        },
    )
