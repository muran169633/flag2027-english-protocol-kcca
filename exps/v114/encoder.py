"""固定SCRFD对齐，AdaFace使用BGR输入与公开预训练骨干。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch

from .common import Array, digest
from .inspection import weights


class Encoder:
    def __init__(self, cfg: dict[str, Any]) -> None:
        from insightface.model_zoo.scrfd import SCRFD

        self.cfg = cfg
        root = Path(cfg["face_weights_root"])
        verified = json.loads((root / "verified.json").read_text())
        if (
            digest(root / cfg["detection_file"])
            != verified["files"][cfg["detection_file"]]["sha256"]
        ):
            raise AssertionError("冻结人脸检测器指纹改变")
        cv2.setNumThreads(cfg["cpu_threads"])
        torch.set_num_threads(cfg["cpu_threads"])
        torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(cfg["gpu_memory_fraction"])
        torch.backends.cuda.matmul.allow_tf32 = cfg["allow_tf32"]
        torch.backends.cudnn.allow_tf32 = cfg["allow_tf32"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = cfg["cpu_threads"]
        options.inter_op_num_threads = cfg["inter_threads"]
        options.log_severity_level = cfg["onnx_log_severity"]
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.detector: Any = SCRFD(
            str(root / cfg["detection_file"]),
            session=ort.InferenceSession(
                str(root / cfg["detection_file"]),
                sess_options=options,
                providers=cfg["onnx_providers"],
            ),
        )
        self.detector.prepare(
            0,
            input_size=tuple(cfg["detection_size"]),
            det_thresh=cfg["detection_threshold"],
        )
        self.detector.nms_thresh = cfg["nms_threshold"]
        self.model, self.metadata = weights(cfg)
        self.model = self.model.to(cfg["device"])
        self.reference: Any = None
        self.checks = 0

    def reference_check(
        self, tensor: torch.Tensor, value: Array, norm: Array
    ) -> dict[str, float]:
        if self.reference is None:
            path = Path(self.cfg["upstream_root"]) / "net.py"
            spec = importlib.util.spec_from_file_location(
                "fixed_adaface_reference", path
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("作者参考实现不能加载")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.reference = (
                module.build_model(self.cfg["reference_model_name"])
                .eval()
                .requires_grad_(False)
            )
            self.reference.load_state_dict(self.model.state_dict(), strict=True)
        with torch.inference_mode():
            expected, magnitude = self.reference(tensor.cpu())
        feature_error = float(np.max(np.abs(value - expected.numpy())))
        norm_error = float(np.max(np.abs(norm / magnitude.numpy() - 1)))
        if max(feature_error, norm_error) > self.cfg["reference_tolerance"]:
            raise AssertionError("实际对齐人脸与作者前向不符")
        return {
            "feature_max_absolute_error": feature_error,
            "norm_max_relative_error": norm_error,
        }

    def encode(
        self, records: list[dict[str, Any]]
    ) -> tuple[Array, Array, list[str], list[dict[str, float]]]:
        from insightface.utils.face_align import norm_crop

        crops, methods = [], []
        for record in records:
            image = cv2.imread(record["face_path"], cv2.IMREAD_COLOR)
            if image is None or image.ndim != 3 or min(image.shape[:2]) < 2:
                raise ValueError("图像无法读取")
            boxes, points = self.detector.detect(image, max_num=0)
            if len(boxes):
                if points is None or not np.isfinite(points).all():
                    raise ValueError("人脸五点无效")
                height, width = image.shape[:2]
                center = (boxes[:, :2] + boxes[:, 2:4]) / 2
                distance = ((center - np.array([width / 2, height / 2])) ** 2).sum(
                    axis=1
                )
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                selected = int(np.argmax(area - self.cfg["center_penalty"] * distance))
                crop = norm_crop(
                    image, points[selected], image_size=self.cfg["recognition_size"]
                )
                methods.append("five_point")
            else:
                crop = cv2.resize(
                    image,
                    (self.cfg["recognition_size"], self.cfg["recognition_size"]),
                    interpolation=cv2.INTER_LINEAR,
                )
                methods.append("resize_fallback")
            crops.append(crop)
        # cv2原生BGR，作者模型要求保持此通道顺序。[B,H,W,3] -> [B,3,H,W]
        tensor = torch.from_numpy(np.stack(crops).transpose(0, 3, 1, 2).copy()).float()
        tensor = (
            tensor / self.cfg["pixel_scale"] - self.cfg["normalization_mean"]
        ) / self.cfg["normalization_std"]
        with torch.inference_mode():
            value, norm = self.model(tensor.to(self.cfg["device"]))
        features, magnitude = value.cpu().numpy(), norm.cpu().numpy()
        if (
            features.shape != (len(records), self.cfg["face_dimension"])
            or not np.isfinite(features).all()
            or not np.isfinite(magnitude).all()
        ):
            raise ValueError("AdaFace输出无效")
        count = min(self.cfg["actual_reference_checks"] - self.checks, len(records))
        diagnostics = []
        if count > 0:
            diagnostics.append(
                self.reference_check(
                    tensor[:count], features[:count], magnitude[:count]
                )
            )
            self.checks += count
        return features, magnitude, methods, diagnostics
