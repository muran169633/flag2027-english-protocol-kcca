"""Standalone copied ArcFace/SCRFD frozen image encoder from v055."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

Array = NDArray[Any]


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class FaceEncoder:
    def __init__(self, cfg: dict[str, Any]) -> None:
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX
        from insightface.model_zoo.scrfd import SCRFD

        self.cfg = cfg
        root = Path(cfg["face_weights_root"])
        verified = json.loads((root / "verified.json").read_text())
        for name in (cfg["recognition_file"], cfg["detection_file"]):
            if sha(root / name) != verified["files"][name]["sha256"]:
                raise ValueError("人脸模型哈希不符")
        if any(
            provider not in ort.get_available_providers()
            for provider in cfg["onnx_providers"]
        ):
            raise ValueError("ONNX 后端不可用")
        cv2.setNumThreads(cfg["cpu_threads"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = cfg["cpu_threads"]
        options.inter_op_num_threads = cfg["inter_threads"]
        options.log_severity_level = cfg["onnx_log_severity"]
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        det, rec = root / cfg["detection_file"], root / cfg["recognition_file"]
        self.detector: Any = SCRFD(
            str(det),
            session=ort.InferenceSession(
                str(det), sess_options=options, providers=cfg["onnx_providers"]
            ),
        )
        self.recognizer: Any = ArcFaceONNX(
            str(rec),
            session=ort.InferenceSession(
                str(rec), sess_options=options, providers=cfg["onnx_providers"]
            ),
        )
        self.detector.prepare(
            0,
            input_size=tuple(cfg["detection_size"]),
            det_thresh=cfg["detection_threshold"],
        )
        self.detector.nms_thresh = cfg["nms_threshold"]
        if (
            self.recognizer.input_size
            != (cfg["recognition_size"], cfg["recognition_size"])
            or self.recognizer.output_shape[-1] != cfg["face_dimension"]
        ):
            raise ValueError("人脸编码器维度不符")

    def encode(self, records: list[dict[str, Any]]) -> tuple[Array, list[str]]:
        from insightface.utils.face_align import norm_crop

        images, methods = [], []
        for record in records:
            image = cv2.imread(record["face_path"], cv2.IMREAD_COLOR)
            if image is None or image.ndim != 3 or min(image.shape[:2]) < 2:
                raise ValueError("人脸图像不可读")
            boxes, points = self.detector.detect(image, max_num=0)
            if len(boxes):
                if points is None or not np.isfinite(points).all():
                    raise ValueError("人脸关键点无效")
                height, width = image.shape[:2]
                centers = (boxes[:, :2] + boxes[:, 2:4]) / 2
                distance = ((centers - np.array([width / 2, height / 2])) ** 2).sum(
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
            images.append(crop)
        result = np.asarray(
            self.recognizer.get_feat(images), dtype=np.float32
        )  # [B, 512]
        if (
            result.shape != (len(records), self.cfg["face_dimension"])
            or not np.isfinite(result).all()
        ):
            raise ValueError("人脸特征无效")
        return result / np.maximum(
            np.linalg.norm(result, axis=1, keepdims=True), 1e-12
        ), methods


def normalize(values: Array) -> Array:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
