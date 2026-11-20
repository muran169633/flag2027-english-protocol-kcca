"""从作者公开链接准备AdaFace权重，记录来源、大小与首次内容指纹。"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests

from .common import digest, save

logger = logging.getLogger(__name__)


def run(cfg: dict[str, Any], output: Path) -> None:
    started = time.monotonic()
    root = Path(cfg["weight_root"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / cfg["weight_filename"]
    partial = path.with_suffix(path.suffix + ".partial")
    verified = root / "verified.json"
    if verified.exists():
        import json

        metadata = json.loads(verified.read_text())
        if not path.exists() or digest(path) != metadata["sha256"]:
            raise AssertionError("既有权重指纹不匹配")
        save(output / "result.json", {**metadata, "reused": True})
        return
    params = {"id": cfg["official_file_id"], "export": "download", "confirm": "t"}
    previous = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={previous}-"} if previous else {}
    with requests.get(
        cfg["download_base"],
        params=params,
        headers=headers,
        stream=True,
        timeout=(cfg["connect_timeout_seconds"], cfg["read_timeout_seconds"]),
    ) as response:
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            raise RuntimeError(
                "公开模型链接未返回模型文件，保留页面状态，不尝试账号登录"
            )
        if previous and response.status_code == 206:
            expected_prefix = f"bytes {previous}-"
            if not response.headers.get("Content-Range", "").startswith(
                expected_prefix
            ):
                raise AssertionError("断点续传偏移错误")
            mode = "ab"
        else:
            previous, mode = 0, "wb"
        expected = (
            previous + int(response.headers["Content-Length"])
            if "Content-Length" in response.headers
            else None
        )
        if expected is not None and expected > cfg["maximum_download_bytes"]:
            raise RuntimeError("文件超出授权大小上限")
        count, logged = previous, previous
        with partial.open(mode) as stream:
            for chunk in response.iter_content(cfg["chunk_bytes"]):
                if chunk:
                    stream.write(chunk)
                    count += len(chunk)
                if (
                    count > cfg["maximum_download_bytes"]
                    or time.monotonic() - started > cfg["maximum_download_seconds"]
                ):
                    raise TimeoutError("下载达到资源预算，保留断点")
                if count - logged >= cfg["download_log_bytes"]:
                    logger.info("AdaFace下载 bytes=%s expected=%s", count, expected)
                    logged = count
        if expected is not None and count != expected:
            raise OSError("下载字节数与响应不符")
        metadata = {
            "repository": cfg["upstream_repository"],
            "source_commit": cfg["upstream_commit"],
            "official_file_id": cfg["official_file_id"],
            "bytes": count,
            "sha256": digest(partial),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "content_disposition": response.headers.get("Content-Disposition"),
            "author_published_checksum": None,
            "seconds": time.monotonic() - started,
            "provenance": "作者README公开链接；首次接收内容SHA，非作者公布校验和",
        }
    partial.replace(path)
    save(verified, metadata)
    save(output / "result.json", metadata)
    logger.info("模型下载完成 %s", metadata)
