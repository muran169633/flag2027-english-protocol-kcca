"""下载固定公开版本并校验文件，不执行训练或访问测试数据。"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("公开模型下载")


def fetch(config: dict[str, Any]) -> None:
    spec = config["resource"]
    target = ROOT / spec["directory"]
    target.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(target / "下载.log")],
    )
    started = time.monotonic()
    records = []
    for name in [*spec["small_files"], spec["weights"]]:
        path = target / name
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if name != spec["weights"] or digest == spec["weights_sha256"]:
                records.append(
                    {"file": name, "sha256": digest, "bytes": path.stat().st_size}
                )
                continue
        url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{name}"
        request = urllib.request.Request(
            url, headers={"User-Agent": "FLAG-public-research/1.0"}
        )
        temporary = target / (name + ".part")
        LOGGER.info("下载=%s", name)
        with (
            urllib.request.urlopen(
                request, timeout=spec["download_timeout_seconds"]
            ) as response,
            temporary.open("wb") as stream,
        ):
            while chunk := response.read(spec["chunk_bytes"]):
                if time.monotonic() - started > spec["download_budget_seconds"]:
                    raise TimeoutError("公开资源下载达到固定预算，保留未完成文件")
                stream.write(chunk)
        with temporary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if name == spec["weights"] and (
            digest != spec["weights_sha256"]
            or temporary.stat().st_size != spec["weights_bytes"]
        ):
            raise ValueError("公开模型的大小或SHA256不符")
        temporary.replace(path)
        records.append({"file": name, "sha256": digest, "bytes": path.stat().st_size})
    (target / "来源核验.json").write_text(
        json.dumps(
            {
                "repo": spec["repo"],
                "revision": spec["revision"],
                "license": spec["license"],
                "files": records,
                "seconds": time.monotonic() - started,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    LOGGER.info("固定公开资源校验完成，耗时=%.2f秒", time.monotonic() - started)


if __name__ == "__main__":
    fetch(json.loads(Path(__file__).with_name("config.json").read_text()))
