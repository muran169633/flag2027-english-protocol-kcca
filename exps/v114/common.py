"""本实验的配置、运行记录与基础计算。"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize(values: Array) -> Array:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def configuration() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("config.json").read_text())


def start_run(cfg: dict[str, Any], purpose: str) -> Path:
    run_id = datetime.now().strftime("%m-%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    output = Path(cfg["output_root"]) / cfg["experiment_id"] / run_id
    output.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output / "run.log", encoding="utf-8"),
        ],
    )
    save(output / "config.json", cfg)
    save(
        output / "environment.json",
        {
            "sys_executable": sys.executable,
            "sys_prefix": sys.prefix,
            "command": sys.argv,
            "purpose": purpose,
        },
    )
    return output
