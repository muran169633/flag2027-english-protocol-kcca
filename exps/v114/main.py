"""独立人脸信息准备入口。"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from .common import configuration, digest, save, start_run
from .intake import run


def main() -> None:
    cfg = configuration()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["intake", "inspect", "smoke", "all"],
        default=cfg["default_mode"],
    )
    cfg["mode"] = parser.parse_args().mode
    expected = Path("/home/xxc/miniconda3/envs/my_project_env")
    if Path(sys.prefix) != expected or not Path(sys.executable).is_relative_to(
        expected
    ):
        raise RuntimeError("必须使用指定Conda环境")
    if time.time() >= datetime.fromisoformat(cfg["deadline"]).timestamp() - 120:
        raise RuntimeError("截止收尾窗口")
    output = start_run(cfg, "adaface-official-weight-intake")
    (output / "source").mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, output / "source" / path.name)
    save(output / "source/config.json", cfg)
    files = [*Path(__file__).parent.glob("*.py"), *Path(cfg["upstream_root"]).iterdir()]
    save(output / "inputs.json", {str(p): digest(p) for p in files if p.is_file()})
    if cfg["mode"] == "intake":
        run(cfg, output)
    elif cfg["mode"] == "inspect":
        from .inspection import run as inspect

        inspect(cfg, output)
    else:
        from .features import run as extract

        tracker = None
        if cfg["mode"] == "all":
            try:
                import wandb

                tracker = wandb.init(
                    project=cfg["wandb_project"],
                    mode=cfg["wandb_mode"],
                    dir=str(output),
                    config=cfg,
                    group=cfg["experiment_id"],
                    id=output.name,
                    job_type="feature_extraction",
                )
            except Exception:
                logging.getLogger(__name__).exception("W&B初始化失败，继续独立保存进度")
        try:
            extract(cfg, output, tracker)
        finally:
            if tracker is not None:
                try:
                    tracker.finish()
                except Exception:
                    logging.getLogger(__name__).exception("W&B关闭失败")


if __name__ == "__main__":
    main()
