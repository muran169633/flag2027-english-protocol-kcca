"""只用训练身份选择公开冻结年龄模型的一致性项。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits

from .model import Projection

Array = NDArray[Any]
ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("嵌套身份验证")


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unit(values: Array, floor: float) -> Array:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), floor)


def eer(labels: Array, values: Array) -> float:
    fpr, tpr, _ = roc_curve(labels, values, drop_intermediate=False)
    difference = fpr + tpr - 1
    right = int(np.flatnonzero(difference >= 0)[0])
    if right == 0:
        return float(fpr[0])
    left = right - 1
    return float(
        fpr[left]
        - difference[left]
        * (fpr[right] - fpr[left])
        / (difference[right] - difference[left])
    )


class History:
    """W&B 故障不影响模型计算，JSONL 始终独立保存。"""

    def __init__(self, config: dict[str, Any], directory: Path) -> None:
        self.directory = directory
        self.rows: list[dict[str, Any]] = []
        self.tracker: Any = None
        if config["mode"] == "full":
            try:
                import wandb

                self.tracker = wandb.init(
                    project=config["wandb"]["project"],
                    mode=config["wandb"]["mode"],
                    dir=str(directory),
                    group=config["experiment_id"],
                    job_type=config["wandb"]["job_type"],
                    id=config["run_id"],
                    config=config,
                )
            except Exception:
                LOGGER.exception("W&B初始化失败，继续独立记录")

    def log(self, values: dict[str, Any]) -> None:
        row = {"step": len(self.rows), **values}
        self.rows.append(row)
        with (self.directory / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.tracker is not None:
            try:
                self.tracker.log(row, step=row["step"])
            except Exception:
                LOGGER.exception("W&B记录失败，JSONL已保存")

    def finish(self) -> None:
        fields = sorted(set().union(*(r.keys() for r in self.rows)))
        with (self.directory / "history.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)
        if self.tracker is not None:
            try:
                self.tracker.finish()
            except Exception:
                LOGGER.exception("W&B关闭失败，保留本地记录")


@torch.inference_mode()
def scores(
    model: Projection, features: dict[str, Array], pairs: Array, config: dict[str, Any]
) -> Array:
    output = []
    model.eval()
    for start in range(0, len(pairs), config["head_batch_size"]):
        batch = pairs[start : start + config["head_batch_size"]]
        output.append(
            model(
                torch.from_numpy(features["face"][batch[:, 0]]).to(config["device"]),
                torch.from_numpy(features["voice"][batch[:, 1]]).to(config["device"]),
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(output)


def frozen_reference(source: Path, fold: int, config: dict[str, Any]) -> Projection:
    trained = read(source / "config.json")
    if "projection_activation" not in trained:
        trained["projection_activation"] = config["network"]["projection_activation"]
    provenance = read(source / "data_sources.json")
    require(
        provenance["source"] == "official_v4_only"
        and provenance["history_training_used"] is False,
        "基线训练来源不符",
    )
    result = read(source / "result.json")
    summary = next(x for x in result["folds"] if x["fold"] == fold)
    path = source / f"fold_{fold}" / "best.pt"
    require(digest(path) == summary["checkpoint_sha256"], "冻结基线权重变化")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Projection(trained, checkpoint["num_classes"], config["seed"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(config["device"]).requires_grad_(False).eval()


def age_map(
    faces: Array, voices: Array, labels: Array, ids: list[str], config: dict[str, Any]
) -> dict[str, float]:
    f = np.array([faces[labels == name].mean() for name in ids])
    v = np.array([voices[labels == name].mean() for name in ids])
    slope = float(
        np.mean((f - f.mean()) * (v - v.mean()))
        / (np.mean((v - v.mean()) ** 2) + config["age_mapping"]["ridge"])
    )
    return {"slope": slope, "intercept": float(f.mean() - slope * v.mean())}


def penalty(
    faces: Array, voices: Array, pairs: Array, mapping: dict[str, float]
) -> Array:
    return (
        faces[pairs[:, 0]]
        - mapping["slope"] * voices[pairs[:, 1]]
        - mapping["intercept"]
    ) ** 2


def execute(
    config: dict[str, Any], directory: Path, history: History, deadline: float
) -> dict[str, Any]:
    for name, item in config["inputs"].items():
        require(
            digest(ROOT / item["path"]) == item["sha256"], f"训练输入指纹变化：{name}"
        )
    labels = np.array(
        [
            line.split()[4]
            for line in (ROOT / config["inputs"]["train_list"]["path"])
            .read_text()
            .splitlines()
        ]
    )
    source = ROOT / config["inner_reference"]
    for filename, expected in config["inner_reference_sha256"].items():
        require(digest(source / filename) == expected, "训练身份内层参考指纹变化")
    trait_run = ROOT / config["trait_feature_run"]
    trait_result = read(trait_run / "result.json")
    trait_config = read(trait_run / "config.json")
    require(
        trait_result["mode"] == "full"
        and trait_result["status"] == "complete"
        and trait_result["rows"] == len(labels)
        and not trait_result["progress_evaluation_used"]
        and not trait_result["audio_encoder_updated"],
        "需要完整且冻结的官方训练属性特征",
    )
    require(
        digest(trait_run / "features.npz") == trait_result["feature_sha256"]
        and trait_config["resource"]["revision"] == config["resource"]["revision"],
        "属性缓存内容或模型版本不符",
    )
    with np.load(trait_run / "features.npz", allow_pickle=False) as data:
        face_age, voice_age = (
            data["face"][:, 2].astype(np.float64),
            data["voice"][:, 0].astype(np.float64),
        )
    features = {
        kind: np.load(ROOT / config["inputs"][name]["path"], allow_pickle=False).astype(
            np.float32
        )
        for kind, name in (("face", "vgg"), ("voice", "wavlm"))
    }
    folds = []
    for fold in config["folds"][:1] if config["mode"] == "smoke" else config["folds"]:
        if time.monotonic() > deadline:
            raise TimeoutError("固定年龄项验证达到预算")
        target = directory / f"fold_{fold}"
        target.mkdir()
        inner_dir = source / f"fold_{fold}"
        split = read(inner_dir / "split.json")
        require(
            not set(split["fit"]) & set(split["inner_select"])
            and not (set(split["fit"]) | set(split["inner_select"]))
            & set(split["outer_evaluate"]),
            "内层身份划分泄漏",
        )
        save(target / "split.json", split)
        current_map = age_map(face_age, voice_age, labels, split["fit"], config)
        saved = torch.load(
            inner_dir / "inner_best.pt", map_location="cpu", weights_only=True
        )
        model = Projection(config["network"], saved["classes"], config["seed"])
        model.load_state_dict(saved["model"], strict=True)
        model = model.to(config["device"]).requires_grad_(False).eval()
        selected = {}
        with np.load(inner_dir / "inner_pairs.npz", allow_pickle=False) as cache:
            for protocol in ("standard", "same_gender"):
                pairs = cache[protocol]
                require(
                    set(labels[pairs[:, :2]].ravel()) <= set(split["inner_select"]),
                    "选权重配对包含拟合或外层身份",
                )
                base = scores(model, features, pairs, config)
                age_penalty = penalty(face_age, voice_age, pairs, current_map)
                choices = [
                    {
                        "weight": weight,
                        "eer": eer(pairs[:, 2], base - weight * age_penalty),
                    }
                    for weight in config["age_weights"]
                ]
                selected[protocol] = min(choices, key=lambda x: x["eer"])
                for index, choice in enumerate(choices):
                    history.log(
                        {
                            "phase": "inner_age",
                            "fold": fold,
                            "protocol": protocol,
                            "candidate": index,
                            **choice,
                        }
                    )
        save(
            target / "selection_before_outer.json",
            {"mapping": current_map, "selected": selected},
        )
        if config["mode"] == "smoke":
            return {
                "status": "complete",
                "mode": "smoke",
                "inner_selected": selected,
                "outer_eer_opened": False,
            }
        train_ids = sorted(set(split["fit"]) | set(split["inner_select"]))
        outer_map = age_map(face_age, voice_age, labels, train_ids, config)
        save(
            target / "age_model.json",
            {
                "mapping": outer_map,
                "weights": {p: c["weight"] for p, c in selected.items()},
                "train_identities": train_ids,
            },
        )
        output: dict[str, Any] = {
            "fold": fold,
            "selected": selected,
            "mapping": outer_map,
            "baseline": {},
            "candidate": {},
            "independence": {},
        }
        predictions: dict[str, Any] = {}
        for protocol in ("standard", "same_gender"):
            reference = ROOT / config["reference_sources"][protocol]
            reference_dir = reference / f"fold_{fold}"
            reference_split = read(reference_dir / "split.json")
            require(
                set(reference_split["train_speakers"]) == set(train_ids)
                and set(reference_split["validation_speakers"])
                == set(split["outer_evaluate"]),
                "原匹配头身份划分不符",
            )
            with np.load(reference_dir / "oof_scores.npz", allow_pickle=False) as data:
                pairs, base = (
                    data[f"{protocol}_pairs"].copy(),
                    data[f"{protocol}_scores"].copy(),
                )
            require(
                set(labels[pairs[:, :2]].ravel()) <= set(split["outer_evaluate"])
                and bool(
                    np.array_equal(
                        labels[pairs[:, 0]] == labels[pairs[:, 1]], pairs[:, 2] == 1
                    )
                ),
                "外层身份或标签不符",
            )
            model = frozen_reference(reference, fold, config)
            replay = scores(model, features, pairs, config)
            require(
                float(np.max(np.abs(replay - base)))
                <= config["compatibility_tolerance"],
                "原匹配头回放不符",
            )
            weight = selected[protocol]["weight"]
            candidate = base - weight * penalty(face_age, voice_age, pairs, outer_map)
            output["baseline"][protocol], output["candidate"][protocol] = (
                eer(pairs[:, 2], base),
                eer(pairs[:, 2], candidate),
            )
            queries = pairs[: config["independence_rows"]]

            def independent(
                current: Array,
                model: Projection = model,
                weight: float = weight,
                mapping: dict[str, float] = outer_map,
            ) -> Array:
                return scores(model, features, current, config) - weight * penalty(
                    face_age, voice_age, current, mapping
                )

            batch_values = independent(queries)
            error = float(
                max(
                    np.max(np.abs(batch_values - independent(queries[::-1])[::-1])),
                    np.max(
                        np.abs(
                            batch_values
                            - np.array(
                                [independent(row[None, :])[0] for row in queries]
                            )
                        )
                    ),
                )
            )
            require(
                error <= config["compatibility_tolerance"], "年龄项预测不是逐trial独立"
            )
            output["independence"][protocol] = error
            (
                predictions[f"{protocol}_pairs"],
                predictions[f"{protocol}_scores"],
                predictions[f"{protocol}_baseline"],
            ) = pairs, candidate, base
        output["baseline_mean"], output["candidate_mean"] = (
            sum(output["baseline"].values()) / 2,
            sum(output["candidate"].values()) / 2,
        )
        np.savez(target / "outer_predictions.npz", **predictions)
        save(target / "result.json", output)
        folds.append(output)
        history.log(
            {
                "phase": "outer",
                "fold": fold,
                "outer/baseline_mean": output["baseline_mean"],
                "outer/candidate_mean": output["candidate_mean"],
            }
        )
        LOGGER.info(
            "阶段=年龄外层完成 fold=%d baseline=%.4f candidate=%.4f",
            fold,
            output["baseline_mean"],
            output["candidate_mean"],
        )
    baseline = {
        p: float(np.mean([f["baseline"][p] for f in folds]))
        for p in ("standard", "same_gender")
    }
    candidate = {
        p: float(np.mean([f["candidate"][p] for f in folds])) for p in baseline
    }
    improvement = (sum(baseline.values()) - sum(candidate.values())) / 2
    nonworse = sum(f["candidate_mean"] <= f["baseline_mean"] for f in folds)
    gate = config["promotion"]
    promote = (
        improvement >= gate["minimum_mean_improvement"]
        and nonworse >= gate["minimum_nonworse_folds"]
        and all(
            candidate[p] <= baseline[p] + gate["maximum_per_protocol_regression"]
            for p in baseline
        )
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties

        target = directory / "plots"
        target.mkdir()
        figure, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(folds))
        ax.bar(
            x - 0.18, [100 * f["baseline_mean"] for f in folds], 0.36, label="baseline"
        )
        ax.bar(
            x + 0.18,
            [100 * f["candidate_mean"] for f in folds],
            0.36,
            label="age consistency",
        )
        ax.set_xlabel("fold")
        ax.set_ylabel("EER (%)")
        ax.set_title(
            "冻结年龄一致性项：训练身份留出验证",
            fontproperties=FontProperties(fname=config["plot_font"]),
        )
        ax.legend()
        figure.tight_layout()
        figure.savefig(target / "年龄项外层对照.png", dpi=config["plot_dpi"])
        plt.close(figure)
    except Exception:
        LOGGER.exception("绘图失败，保留全部指标")
    return {
        "status": "complete",
        "mode": "full",
        "baseline": baseline,
        "candidate": candidate,
        "mean_improvement": improvement,
        "nonworse_folds": nonworse,
        "promotion_passed": promote,
        "folds": folds,
        "scope": "仅训练身份英语五折，内层代理匹配头选年龄权重；外层已用于历史研究，非全新盲测",
        "progress_evaluation_used": False,
        "new_submission_generated": False,
    }


def main() -> int:
    config_path = Path(__file__).with_name("config.json")
    config = read(config_path)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default=config["mode"])
    config["mode"] = parser.parse_args().mode
    expected = Path(config["training_python"]).resolve().parent.parent
    require(
        Path(sys.prefix).resolve() == expected
        and Path(sys.executable).resolve().parent.parent == expected,
        "训练必须使用my_project_env",
    )
    config["run_id"] = (
        datetime.now().astimezone().strftime("%m-%d-%H%M%S")
        + "-"
        + config["mode"]
        + "-"
        + uuid.uuid4().hex[:6]
    )
    config["entrypoint"] = (
        f"{sys.executable} -m exps.{config['experiment_id']}.{Path(__file__).stem} --mode {config['mode']}"
    )
    directory = (
        ROOT / config["output_root"] / config["experiment_id"] / config["run_id"]
    )
    directory.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(directory / "run.log", encoding="utf-8"),
        ],
    )
    save(directory / "config.json", config)
    save(
        directory / "environment.json",
        {
            "executable": sys.executable,
            "prefix": sys.prefix,
            "packages": {
                p: importlib.metadata.version(p)
                for p in [
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "wandb",
                    "matplotlib",
                    "torch",
                ]
            },
        },
    )
    snapshot = directory / "source"
    snapshot.mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, snapshot / source.name)
    shutil.copy2(config_path, snapshot / config_path.name)
    LOGGER.info(
        "阶段=启动 run=%s 预算=%ds", config["run_id"], config["compute_budget_seconds"]
    )
    torch.set_num_threads(config["cpu_threads"])
    require(torch.cuda.is_available(), "配置要求CUDA设备")
    total_memory = torch.cuda.get_device_properties(config["device"]).total_memory
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, config["gpu_memory_limit_gib"] * 1024**3 / total_memory),
        device=config["device"],
    )
    torch.backends.cuda.matmul.allow_tf32 = config["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = config["allow_tf32"]
    started = time.monotonic()
    history = History(config, directory)
    try:
        with threadpool_limits(config["cpu_threads"]):
            result = execute(
                config, directory, history, started + config["compute_budget_seconds"]
            )
        result["wall_seconds"] = time.monotonic() - started
        save(directory / "result.json", result)
        (directory / "exit_code.txt").write_text("0\n")
        LOGGER.info(
            "阶段=完成 结果=%s",
            json.dumps(
                {k: v for k, v in result.items() if k != "folds"}, ensure_ascii=False
            ),
        )
        return 0
    except Exception:
        LOGGER.exception("阶段=失败")
        (directory / "exit_code.txt").write_text("1\n")
        raise
    finally:
        history.finish()


if __name__ == "__main__":
    raise SystemExit(main())
