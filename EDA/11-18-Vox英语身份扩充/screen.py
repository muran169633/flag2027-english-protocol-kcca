"""对既有 VoxCeleb2 外部媒体逐条做英语语言筛选，按原始片段键关联画面。"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.classifiers import EncoderClassifier
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CFG = json.loads((HERE / "config.json").read_text())
OUT = HERE / "output"
LOG = logging.getLogger(__name__)


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_wave(path: str) -> np.ndarray:
    signal, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = signal.mean(axis=1)
    if rate != CFG["audio_rate"]:
        unit = math.gcd(rate, CFG["audio_rate"])
        wave = resample_poly(wave, CFG["audio_rate"] // unit, rate // unit)
    assert wave.size and np.isfinite(wave).all()
    return wave


def labeled_checks(meta: list[dict[str, str]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(CFG["check_seed"])
    rows: list[dict[str, Any]] = []
    for lang in CFG["check_languages"]:
        group = [
            r
            for r in meta
            if r["src"] in {"v1", "v2", "v3"} and r["lang"] == lang and r["video"]
        ]
        seen = set()
        for index in rng.permutation(len(group)):
            r = group[index]
            key = (r["id"], r["video"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "path": r["path"],
                    "id": r["id"],
                    "video": r["video"],
                    "role": "check",
                    "language": lang,
                }
            )
            if len(seen) == CFG["check_per_language"]:
                break
        assert len(seen) == CFG["check_per_language"]
    return rows


@torch.inference_mode()
def main() -> None:
    assert (
        Path(sys.executable).parent.parent
        == Path(sys.prefix)
        == Path(CFG["python_prefix"])
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(OUT / "语言筛选.log"), logging.StreamHandler()],
    )
    LOG.info("解释器 %s 环境 %s", sys.executable, sys.prefix)
    start = time.monotonic()
    torch.set_num_threads(CFG["resource_cpu_threads"])
    torch.cuda.set_per_process_memory_fraction(CFG["resource_gpu_fraction"])
    ref = json.loads((ROOT / CFG["reference_config"]).read_text())
    voice_path = ROOT / ref["metadata"]["voice"][1]
    face_path = ROOT / ref["metadata"]["face"][1]
    voices = json.loads(voice_path.read_text())
    faces = json.loads(face_path.read_text())
    historical = json.loads((ROOT / ref["metadata"]["voice"][0]).read_text())
    rows = labeled_checks(historical)
    for i, r in enumerate(voices):
        assert r["src"] == "vox2t" and r["lang"] == "unknown"
        rows.append(
            {
                "id": r["id"],
                "video": r["video"],
                "path": r["path"],
                "role": "external",
                "index": str(i),
                "language": "unknown",
            }
        )
    sb_path = ROOT / CFG["ecapa_dir"]
    # 本地官方 YAML 及校验过的权重；两个判别器均只执行推理。
    sb = EncoderClassifier.from_hparams(
        source=str(sb_path),
        savedir=str(OUT / "语言模型链接"),
        overrides={"pretrained_path": str(sb_path)},
        run_opts={"device": "cuda"},
    )
    sb.eval()
    en = sb.hparams.label_encoder.lab2ind["en: English"]
    wh_path = Path(CFG["whisper_dir"]).expanduser()
    wh = (
        WhisperForConditionalGeneration.from_pretrained(
            str(wh_path), local_files_only=True, dtype=torch.float16
        )
        .eval()
        .cuda()
    )
    extract = WhisperFeatureExtractor.from_pretrained(
        str(wh_path), local_files_only=True
    )
    tokens = json.loads((ROOT / CFG["whisper_language_tokens"]).read_text())
    langs = [key[2:-2] for key in tokens]
    token_indices = torch.tensor(list(tokens.values()), device="cuda")
    wi = langs.index("en")
    batch = CFG["language_batch"]
    for begin in range(0, len(rows), batch):
        current = rows[begin : begin + batch]
        waves = [read_wave(r["path"]) for r in current]
        lengths = [len(x) for x in waves]
        if max(lengths) > CFG["max_seconds"] * CFG["audio_rate"]:
            waves = [x[: int(CFG["max_seconds"] * CFG["audio_rate"])] for x in waves]
        features = extract(
            waves, sampling_rate=CFG["audio_rate"], return_tensors="pt", device="cuda"
        ).input_features
        ids = torch.full(
            (len(waves), 1), wh.config.decoder_start_token_id, device="cuda"
        )
        logits = wh(
            input_features=features.to(device="cuda", dtype=torch.float16),
            decoder_input_ids=ids,
            use_cache=False,
        ).logits[:, 0, token_indices]
        wp = logits.float().softmax(1).cpu().numpy()
        for j, (r, x) in enumerate(zip(current, waves, strict=True)):
            # ECAPA 单条推理，短音频不受同批零填充影响。
            prob, _, _, lab = sb.classify_batch(torch.from_numpy(x))
            sb_en = float(prob[0, en].exp().item())
            r.update(
                {
                    "seconds": lengths[j] / CFG["audio_rate"],
                    "whisper_top": langs[int(wp[j].argmax())],
                    "whisper_en": float(wp[j, wi]),
                    "ecapa_top": str(lab[0]).split(":")[0],
                    "ecapa_en": sb_en,
                }
            )
            r["selected"] = bool(
                CFG["min_seconds"] <= r["seconds"] <= CFG["max_seconds"]
                and r["whisper_top"] == r["ecapa_top"] == "en"
                and r["whisper_en"] >= CFG["min_whisper_en"]
                and sb_en >= CFG["min_ecapa_en"]
            )
        if begin % (batch * 20) == 0:
            LOG.info(
                "已判别 %d/%d，耗时 %.1fs",
                begin + len(waves),
                len(rows),
                time.monotonic() - start,
            )
    source_rows = [r for r in rows if r["role"] == "external"]
    selected = {
        (r["id"], Path(r["path"]).stem): int(r["index"])
        for r in source_rows
        if r["selected"]
    }
    eligible_faces = {
        j
        for j, r in enumerate(faces)
        if (r["id"], Path(r["path"]).stem.rsplit("_", 1)[0]) in selected
    }
    counts = Counter(r["id"] for r in source_rows if r["selected"])
    keep_ids = {
        n
        for n, count in counts.items()
        if count >= CFG["minimum_videos"]
        and len({r["video"] for r in source_rows if r["id"] == n and r["selected"]})
        >= CFG["minimum_videos"]
    }
    voice_idx = sorted(v for (n, _), v in selected.items() if n in keep_ids)
    face_idx = sorted(j for j in eligible_faces if faces[j]["id"] in keep_ids)
    matched = Counter(
        (faces[j]["id"], Path(faces[j]["path"]).stem.rsplit("_", 1)[0])
        for j in face_idx
    )
    assert matched == Counter(
        {(voices[i]["id"], Path(voices[i]["path"]).stem): 2 for i in voice_idx}
    )
    checks = {}
    for lang in CFG["check_languages"]:
        rr = [r for r in rows if r["role"] == "check" and r["language"] == lang]
        checks[lang] = {
            "total": len(rr),
            "selected": sum(r["selected"] for r in rr),
            "duration_eligible": sum(
                CFG["min_seconds"] <= r["seconds"] <= CFG["max_seconds"] for r in rr
            ),
        }
    result: dict[str, Any] = {
        "checks": checks,
        "external_voices": len(voices),
        "selected_before_identity_minimum": len(selected),
        "identities": len(keep_ids),
        "selected_voice_rows": voice_idx,
        "selected_face_rows": face_idx,
        "config_sha256": digest(HERE / "config.json"),
        "voice_metadata_sha256": digest(voice_path),
        "face_metadata_sha256": digest(face_path),
        "model_sha256": {
            p.name: digest(p)
            for p in [
                wh_path / "model.safetensors",
                sb_path / "embedding_model.ckpt",
                sb_path / "classifier.ckpt",
                sb_path / "label_encoder.txt",
                sb_path / "hyperparams.yaml",
            ]
        },
        "elapsed": time.monotonic() - start,
    }
    (OUT / "逐片段语言判别.json").write_text(json.dumps(rows, ensure_ascii=False))
    (OUT / "英语片段筛选.json").write_text(json.dumps(result, indent=2))
    LOG.info(
        "筛选结束 %d 个身份，%d 段录音，%d 张脸；语言检查 %s",
        len(keep_ids),
        len(voice_idx),
        len(face_idx),
        checks,
    )


if __name__ == "__main__":
    main()
