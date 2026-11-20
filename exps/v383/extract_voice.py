"""v383 阶段1：通用语音冻结编码，逐样本、单样本 L2。

- age6：audeering wav2vec2-large-robust-6-ft-age-gender 隐藏层均值（1024，v377 已下载并核验的本地权重）；
- whisper：openai/whisper-small 编码器，按真实长度截取帧后均值（768）。
音频与其他语音特征一致：单声道、16 kHz、中心截取最多 6 秒。无跨样本统计。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PARTS = {
    "train": ROOT / "outputs/v379/features/train_voices.json",
    "eval": ROOT / "outputs/v379/features/eval_voices.json",
    "vox2t": ROOT / "outputs/v380/features/vox2t_voices.json",
}
OUT = ROOT / "outputs/v383/features"
AGE = ROOT / "data/pretrained/attributes/audeering_6/a681b720dafd12b9dd7b6d13fb437c7b6b197fd3"
SR, SEC = 16000, 6.0


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def wave(path):
    v, r = sf.read(path, dtype="float32", always_2d=True)
    w = torch.from_numpy(v.mean(1))
    if r != SR:
        w = torchaudio.functional.resample(w, r, SR)
    n = int(SR * SEC)
    if len(w) > n:
        s = (len(w) - n) // 2
        w = w[s : s + n]
    if len(w) < 1600:
        w = torch.nn.functional.pad(w, (0, 1600 - len(w)))
    return w.numpy()


def main():
    from transformers import Wav2Vec2FeatureExtractor, WhisperFeatureExtractor, WhisperModel

    from exps.v377.traits import AgeGender

    OUT.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(0.3, device="cuda:0")
    dev = "cuda:0"
    age = AgeGender.from_pretrained(str(AGE)).to(dev).eval()
    aproc = Wav2Vec2FeatureExtractor.from_pretrained(str(AGE))
    wenc = WhisperModel.from_pretrained("openai/whisper-small").encoder.to(dev).eval()
    wproc = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    log("模型加载完成")
    for part in sys.argv[1:]:
        recs = json.loads(PARTS[part].read_text())
        a_out, w_out = [], []
        for i, r in enumerate(recs):
            w = wave(r["path"])
            with torch.no_grad():
                x = aproc(w, sampling_rate=SR, return_tensors="pt")["input_values"].to(dev)
                a_out.append(age(x)[0][0].float().cpu().numpy())
                f = wproc(w, sampling_rate=SR, return_tensors="pt")["input_features"].to(dev)
                h = wenc(f).last_hidden_state[0]  # [1500, 768]，每帧 20 ms
                frames = max(1, min(h.shape[0], int(np.ceil(len(w) / SR * 50))))
                w_out.append(h[:frames].mean(0).float().cpu().numpy())
            if i % 2000 == 0:
                log(f"{part} {i}/{len(recs)}")
        res = {}
        for k, v in (("age6", np.stack(a_out)), ("whisper", np.stack(w_out))):
            res[k] = (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        np.savez(OUT / f"{part}_voices_generic.npz", **res)
        log(f"{part} 完成 {len(recs)}")


if __name__ == "__main__":
    main()
