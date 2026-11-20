"""v318：ResNet293-LM 截断6秒 vs 完整长度 vs 多窗平均 的声纹提取。

模式：
  crop6 : 与v088相同（取中间6秒），用于逐值回放v088缓存确认实现一致；
  full  : 完整波形一次前向（超过max_full_seconds时取中间该长度，防显存溢出）；
  multi : 6秒窗、3秒步长滑窗，各窗L2归一化后平均再归一化（不足6秒等同crop6）。
python -m exps.v318.extract --sources official_train flag_dev_unlabelled historical_expanded
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from torchaudio.compliance import kaldi

from .model import SpeakerResNet

CFG88 = json.loads(Path("exps/v088/config.json").read_text())
MAX_FULL = 60.0


def wave(path: str) -> torch.Tensor:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    w = torch.from_numpy(x.mean(1).copy())
    if sr != CFG88["sample_rate"]:
        w = F.resample(w, sr, CFG88["sample_rate"])
    return w


def fb(w: torch.Tensor) -> torch.Tensor:
    c = CFG88
    m = kaldi.fbank(w.unsqueeze(0) * c["pcm_scale"], num_mel_bins=c["mel_bins"], frame_length=c["frame_length_ms"],
                    frame_shift=c["frame_shift_ms"], dither=c["fbank_dither"], sample_frequency=c["sample_rate"],
                    window_type=c["window_type"], use_energy=c["use_energy"])
    if len(m) < c["minimum_fbank_frames"]:
        m = m.repeat(math.ceil(c["minimum_fbank_frames"] / len(m)), 1)[: c["minimum_fbank_frames"]]
    return m - m.mean(0, keepdim=True)


def center(w, sec):
    n = round(sec * CFG88["sample_rate"])
    if len(w) > n:
        s = (len(w) - n) // 2
        w = w[s : s + n]
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    a = ap.parse_args()
    if sys.prefix != "/home/xxc/miniconda3/envs/my_project_env":
        raise RuntimeError("解释器不属于指定Conda环境")
    torch.set_num_threads(2)
    model = SpeakerResNet(CFG88).eval().requires_grad_(False)
    state = torch.load(Path(CFG88["weights_root"]) / CFG88["checkpoint_file"], map_location="cpu", weights_only=True)
    tgt = model.state_dict()
    model.load_state_dict({k: v for k, v in state.items() if k in tgt}, strict=True)
    torch.cuda.set_per_process_memory_fraction(0.2, device="cuda:0")
    model = model.to("cuda:0")
    src = {s["name"]: s for s in CFG88["sources"]}
    out = Path("outputs/v318") / datetime.now().strftime("%m-%d-%H%M%S")
    out.mkdir(parents=True)
    ref = np.load("outputs/v088/11-07-122205-36c186/features.npz")
    sr = CFG88["sample_rate"]

    def emb(w):
        with torch.inference_mode():
            e = model(fb(w).unsqueeze(0).to("cuda:0")).cpu().numpy()[0]
        return e / np.linalg.norm(e)

    log = {}
    for name in a.sources:
        recs = json.loads(Path(src[name]["manifest"]).read_text())
        roles = src[name]["roles"]
        keys, paths, lookup = {}, [], np.full(len(recs), -1)
        for i, r in enumerate(recs):
            if roles and r.get("role") not in roles:
                continue
            k = r["voice_sha256"]
            if k not in keys:
                keys[k] = len(paths)
                paths.append(r["voice_path"])
            lookup[i] = keys[k]
        res = {m: np.zeros((len(paths), 256), np.float32) for m in ("crop6", "full", "multi")}
        dur = np.zeros(len(paths))
        t0 = time.monotonic()
        for j, p in enumerate(paths):
            w = wave(p)
            dur[j] = len(w) / sr
            res["crop6"][j] = emb(center(w, 6.0))
            res["full"][j] = emb(center(w, MAX_FULL)) if dur[j] > 6.0 else res["crop6"][j]
            if dur[j] > 6.0:
                win, hop = 6 * sr, 3 * sr
                starts = list(range(0, len(w) - win + 1, hop))
                if starts[-1] + win < len(w):
                    starts.append(len(w) - win)
                v = np.mean([emb(w[s : s + win]) for s in starts], 0)
                res["multi"][j] = v / np.linalg.norm(v)
            else:
                res["multi"][j] = res["crop6"][j]
            if j % 500 == 0:
                print(name, j, len(paths), f"{time.monotonic() - t0:.0f}s", flush=True)
        # 回放v088缓存（crop6应一致）
        rows = np.flatnonzero(lookup >= 0)
        old = ref["voice"][ref[name][rows]]
        new = res["crop6"][lookup[rows]]
        cos = float(np.min(np.sum(old * new, 1)))
        log[name] = {"voices": len(paths), "seconds": time.monotonic() - t0, "crop6_vs_v088_min_cosine": cos,
                     "over6s_fraction": float((dur > 6).mean())}
        print(name, log[name], flush=True)
        np.savez_compressed(out / f"{name}.npz", lookup=lookup, duration=dur, **res)
    (out / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print(out)


if __name__ == "__main__":
    main()
