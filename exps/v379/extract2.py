"""v379 阶段1b：逐样本冻结 VGG-Face fc7（4096，与官方特征逐值一致）与 WavLM-base-plus-sv（512）。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FEAT = ROOT / "outputs/v379/features"


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def main():
    from exps.v007.config import Config
    from exps.v007.encoders import FaceEncoder, load_face
    from exps.v372.submit import FrozenVoice

    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(0.25, device="cuda:0")
    cfg = Config()
    enc = FaceEncoder(cfg).to("cuda:0")
    vcfg = json.loads((ROOT / "exps/v372/config.json").read_text())
    vcfg["device"] = "cuda:0"
    voice = FrozenVoice(vcfg)
    for part in sys.argv[1:]:
        fm = json.loads((FEAT / f"{part}_faces.json").read_text())
        vm = json.loads((FEAT / f"{part}_voices.json").read_text())
        out = []
        for s in range(0, len(fm), 32):
            x = torch.stack([load_face(Path(r["path"]), cfg) for r in fm[s : s + 32]]).to("cuda:0")
            with torch.no_grad():
                _, g = enc(x)
            out.append(g.float().cpu().numpy())
            if (s // 32) % 100 == 0:
                log(f"{part} VGG {s}/{len(fm)}")
        vgg = np.vstack(out)
        vgg = vgg / np.maximum(np.linalg.norm(vgg, axis=1, keepdims=True), 1e-12)
        wl = []
        for i, r in enumerate(vm):
            wl.append(voice.encode(Path(r["path"])))
            if i % 2000 == 0:
                log(f"{part} WavLM {i}/{len(vm)}")
        np.savez(FEAT / f"{part}_faces_vgg.npz", vgg=vgg.astype(np.float16))
        np.savez(FEAT / f"{part}_voices_wavlm.npz", wavlm=np.stack(wl).astype(np.float32))
        log(f"{part} 完成")


if __name__ == "__main__":
    main()
