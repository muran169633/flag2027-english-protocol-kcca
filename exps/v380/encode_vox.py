"""v380 阶段1：VoxCeleb2 抽取媒体的逐样本冻结编码（与 v379 完全相同的七个编码器）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    src, part = Path(sys.argv[1]), sys.argv[2]
    out = ROOT / "outputs/v380/features"
    out.mkdir(parents=True, exist_ok=True)
    from exps.v379.extract import encode, log
    from exps.v007.config import Config
    from exps.v007.encoders import FaceEncoder, load_face
    from exps.v372.submit import FrozenVoice

    faces = json.loads((src / "faces.json").read_text())
    voices = json.loads((src / "voices.json").read_text())
    (out / f"{part}_faces.json").write_text(json.dumps(faces))
    (out / f"{part}_voices.json").write_text(json.dumps(voices))
    fe = encode(faces, [], out)
    ve = encode([], voices, out)
    cfg = Config()
    enc = FaceEncoder(cfg).to("cuda:0")
    vg = []
    for s in range(0, len(faces), 32):
        x = torch.stack([load_face(Path(r["path"]), cfg) for r in faces[s : s + 32]]).to("cuda:0")
        with torch.no_grad():
            vg.append(enc(x)[1].float().cpu().numpy())
    vg = np.vstack(vg)
    fe["vgg"] = (vg / np.maximum(np.linalg.norm(vg, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    vcfg = json.loads((ROOT / "exps/v372/config.json").read_text())
    vcfg["device"] = "cuda:0"
    voice = FrozenVoice(vcfg)
    ve["wavlm"] = np.stack([voice.encode(Path(r["path"])) for r in voices]).astype(np.float32)
    np.savez(out / f"{part}_faces.npz", **fe)
    np.savez(out / f"{part}_voices.npz", **ve)
    log(f"{part} 完成 脸 {len(faces)} 声 {len(voices)}")


if __name__ == "__main__":
    main()
