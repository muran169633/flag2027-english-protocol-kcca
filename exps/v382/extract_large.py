"""v382 阶段1：更大的通用视觉冻结编码——CLIP ViT-L/14 与 SigLIP ViT-B/16 图像嵌入，逐样本、单样本 L2。

输入为 v379/v380 已有清单（train / vox2t / eval），不新增数据；无跨样本统计。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PARTS = {
    "train": ROOT / "outputs/v379/features/train_faces.json",
    "eval": ROOT / "outputs/v379/features/eval_faces.json",
    "vox2t": ROOT / "outputs/v380/features/vox2t_faces.json",
}
OUT = ROOT / "outputs/v382/features"
MODELS = {"clipL": "openai/clip-vit-large-patch14", "siglip": "google/siglip-base-patch16-224"}


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def main():
    from transformers import AutoImageProcessor, CLIPVisionModelWithProjection, SiglipVisionModel

    OUT.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(0.3, device="cuda:0")
    dev = "cuda:0"
    clip = CLIPVisionModelWithProjection.from_pretrained(MODELS["clipL"]).to(dev).eval()
    cproc = AutoImageProcessor.from_pretrained(MODELS["clipL"])
    sig = SiglipVisionModel.from_pretrained(MODELS["siglip"]).to(dev).eval()
    sproc = AutoImageProcessor.from_pretrained(MODELS["siglip"])
    log("模型加载完成")
    for part in sys.argv[1:]:
        recs = json.loads(PARTS[part].read_text())
        outs = {"clipL": [], "siglip": []}
        for s in range(0, len(recs), 64):
            imgs = [Image.open(r["path"]).convert("RGB") for r in recs[s : s + 64]]
            with torch.no_grad():
                px = cproc(images=imgs, return_tensors="pt")["pixel_values"].to(dev)
                outs["clipL"].append(clip(pixel_values=px).image_embeds.float().cpu().numpy())
                px = sproc(images=imgs, return_tensors="pt")["pixel_values"].to(dev)
                outs["siglip"].append(sig(pixel_values=px).pooler_output.float().cpu().numpy())
            if (s // 64) % 50 == 0:
                log(f"{part} {s}/{len(recs)}")
        res = {}
        for k, v in outs.items():
            v = np.vstack(v)
            res[k] = (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        np.savez(OUT / f"{part}_faces_large.npz", **res)
        log(f"{part} 完成 {len(recs)}")


if __name__ == "__main__":
    main()
