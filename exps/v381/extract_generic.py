"""v381 阶段1：通用视觉冻结编码（CLIP ViT-B/16 图像嵌入、DINOv2 ViT-B/14 CLS），逐样本、单样本 L2。

动机：身份识别人脸模型（ArcFace/AdaFace）刻意去除年龄/体型/肤色等软生物特征，
而通用表示保留这些与声音相关的线索（v379 中 VGG fc7 优于 ArcFace/AdaFace）。
输入为 v379/v380 已有清单（train / vox2t / eval），不新增数据。
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
OUT = ROOT / "outputs/v381/features"
CLIP_ID = "openai/clip-vit-base-patch16"


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def main():
    import timm
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    OUT.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(0.25, device="cuda:0")
    dev = "cuda:0"
    clip = CLIPVisionModelWithProjection.from_pretrained(CLIP_ID).to(dev).eval()
    proc = CLIPImageProcessor.from_pretrained(CLIP_ID)
    dino = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0).to(dev).eval()
    dcfg = timm.data.resolve_model_data_config(dino)
    dtf = timm.data.create_transform(**dcfg, is_training=False)
    log(f"DINOv2 预处理 {dcfg}")
    for part in sys.argv[1:]:
        recs = json.loads(PARTS[part].read_text())
        c_out, d_out = [], []
        for s in range(0, len(recs), 64):
            imgs = [Image.open(r["path"]).convert("RGB") for r in recs[s : s + 64]]
            with torch.no_grad():
                px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev)
                c_out.append(clip(pixel_values=px).image_embeds.float().cpu().numpy())
                dx = torch.stack([dtf(im) for im in imgs]).to(dev)
                d_out.append(dino(dx).float().cpu().numpy())
            if (s // 64) % 50 == 0:
                log(f"{part} {s}/{len(recs)}")
        res = {}
        for k, v in (("clip", np.vstack(c_out)), ("dino", np.vstack(d_out))):
            res[k] = (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        np.savez(OUT / f"{part}_faces_generic.npz", **res)
        log(f"{part} 完成 {len(recs)}")


if __name__ == "__main__":
    main()
