"""v379 阶段1：逐样本冻结编码（v4 官方训练 + MAV-Celeb v1/v2/v3 历史带身份数据 + Evaluation）。

每条媒体单独编码，只做单样本 L2 归一化；不使用任何跨样本统计（FAQ 13）。
编码器与 v353 阶段1相同：ArcFace glintr100、AdaFace、ECAPA、ERes2NetV2、ResNet293。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "data/MAV_Celeb_history"
FLAG = ROOT / "data/FLAG_2027"
SOURCES = {"v1": HIST / "mavceleb_v1_complete/v1", "v2": HIST / "mavceleb_v2_complete/v2", "v3": HIST / "mavceleb_v3_tran"}
FACES_PER_VIDEO = 6
VOICES_PER_VIDEO = 12


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def spread(items, k):
    items = sorted(items)
    if len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idx]


def build_records():
    faces, voices = [], []
    # v4 官方训练：一脸一声，同名配对
    for line in (FLAG / "train_set/train_set/train_English.txt").read_text().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        ident = parts[4]
        v = FLAG / "train_set/train_set" / parts[2]
        f = FLAG / "train_set/train_set" / parts[3]
        faces.append({"src": "v4", "id": f"v4_{ident}", "lang": "English", "video": "", "path": str(f)})
        voices.append({"src": "v4", "id": f"v4_{ident}", "lang": "English", "video": "", "path": str(v)})
    for src, root in SOURCES.items():
        for vid_dir in sorted((root / "voices").glob("*/*/*")):
            ident, lang, video = vid_dir.parts[-3:]
            fdir = root / "faces" / ident / lang / video
            for w in spread([str(x) for x in vid_dir.glob("*.wav")], VOICES_PER_VIDEO):
                voices.append({"src": src, "id": f"{src}_{ident}", "lang": lang, "video": video, "path": w})
            for j in spread([str(x) for x in fdir.glob("*.jpg")], FACES_PER_VIDEO):
                faces.append({"src": src, "id": f"{src}_{ident}", "lang": lang, "video": video, "path": j})
    return faces, voices


def eval_records():
    m = json.loads((FLAG / "derived/evaluation_phase/manifest.json").read_text())
    rows = m["records"] if isinstance(m, dict) and "records" in m else m
    return rows


def encode(faces, voices, out: Path):
    import torch

    sys.path.insert(0, str(ROOT))
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(0.2, device="cuda:0")
    from exps.v066.audio import extract as extract_ecapa_eres
    from exps.v066.face import FaceEncoder
    from exps.v114.encoder import Encoder
    from exps.v318 import extract as r318

    res = {}
    if faces:
        fe = FaceEncoder(json.loads((ROOT / "exps/v066/config.json").read_text()))
        en = Encoder(json.loads((ROOT / "exps/v114/config.json").read_text()))
        arc, ada = [], []
        for s in range(0, len(faces), 16):
            b = [{"face_path": r["path"]} for r in faces[s : s + 16]]
            a, _ = fe.encode(b)
            d, _, _, _ = en.encode(b)
            arc.append(a)
            ada.append(d)
            if (s // 16) % 100 == 0:
                log(f"人脸 {s}/{len(faces)}")
        res["arcface"] = np.vstack(arc).astype(np.float32)
        res["adaface"] = np.vstack(ada).astype(np.float32)
    if voices:
        vrecs = [{"voice_path": r["path"], "voice_sha256": sha(r["path"])} for r in voices]
        cfg66 = {**json.loads((ROOT / "exps/v066/config.json").read_text()), "time_limit_seconds": 36000}
        (out / "audio").mkdir(exist_ok=True)
        ecapa, eres = extract_ecapa_eres(vrecs, cfg66, out / "audio", "all")
        res["ecapa"], res["eres"] = ecapa.astype(np.float32), eres.astype(np.float32)
        log("ECAPA/ERes 完成")
        model = r318.SpeakerResNet(r318.CFG88).eval().requires_grad_(False)
        st = torch.load(Path(r318.CFG88["weights_root"]) / r318.CFG88["checkpoint_file"], map_location="cpu", weights_only=True)
        tgt = model.state_dict()
        model.load_state_dict({k: v for k, v in st.items() if k in tgt}, strict=True)
        model = model.to("cuda:0")
        out_r = []
        for i, r in enumerate(voices):
            w = r318.center(r318.wave(r["path"]), 6.0)
            with torch.inference_mode():
                e = model(r318.fb(w).unsqueeze(0).to("cuda:0")).cpu().numpy()[0]
            out_r.append(e / np.linalg.norm(e))
            if i % 2000 == 0:
                log(f"ResNet {i}/{len(voices)}")
        res["resnet"] = np.stack(out_r).astype(np.float32)
    for k in res:
        x = res[k]
        res[k] = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/v379/features"))
    ap.add_argument("--part", choices=["train", "smoke", "eval"], default="train")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.part == "eval":
        recs = eval_records()
        faces = [{"pair_id": r["pair_id"], "track": r["track"], "lang": r["language"], "path": str(ROOT / r["face_path"])} for r in recs]
        voices = [{"pair_id": r["pair_id"], "track": r["track"], "lang": r["language"], "path": str(ROOT / r["voice_path"])} for r in recs]
    else:
        faces, voices = build_records()
    if a.part == "smoke":
        faces, voices = faces[:20] + faces[-20:], voices[:20] + voices[-20:]
    log(f"人脸 {len(faces)} 语音 {len(voices)}")
    (out / f"{a.part}_faces.json").write_text(json.dumps(faces))
    (out / f"{a.part}_voices.json").write_text(json.dumps(voices))
    t = time.time()
    fe = encode(faces, [], out)
    np.savez(out / f"{a.part}_faces.npz", **fe)
    log(f"人脸写出 {time.time()-t:.0f}s")
    ve = encode([], voices, out)
    np.savez(out / f"{a.part}_voices.npz", **ve)
    log(f"全部完成 {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
