"""v380 阶段0：从 VoxCeleb2 视频 zip 中按固定规则抽取人脸帧与语音（不整包解压）。

每个身份最多 VIDEOS_PER_ID 个 YouTube 视频（排序后等距选取），每个视频取排序后的第一个片段：
- 语音：ffmpeg 转 16 kHz 单声道 wav；
- 人脸：片段 25%/75% 时刻各一帧 jpg（VoxCeleb2 片段本身是人脸跟踪裁剪）。
纯确定性规则，不看任何 FLAG 评测数据。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

VIDEOS_PER_ID = 30
FRAME_POS = (0.25, 0.75)


def spread(items, k):
    items = sorted(items)
    if len(items) <= k:
        return items
    return [items[i] for i in np.linspace(0, len(items) - 1, k).round().astype(int)]


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    zf = zipfile.ZipFile(a.zip)
    clips = defaultdict(lambda: defaultdict(list))
    for n in zf.namelist():
        if n.endswith(".mp4"):
            parts = Path(n).parts
            ident, video = parts[-3], parts[-2]
            clips[ident][video].append(n)
    print("身份", len(clips), "视频", sum(len(v) for v in clips.values()), flush=True)
    faces, voices = [], []
    with tempfile.TemporaryDirectory() as td:
        for i, ident in enumerate(sorted(clips)):
            for video in spread(list(clips[ident]), VIDEOS_PER_ID):
                member = sorted(clips[ident][video])[0]
                tmp = Path(td) / "c.mp4"
                tmp.write_bytes(zf.read(member))
                stem = f"{video}_{Path(member).stem}"
                vdir, fdir = out / "voices" / ident, out / "faces" / ident
                vdir.mkdir(parents=True, exist_ok=True)
                fdir.mkdir(parents=True, exist_ok=True)
                wav = vdir / f"{stem}.wav"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
                voices.append({"src": a.tag, "id": f"{a.tag}_{ident}", "lang": "unknown", "video": video, "path": str(wav.resolve())})
                d = duration(tmp)
                for k, pos in enumerate(FRAME_POS):
                    jpg = fdir / f"{stem}_{k}.jpg"
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{d * pos:.3f}", "-i", str(tmp), "-frames:v", "1", "-q:v", "2", str(jpg)], check=True)
                    faces.append({"src": a.tag, "id": f"{a.tag}_{ident}", "lang": "unknown", "video": video, "path": str(jpg.resolve())})
            if i % 10 == 0:
                print(f"{i}/{len(clips)} 脸 {len(faces)} 声 {len(voices)}", flush=True)
    (out / "faces.json").write_text(json.dumps(faces))
    (out / "voices.json").write_text(json.dumps(voices))
    print("完成 脸", len(faces), "声", len(voices), flush=True)


if __name__ == "__main__":
    main()
