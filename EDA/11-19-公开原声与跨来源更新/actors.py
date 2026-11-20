"""从两次独立拍摄的原件中提取固定帧的人脸和另一拍的说话声音。"""

from __future__ import annotations

import argparse
import itertools
import logging
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
from frozen_frontend import faces, generic, speakers
from media_common import CFG, HERE, PC, REF, ROOT, SIDES, read, setup, sha, write

from exps.v066.face import FaceEncoder

SPEC = read(HERE / "演员验证配置.json")
LOG = logging.getLogger(__name__)


def select() -> tuple[list[dict], list[dict]]:
    clips = {}
    for clip in read(ROOT / SPEC["manifest"]):
        codes = Path(clip["member"]).stem.split("-")
        medium, task, emotion, level, text, rep, actor = codes
        if [medium, task] != SPEC["audio_visual_speech"] or level != SPEC["intensity"]:
            continue
        if emotion not in SPEC["emotions"] or text not in SPEC["statements"]:
            continue
        key = (actor, emotion, text, rep)
        assert key not in clips and clip["zip_member_crc_verified"]
        assert clip["actor"] == int(actor) and clip["identity"] == "RAVDESS_" + actor
        clips[key] = clip
    actors = {k[0] for k in clips}
    assert len(actors) == 24
    rows = []
    for actor, emotion, sentence in itertools.product(
        sorted(actors), SPEC["emotions"], SPEC["statements"]
    ):
        row = {"id": "RAVDESS_" + actor, "emotion": emotion, "sentence": sentence}
        for side in SIDES:
            a = clips[actor, emotion, sentence, SPEC[side + "_repetition"]]
            gender = {"Male": "m", "Female": "f"}[a["gender"]]
            assert row.get("gender", gender) == gender
            row.update(
                {
                    side: a["video_path"],
                    side + "_sha256": a["video_sha256"],
                    "gender": gender,
                }
            )
        rows.append(row)
    duplicates = [r for r in rows if r["face_sha256"] == r["voice_sha256"]]
    for row in duplicates:
        assert all(sha(ROOT / row[s]) == row[s + "_sha256"] for s in SIDES)
    dropped = {r["emotion"] for r in duplicates}
    rows = [r for r in rows if r["emotion"] not in dropped]
    assert len(SPEC["emotions"]) - len(dropped) >= 2
    assert len({r[s + "_sha256"] for r in rows for s in SIDES}) == len(rows) * 2
    assert len({r["id"] for r in rows if r["gender"] == "f"}) == 12
    return rows, duplicates


def extract_pair(row: dict, dst: Path, detector: FaceEncoder | None) -> dict:
    for side in SIDES:
        assert sha(ROOT / row[side]) == row[side + "_sha256"]
    audio = dst.with_suffix(".wav")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(ROOT / row["voice"]),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SPEC["audio_rate"]),
            "-c:a",
            "pcm_s16le",
            str(audio),
        ],
        check=True,
    )
    wav, rate = sf.read(audio, dtype="int16")
    assert (
        rate == SPEC["audio_rate"]
        and wav.ndim == 1
        and len(wav) > rate
        and np.max(abs(wav.astype(int))) > 0
    )
    cap = cv2.VideoCapture(
        str(ROOT / row["face"]), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    assert cap.isOpened()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert total > 1
    try:
        for fraction in SPEC["frame_fractions"]:
            frame_index = round((total - 1) * fraction)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, pix = cap.read()
            if not ok:
                continue
            if detector is None:
                height, width = pix.shape[:2]
                size = min(height, width)
                x0, y0 = (width - size) // 2, (height - size) // 2
                x1, y1 = x0 + size, y0 + size
            else:
                boxes, _ = detector.detector.detect(pix, max_num=0)
                if len(boxes) != 1 or boxes[0, 4] < CFG["det_threshold"]:
                    continue
                left, top, right, bottom, _ = boxes[0]
                if min(right - left, bottom - top) < CFG["minimum_face_pixels"]:
                    continue
                center = np.array([(left + right) / 2, (top + bottom) / 2])
                pad = CFG["face_padding"] * max(right - left, bottom - top) / 2
                x0, y0 = np.maximum(np.rint(center - pad).astype(int), 0)
                x1, y1 = np.minimum(np.rint(center + pad).astype(int), pix.shape[1::-1])
            face = dst.with_suffix(".png")
            assert cv2.imwrite(str(face), pix[y0:y1, x0:x1])
            return {
                "frame": frame_index,
                "frames": total,
                "fraction": fraction,
                "box": list(map(int, [x0, y0, x1, y1])),
                "samples": len(wav),
                "crop_sha256": sha(face),
                "wav_sha256": sha(audio),
            }
    finally:
        cap.release()
    raise ValueError("给定帧均未检测到合格单脸 " + row["face"])


@torch.inference_mode()
def encode_media(images: list[Path], waves: list[Path], out: Path) -> tuple[dict, dict]:
    controls = [Path(read(ROOT / REF["metadata"][s][1])[0]["path"]) for s in SIDES]
    images = images + [controls[0]]
    waves = waves + [controls[1]]
    blocks = faces(images)
    with tempfile.TemporaryDirectory(dir=out) as temp:
        blocks.update(speakers(waves, Path(temp)))
    blocks.update(generic(waves))
    fb = REF["feature_blocks"]
    groups = [[fb["face"][1], fb["face_additional"][1]], fb["voice_extra"]]
    errors = {}
    for filename, key in groups[0] + groups[1]:
        with np.load(ROOT / filename, allow_pickle=False) as z:
            ref = z[key][0]
        x = blocks[key][-1]
        err = float(np.max(abs(x - ref)))
        cos = float((x @ ref) / (np.linalg.norm(x) * np.linalg.norm(ref)))
        assert cos > 0.9999 and err < 4e-4
        errors[key] = {"max_abs": err, "cosine": cos}
    data = {
        s: np.column_stack([blocks[k][:-1] for _, k in group])
        for s, group in zip(SIDES, groups, strict=True)
    }
    assert all(np.isfinite(x).all() for x in data.values())
    return data, errors


@torch.inference_mode()
def run(out: Path) -> None:
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    setup(out, "演员编码.log")
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, duplicates = select()
    lock = out / "解码前验证设计.json"
    write(
        lock,
        {
            "config": SPEC,
            "manifest_sha256": sha(ROOT / SPEC["manifest"]),
            "record_sha256": sha(ROOT / SPEC["record"]),
            "duplicates": duplicates,
            "rows": rows,
        },
    )
    detector = FaceEncoder(
        {
            **read(ROOT / "exps/v066/config.json"),
            "cpu_threads": PC["resource_cpu_threads"],
        }
    )
    folder = out / "逐拍媒体"
    folder.mkdir()
    stems = [folder / ("_".join([r["id"], r["emotion"], r["sentence"]])) for r in rows]
    decoded = []
    for i, (row, p) in enumerate(zip(rows, stems, strict=True), 1):
        decoded.append(extract_pair(row, p, detector))
        if i % 48 == 0:
            LOG.info("已核对并解码 %d / %d 对原件", i, len(rows))
    del detector
    data, errors = encode_media(
        [p.with_suffix(".png") for p in stems],
        [p.with_suffix(".wav") for p in stems],
        out,
    )
    dest = out / "两拍原始特征.npz"
    np.savez_compressed(dest, face=data["face"], voice=data["voice"])
    write(
        out / "演员特征核验.json",
        {
            "lock_sha256": sha(lock),
            "decoded": decoded,
            "controls": errors,
            "features_sha256": sha(dest),
            "seconds": time.monotonic() - start,
        },
    )
    LOG.info("完成 %d 对逐拍编码和 %d 路外部锚点复核", len(rows), len(errors))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    run(ap.parse_args().out)
