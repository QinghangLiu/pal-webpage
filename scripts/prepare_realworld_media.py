"""Publish four complete real-world demos at one uniform source playback speed.

Run from MemByPlan: .venv/bin/python pal-webpage/scripts/prepare_realworld_media.py
Requires OpenCV and imageio-ffmpeg. The evaluator's annotated review videos
provide the shared 8 fps playback timeline; raw observations provide sharper
camera images where available. No idle intervals or action segments are cut.
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

SITE = Path(__file__).resolve().parents[1]
ROOT = SITE.parent
SPEED = 4
REVIEW_FPS = 8
DEMOS = [
    ("PatternLock", "demopl", "PickXtimes", "patternlock", False),
    ("PickXtimes", "demopxt", "PickXtimes", "pickxtimes", True),
    ("VideoPlaceOrder", "demovpo", "VideoPlaceOrder", "video_place_order", True),
    ("VideoUnmask", "demovu", "PickXtimes", "video_unmask", True),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / (
        "data/old_pxt/eval_bundle_ours_framesamp-modul_memer_20260912"
    ))
    args = parser.parse_args()
    output_dir = SITE / "assets/media/realworld"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for task, demo, directory, name, use_raw in DEMOS:
        episode = args.bundle / "runs/backup" / demo / directory / "ep0"
        review = episode / "annotated_execution_event_review.mp4"
        source = episode / "raw_observations/base.mp4" if use_raw else review
        capture = cv2.VideoCapture(str(source))
        reference = cv2.VideoCapture(str(review))
        if not capture.isOpened() or not reference.isOpened():
            raise RuntimeError(f"Cannot open demo {demo}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if reference.get(cv2.CAP_PROP_FPS) != REVIEW_FPS:
            raise ValueError(f"Unexpected review timebase: {demo}")
        if int(reference.get(cv2.CAP_PROP_FRAME_COUNT)) != frames:
            raise ValueError(f"Raw/review frames are not aligned: {demo}")
        reference.release()
        # Review: remove both text columns. Raw: retain the original table crop.
        crop = (300, 0, 720, 720) if use_raw else (360, 0, 384, 384)
        x, y, width, height = crop
        if x + width > capture.get(cv2.CAP_PROP_FRAME_WIDTH):
            raise ValueError(f"Invalid camera crop: {demo}")
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"Cannot decode poster: {demo}")
        stem = name + "_demo_4x"
        poster = output_dir / (stem + "_poster.jpg")
        scene = frame[y:y + height, x:x + width]
        size = 640 if use_raw else 384
        scene = cv2.resize(scene, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(poster), scene, [cv2.IMWRITE_JPEG_QUALITY, 94])
        output = output_dir / (stem + ".mp4")
        # All cameras use the same review timeline, even when raw files have a
        # different container fps. Speed is baked in, including fullscreen/no-JS.
        pts_scale = fps / REVIEW_FPS / SPEED
        filters = f"crop={width}:{height}:{x}:{y},scale={size}:{size},setpts={pts_scale}*(PTS-STARTPTS),fps=30"
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-vf", filters, "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ], check=True)
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(output),
            "-an", "-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0",
            "-row-mt", "1", "-deadline", "good", "-cpu-used", "4",
            str(output.with_suffix(".webm")),
        ], check=True)
        records.append({
            "task": task, "demo": demo, "source": str(source.relative_to(ROOT)),
            "review": str(review.relative_to(ROOT)), "source_frames": frames,
            "source_container_fps": fps, "reference_playback_fps": REVIEW_FPS,
            "reference_duration_seconds": frames / REVIEW_FPS,
            "speed": SPEED, "speed_reference": "annotated review playback; not wall-clock execution",
            "crop": {"x": x, "y": y, "width": width, "height": height},
            "video": str(output.relative_to(SITE)), "poster": str(poster.relative_to(SITE)),
            "outcome": "Task completion visually reviewed; evaluator flag is unknown for these demo captures.",
        })
        print(f"Prepared {task} / {demo}: {frames / REVIEW_FPS:.2f}s → {frames / REVIEW_FPS / SPEED:.2f}s", flush=True)
    (output_dir / "sources.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
