"""Extract the published camera views from selected evaluation rollouts.

Run with the MemByPlan virtualenv: .venv/bin/python pal-webpage/scripts/prepare_simulation_media.py
Requires OpenCV, Pillow, and imageio-ffmpeg; no model inference is used.
"""

import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
from PIL import Image

SITE = Path(__file__).resolve().parents[1]
ROOT = SITE.parent
EVAL = ROOT / "runs/eval/main_results/checkpoint-10790_all_env_20260819-135450"
FIGURES = SITE / "assets/figures/simulation"
VIDEOS = SITE / "assets/media/simulation"
EPISODES = [
    ("PickXtimes", 26, "medium", "pickxtimes_sim", {"poster": 142}),
    ("VideoPlaceOrder", 21, "easy", "video_place_order_sim", {"poster": 487}),
    ("VideoUnmask", 2, "medium", "video_unmask_sim", {
        "poster": 99, "observe": 0, "remember": 32, "plan": 67, "act": 198,
    }),
]


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    manifest = []
    for task, episode, difficulty, name, frames in EPISODES:
        source = EVAL / task / "shards/worker3_gpu3" / task / f"ep{episode}" / (
            f"{task}_ep{episode}_success_{difficulty}_vlm_vla.mp4"
        )
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {source}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width != 512 or height < 256:
            raise ValueError(f"Unexpected camera layout: {width}x{height}")
        record = {
            "task": task, "episode": episode, "difficulty": difficulty,
            "source": str(source.relative_to(ROOT)),
            "crop": {"x": 0, "y": height - 256, "width": 512, "height": 256},
            "fps": capture.get(cv2.CAP_PROP_FPS), "screenshots": {},
        }
        for label, frame_index in frames.items():
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode {source}, frame {frame_index}")
            scene = frame[-256:, :]
            output = FIGURES / f"{name}_{label}.jpg"
            Image.fromarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)).save(
                output, quality=94, subsampling=0,
            )
            record["screenshots"][label] = {
                "frame": frame_index, "path": str(output.relative_to(SITE)),
            }
            if task == "VideoUnmask" and label in ("observe", "remember"):
                overview = FIGURES / f"video_unmask_{label}_overview.jpg"
                Image.fromarray(cv2.cvtColor(scene[:, :256], cv2.COLOR_BGR2RGB)).save(
                    overview, quality=94, subsampling=0,
                )
        capture.release()
        output = VIDEOS / f"{name}.mp4"
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
            "-y", "-i", str(source), "-vf", "crop=512:256:0:ih-256",
            "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ], check=True)
        record["video"] = str(output.relative_to(SITE))
        manifest.append(record)
        print(f"Prepared {task} ep{episode}: {len(frames)} screenshots and camera-only video")
    (VIDEOS / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name in ("patternlock", "video_place_order", "video_unmask"):
        capture = cv2.VideoCapture(str(SITE / "assets/media" / f"{name}.mp4"))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open real-world video: {name}")
        frame_index = min(20, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot extract real-world poster: {name}")
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(
            SITE / "assets/media" / f"{name}_poster.jpg", quality=94, subsampling=0,
        )
        capture.release()
    for source in (SITE / "assets/media").rglob("*.mp4"):
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
            "-y", "-i", str(source), "-an", "-c:v", "libvpx-vp9", "-crf", "32",
            "-b:v", "0", "-row-mt", "1", "-deadline", "good", "-cpu-used", "4",
            str(source.with_suffix(".webm")),
        ], check=True)
        print(f"Prepared WebM alternative: {source.stem}")


if __name__ == "__main__":
    main()
