import itertools
import random
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from src.utils.io import ensure_dir, write_json, write_jsonl
from src.utils.logging import setup_logger


SHAPES = ["square", "circle", "triangle"]
COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
}
DIRECTIONS = ["left", "right", "up", "down"]


def _draw_shape(frame: np.ndarray, shape: str, color: tuple, center: tuple, size: int) -> None:
    if shape == "square":
        top_left = (center[0] - size, center[1] - size)
        bottom_right = (center[0] + size, center[1] + size)
        cv2.rectangle(frame, top_left, bottom_right, color, thickness=-1)
    elif shape == "circle":
        cv2.circle(frame, center, size, color, thickness=-1)
    else:  # triangle
        pts = np.array(
            [
                [center[0], center[1] - size],
                [center[0] - size, center[1] + size],
                [center[0] + size, center[1] + size],
            ],
            np.int32,
        )
        cv2.fillConvexPoly(frame, pts, color)


def _trajectory(frame_size: int, direction: str, step: int, total_steps: int) -> tuple:
    margin = frame_size // 5
    span = frame_size - 2 * margin
    progress = step / max(total_steps - 1, 1)
    if direction == "left":
        x = int(margin + span * (1 - progress))
        y = frame_size // 2
    elif direction == "right":
        x = int(margin + span * progress)
        y = frame_size // 2
    elif direction == "up":
        x = frame_size // 2
        y = int(margin + span * (1 - progress))
    else:  # down
        x = frame_size // 2
        y = int(margin + span * progress)
    return x, y


def _make_caption(color: str, shape: str, direction: str) -> List[str]:
    templates = [
        "a {color} {shape} moves {direction}",
        "the {color} {shape} travels {direction}",
        "a tiny {color} {shape} drifting {direction}",
    ]
    return [t.format(color=color, shape=shape, direction=direction) for t in templates]


def generate_demo(cfg: Dict, project_root: Path, force: bool = False) -> None:
    logger = setup_logger()
    dataset_cfg = cfg.get("dataset", {})
    root = project_root / dataset_cfg.get("root", "data/demo")
    videos_dir = root / "videos"
    ensure_dir(videos_dir)
    captions_path = root / "captions.jsonl"
    splits_path = root / "splits.json"
    flag_path = root / "generated.flag"

    if flag_path.exists() and not force:
        logger.info("Demo dataset already exists, skipping generation.")
        return

    num_videos = int(dataset_cfg.get("num_videos", 20))
    frame_size = int(dataset_cfg.get("frame_size", 64))
    fps = int(dataset_cfg.get("fps", 4))
    duration = float(dataset_cfg.get("duration_seconds", 2))
    captions_per_video = int(dataset_cfg.get("captions_per_video", 3))
    total_frames = max(2, int(duration * fps))

    random.seed(cfg.get("seed", 1234))
    np.random.seed(cfg.get("seed", 1234))

    logger.info("Generating synthetic demo videos: %d clips", num_videos)
    caption_rows: List[Dict] = []
    video_ids: List[str] = []

    shape_cycle = itertools.cycle(SHAPES)
    color_cycle = itertools.cycle(COLORS.keys())
    direction_cycle = itertools.cycle(DIRECTIONS)

    for idx in range(num_videos):
        video_id = f"video_{idx:03d}"
        video_ids.append(video_id)
        shape = next(shape_cycle)
        color_name = next(color_cycle)
        color_bgr = COLORS[color_name]
        direction = next(direction_cycle)

        writer = cv2.VideoWriter(
            str(videos_dir / f"{video_id}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_size, frame_size),
        )
        for step in range(total_frames):
            frame = np.full((frame_size, frame_size, 3), 255, dtype=np.uint8)
            center = _trajectory(frame_size, direction, step, total_frames)
            _draw_shape(frame, shape, color_bgr, center, size=frame_size // 6)
            writer.write(frame)
        writer.release()

        captions = _make_caption(color_name, shape, direction)
        for cap in captions[:captions_per_video]:
            caption_rows.append({"video_id": video_id, "caption": cap})

    # splits
    train_n = int(dataset_cfg.get("train_split", num_videos * 0.6))
    val_n = int(dataset_cfg.get("val_split", max(1, num_videos * 0.2)))
    test_n = int(dataset_cfg.get("test_split", max(1, num_videos - train_n - val_n)))
    train_ids = video_ids[:train_n]
    val_ids = video_ids[train_n : train_n + val_n]
    test_ids = video_ids[train_n + val_n : train_n + val_n + test_n]

    write_jsonl(captions_path, caption_rows)
    write_json(
        splits_path,
        {"train": train_ids, "val": val_ids, "test": test_ids, "all": video_ids},
    )
    flag_path.touch()
    logger.info("Demo dataset generated at %s", root)
