from pathlib import Path
from typing import Dict

import cv2

from src.utils.io import ensure_dir, read_json
from src.utils.logging import setup_logger


def preprocess_videos(cfg: Dict, project_root: Path) -> None:
    logger = setup_logger()
    dataset_cfg = cfg.get("dataset", {})
    prep_cfg = cfg.get("preprocess", {})

    root = project_root / dataset_cfg.get("root", "data/demo")
    videos_dir = root / "videos"
    frames_root = project_root / prep_cfg.get("frames_root", "data/demo/frames")
    ensure_dir(frames_root)

    splits = read_json(root / "splits.json")
    target_ids = splits.get("all", [])

    sample_rate = int(prep_cfg.get("frame_sample_rate", 1))
    target_size = int(prep_cfg.get("target_size", 64))
    max_frames = int(prep_cfg.get("max_frames", 16))

    logger.info("Extracting frames for %d videos", len(target_ids))
    for vid in target_ids:
        video_path = videos_dir / f"{vid}.mp4"
        if not video_path.exists():
            logger.warning("Missing video %s, skipping", video_path)
            continue
        out_dir = frames_root / vid
        ensure_dir(out_dir)

        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        saved = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_rate == 0:
                resized = cv2.resize(frame, (target_size, target_size))
                out_path = out_dir / f"frame_{saved:04d}.jpg"
                cv2.imwrite(str(out_path), resized)
                saved += 1
                if saved >= max_frames:
                    break
            frame_idx += 1
        cap.release()
    logger.info("Finished frame extraction")
