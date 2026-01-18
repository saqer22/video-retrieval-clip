import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from src.models.clip_backbone import ClipBackbone, get_device
from src.models.pooling import mean_pool
from src.seed import set_global_seed
from src.utils.io import ensure_dir, read_json, read_yaml, save_array, write_json
from src.utils.logging import setup_logger


def _sample_frames(frame_paths: List[Path], count: int) -> List[Path]:
    if count <= 0 or not frame_paths:
        return []
    count = min(count, len(frame_paths))
    if count == len(frame_paths):
        return frame_paths
    indices = np.linspace(0, len(frame_paths) - 1, count, dtype=int)
    return [frame_paths[i] for i in indices]


def encode_videos_for_baseline(cfg: Dict, baseline: str, frame_count: int, run_dir: Path, use_head: bool = False) -> Path:
    logger = setup_logger()
    dataset_cfg = cfg.get("dataset", {})
    prep_cfg = cfg.get("preprocess", {})
    model_cfg = cfg.get("model", {})

    device = get_device(model_cfg.get("device", "auto"))
    set_global_seed(cfg.get("seed"))

    backbone = ClipBackbone(model_cfg.get("clip_model", "ViT-B-32"), model_cfg.get("pretrained", "openai"), device)
    
    # Fit text encoder on captions if using simple mode
    root = Path(cfg.get("project_root", "."))
    data_root = root / dataset_cfg.get("root", "data/demo")
    from src.utils.io import read_jsonl
    captions_data = read_jsonl(data_root / "captions.jsonl")
    all_captions = [c.get("caption", "") for c in captions_data]
    backbone.fit_text_encoder(all_captions)
    
    frames_root = root / prep_cfg.get("frames_root", "data/demo/frames")
    splits = read_json(data_root / "splits.json")
    candidate_ids = splits.get("test", splits.get("all", []))

    embeddings: List[np.ndarray] = []
    ids: List[str] = []
    
    target_size = int(prep_cfg.get("target_size", 64))

    logger.info("Encoding %d videos for baseline %s (frame_count=%d)", len(candidate_ids), baseline, frame_count)
    for vid in candidate_ids:
        frame_dir = frames_root / vid
        if not frame_dir.exists():
            logger.warning("Missing frames for %s, skipping", vid)
            continue
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        selected = _sample_frames(frame_paths, frame_count)
        if not selected:
            logger.warning("No frames selected for %s, skipping", vid)
            continue

        # Encode each frame
        frame_feats = []
        for fp in selected:
            feat = backbone.encode_image_from_file(str(fp), target_size=target_size)
            frame_feats.append(feat)
        
        # Mean pool across frames
        feats_array = np.stack(frame_feats)
        pooled = mean_pool(feats_array)
        
        embeddings.append(pooled)
        ids.append(vid)

    ensure_dir(run_dir)
    emb_path = run_dir / f"{baseline}_embeddings.npy"
    ids_path = run_dir / f"{baseline}_ids.json"
    save_array(emb_path, np.stack(embeddings))
    write_json(ids_path, ids)
    logger.info("Saved embeddings to %s", emb_path)
    return emb_path


def main():
    parser = argparse.ArgumentParser(description="Encode videos for retrieval")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline name")
    parser.add_argument("--frame-count", type=int, required=True, help="Frames to sample per video")
    parser.add_argument("--run-dir", type=str, required=True, help="Output directory for embeddings")
    parser.add_argument("--use-head", action="store_true", help="Apply projection head")
    args = parser.parse_args()

    cfg = read_yaml(Path(args.config))
    cfg["project_root"] = str(Path(args.config).parent.parent)
    encode_videos_for_baseline(cfg, args.baseline, args.frame_count, Path(args.run_dir), use_head=args.use_head)


if __name__ == "__main__":
    main()
