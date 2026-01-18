import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.models.clip_backbone import ClipBackbone, get_device
from src.seed import set_global_seed
from src.utils.io import load_array, read_json, read_jsonl, read_yaml


def load_embeddings(run_dir: Path, baseline: str) -> Tuple[np.ndarray, List[str]]:
    emb = load_array(run_dir / f"{baseline}_embeddings.npy")
    ids = read_json(run_dir / f"{baseline}_ids.json")
    return emb, ids


def retrieve(query: str, cfg: Dict, baseline: str, run_dir: Path, topk: int = 5, backbone=None) -> List[Tuple[str, float]]:
    if backbone is None:
        device = get_device(cfg.get("model", {}).get("device", "auto"))
        set_global_seed(cfg.get("seed"))
        backbone = ClipBackbone(cfg["model"]["clip_model"], cfg["model"]["pretrained"], device)
        
        # Fit text encoder
        root = Path(cfg.get("project_root", "."))
        data_root = root / cfg.get("dataset", {}).get("root", "data/demo")
        captions_data = read_jsonl(data_root / "captions.jsonl")
        all_captions = [c.get("caption", "") for c in captions_data]
        backbone.fit_text_encoder(all_captions)

    emb, ids = load_embeddings(run_dir, baseline)
    text_feat = backbone.encode_texts([query])[0]
    sims = np.dot(emb, text_feat)
    order = np.argsort(-sims)
    results = [(ids[i], float(sims[i])) for i in order[:topk]]
    return results


def main():
    parser = argparse.ArgumentParser(description="Retrieve videos for a query")
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    cfg = read_yaml(Path(args.config))
    cfg["project_root"] = str(Path(args.config).parent.parent)
    results = retrieve(args.query, cfg, args.baseline, Path(args.run_dir), topk=args.topk)
    for vid, score in results:
        print(f"{vid}\t{score:.3f}")


if __name__ == "__main__":
    main()
