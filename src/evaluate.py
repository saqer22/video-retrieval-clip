import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.models.clip_backbone import ClipBackbone, get_device
from src.seed import set_global_seed
from src.utils.io import load_array, read_json, read_jsonl, read_yaml, write_json, ensure_dir
from src.utils.logging import setup_logger
from src.utils.metrics import compute_metrics


def _load_embeddings(run_dir: Path, baseline: str) -> Tuple[np.ndarray, List[str]]:
    emb = load_array(run_dir / f"{baseline}_embeddings.npy")
    ids = read_json(run_dir / f"{baseline}_ids.json")
    return emb, ids


def _compute_ranks(similarities: np.ndarray, gt_index: int) -> int:
    order = np.argsort(-similarities)
    rank = int(np.where(order == gt_index)[0][0])
    return rank


def evaluate_baseline(cfg: Dict, baseline_cfg: Dict, run_dir: Path, backbone=None) -> Tuple[Dict[str, float], Dict[str, float]]:
    logger = setup_logger()
    dataset_cfg = cfg.get("dataset", {})
    data_root = Path(cfg.get("project_root", ".")) / dataset_cfg.get("root", "data/demo")
    captions = read_jsonl(data_root / "captions.jsonl")
    split_ids = set(read_json(data_root / "splits.json").get("test", []))
    test_rows = [c for c in captions if c.get("video_id") in split_ids]
    video_lookup = {vid: idx for idx, vid in enumerate(sorted(list(split_ids)))}

    device = get_device(cfg.get("model", {}).get("device", "auto"))
    set_global_seed(cfg.get("seed"))

    baseline_name = baseline_cfg.get("name")
    logger.info("Evaluating baseline %s on %d queries", baseline_name, len(test_rows))

    if baseline_name == "random":
        ranks = []
        rng = np.random.default_rng(cfg.get("seed", 1234))
        for row in test_rows:
            candidate_count = len(split_ids)
            perm = rng.permutation(candidate_count)
            gt = video_lookup[row["video_id"]]
            rank = int(np.where(perm == gt)[0][0])
            ranks.append(rank)
        metrics = compute_metrics(ranks, cfg.get("eval", {}).get("topk", [1, 5, 10]))
        retrieval_time = {"text_and_search_ms": 0.0, "search_only_ms": 0.0}
        return metrics, retrieval_time

    embeddings, ids = _load_embeddings(run_dir, baseline_name)
    id_to_index = {vid: idx for idx, vid in enumerate(ids)}

    if backbone is None:
        backbone = ClipBackbone(cfg["model"]["clip_model"], cfg["model"]["pretrained"], device)
        # Fit text encoder on all captions
        all_captions = [c.get("caption", "") for c in captions]
        backbone.fit_text_encoder(all_captions)

    ranks: List[int] = []
    text_times: List[float] = []
    search_times: List[float] = []

    for row in test_rows:
        caption = row.get("caption", "")
        vid = row.get("video_id")
        if vid not in id_to_index:
            continue
        start_text = time.perf_counter()
        text_feat = backbone.encode_texts([caption])[0]
        text_time = time.perf_counter() - start_text

        start_search = time.perf_counter()
        sims = np.dot(embeddings, text_feat)
        rank = _compute_ranks(sims, id_to_index[vid])
        search_time = time.perf_counter() - start_search

        ranks.append(rank)
        text_times.append(text_time)
        search_times.append(search_time)

    metrics = compute_metrics(ranks, cfg.get("eval", {}).get("topk", [1, 5, 10]))
    if text_times and search_times:
        retrieval_time = {
            "text_and_search_ms": float(np.mean(np.array(text_times) + np.array(search_times)) * 1000),
            "search_only_ms": float(np.mean(search_times) * 1000),
        }
    else:
        retrieval_time = {"text_and_search_ms": 0.0, "search_only_ms": 0.0}
    return metrics, retrieval_time


def evaluate_all(cfg: Dict, run_dir: Path) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    logger = setup_logger()
    metrics_all: Dict[str, Dict] = {}
    times_all: Dict[str, Dict] = {}
    
    # Create backbone once and reuse
    device = get_device(cfg.get("model", {}).get("device", "auto"))
    backbone = ClipBackbone(cfg["model"]["clip_model"], cfg["model"]["pretrained"], device)
    
    # Fit text encoder
    data_root = Path(cfg.get("project_root", ".")) / cfg.get("dataset", {}).get("root", "data/demo")
    captions = read_jsonl(data_root / "captions.jsonl")
    all_captions = [c.get("caption", "") for c in captions]
    backbone.fit_text_encoder(all_captions)
    
    for baseline_cfg in cfg.get("eval", {}).get("baselines", []):
        metrics, t = evaluate_baseline(cfg, baseline_cfg, run_dir, backbone=backbone)
        metrics_all[baseline_cfg["name"]] = metrics
        times_all[baseline_cfg["name"]] = t
    return metrics_all, times_all


def save_results_csv(metrics: Dict[str, Dict], times: Dict[str, Dict], out_path: Path) -> None:
    rows = []
    for baseline, vals in metrics.items():
        row = {"baseline": baseline}
        row.update(vals)
        row.update(times.get(baseline, {}))
        rows.append(row)
    df = pd.DataFrame(rows)
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    cfg = read_yaml(Path(args.config))
    cfg["project_root"] = str(Path(args.config).parent.parent)
    run_dir = Path(args.run_dir)

    metrics, times = evaluate_all(cfg, run_dir)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "retrieval_time.json", times)
    save_results_csv(metrics, times, run_dir / "tables" / "results.csv")


if __name__ == "__main__":
    main()
