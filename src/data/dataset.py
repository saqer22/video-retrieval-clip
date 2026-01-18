from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from src.utils.io import read_json, read_jsonl


class CaptionDataset(Dataset):
    def __init__(self, root: Path, split: str = "train") -> None:
        captions_path = root / "captions.jsonl"
        splits_path = root / "splits.json"
        captions = read_jsonl(captions_path)
        split_ids = read_json(splits_path).get(split, [])
        split_set = set(split_ids)
        self.samples = [c for c in captions if c.get("video_id") in split_set]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        return {
            "caption": item.get("caption", ""),
            "video_id": item.get("video_id", ""),
        }


def load_captions_by_split(root: Path, split: str) -> List[Dict]:
    captions = read_jsonl(root / "captions.jsonl")
    split_ids = set(read_json(root / "splits.json").get(split, []))
    return [c for c in captions if c.get("video_id") in split_ids]
