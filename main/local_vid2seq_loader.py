"""
本地帧 + 标准 vid2seq JSON 数据加载器。
- 扫描 dataset_root 下的子目录（视频ID，例如 000001），要求存在 vid2seq_{id}.json 和帧 JPG。
- 不依赖 progress 文件，不调用外部 API。
- 输出与当前 PyTorch 训练脚本兼容的批字典：frames (B, T, 3, H, W)，captions(list)，timestamps(list)，video_ids。
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class LocalVid2SeqDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        split_ratio: float = 0.8,
        num_frames: int = 100,
        image_size: Tuple[int, int] = (224, 224),
        seed: int = 42,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.split_ratio = split_ratio
        self.num_frames = num_frames
        self.image_size = image_size
        self.seed = seed

        self.video_ids = self._discover_videos()
        self.video_ids = self._split_ids(self.video_ids)

    def _discover_videos(self) -> List[str]:
        ids: List[str] = []
        for item in sorted(self.dataset_root.iterdir()):
            if not item.is_dir():
                continue
            vid = item.name
            json_path = item / f"vid2seq_{vid}.json"
            if json_path.exists():
                ids.append(vid)
        if not ids:
            raise FileNotFoundError(f"在 {self.dataset_root} 未找到任何 vid2seq_*.json")
        return ids

    def _split_ids(self, ids: List[str]) -> List[str]:
        rng = random.Random(self.seed)
        ids_shuffled = ids.copy()
        rng.shuffle(ids_shuffled)
        cut = int(len(ids_shuffled) * self.split_ratio)
        if self.split == "train":
            return ids_shuffled[:cut]
        elif self.split == "val":
            return ids_shuffled[cut:]
        else:
            raise ValueError(f"unknown split: {self.split}")

    def __len__(self) -> int:
        return len(self.video_ids)

    def _load_frames(self, vid: str) -> torch.Tensor:
        video_dir = self.dataset_root / vid
        frames = sorted(video_dir.glob("*.jpg"))
        if not frames:
            raise FileNotFoundError(f"{video_dir} 下未找到帧 JPG")
        idxs = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
        imgs = []
        for i in idxs:
            img = Image.open(frames[i]).convert("RGB")
            img = img.resize(self.image_size, Image.BILINEAR)
            arr = np.array(img).astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))  # CHW
            imgs.append(arr)
        return torch.from_numpy(np.stack(imgs, axis=0))  # (T, 3, H, W)

    def _load_annotations(self, vid: str) -> Dict[str, Any]:
        json_path = self.dataset_root / vid / f"vid2seq_{vid}.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        captions: List[str] = []
        timestamps: List[Tuple[float, float]] = []
        for line in data:
            # 形如 "<12> <14>  raise  ."
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                start_bin = int(parts[0].strip("<>") )
                end_bin = int(parts[1].strip("<>") )
            except ValueError:
                continue
            caption = " ".join(parts[2:-1]).strip()
            captions.append(caption)
            # 与此前脚本一致：bin 转秒假设 0.5s/bin
            timestamps.append((start_bin * 0.5, end_bin * 0.5))
        return {
            "captions": captions,
            "timestamps": timestamps,
            "video_id": vid,
            "num_events": len(captions),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        vid = self.video_ids[idx]
        frames = self._load_frames(vid)  # (T, 3, H, W)
        ann = self._load_annotations(vid)
        return {
            "frames": frames,
            "captions": ann["captions"],
            "timestamps": ann["timestamps"],
            "video_id": ann["video_id"],
            "num_events": ann["num_events"],
        }


def local_vid2seq_collate(max_events: int = 20):
    def _fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(batch)
        frames = torch.stack([b["frames"] for b in batch])  # (B, T, 3, H, W)
        captions_batch: List[List[str]] = []
        timestamps_batch: List[List[Tuple[float, float]]] = []
        video_ids: List[str] = []
        for b in batch:
            caps = b["captions"][:max_events]
            ts = b["timestamps"][:max_events]
            pad_n = max_events - len(caps)
            caps = caps + [""] * pad_n
            ts = ts + [(0.0, 0.0)] * pad_n
            captions_batch.append(caps)
            timestamps_batch.append(ts)
            video_ids.append(b["video_id"])
        return {
            "frames": frames,
            "captions": captions_batch,
            "timestamps": timestamps_batch,
            "video_ids": video_ids,
        }

    return _fn


def get_local_dataloaders(
    dataset_root: str,
    batch_size: int = 4,
    num_workers: int = 0,
    num_frames: int = 100,
    split_ratio: float = 0.8,
    seed: int = 42,
    image_size: Tuple[int, int] = (224, 224),
    max_events: int = 20,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = LocalVid2SeqDataset(
        dataset_root=dataset_root,
        split="train",
        split_ratio=split_ratio,
        num_frames=num_frames,
        image_size=image_size,
        seed=seed,
    )
    val_ds = LocalVid2SeqDataset(
        dataset_root=dataset_root,
        split="val",
        split_ratio=split_ratio,
        num_frames=num_frames,
        image_size=image_size,
        seed=seed,
    )
    collate_fn = local_vid2seq_collate(max_events=max_events)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader
