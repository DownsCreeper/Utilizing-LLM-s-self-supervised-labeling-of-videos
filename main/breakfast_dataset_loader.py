"""
Minimal compatibility wrapper so that breakfast_vid2seq_train.py can import
get_breakfast_dataloaders. This reuses the local JSON+frames loader.
"""
from typing import Tuple
from torch.utils.data import DataLoader

from main.local_vid2seq_loader import get_local_dataloaders


def get_breakfast_dataloaders(
    dataset_root: str,
    batch_size: int = 8,
    num_workers: int = 0,
    num_frames: int = 100,
    split_ratio: float = 0.8,
    use_completed_only: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    # use_completed_only is ignored; kept for API compatibility
    return get_local_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
        num_frames=num_frames,
        split_ratio=split_ratio,
    )
