"""
Caption-level vocabulary utilities for vid2seq training/prediction.
- Builds a vocabulary over full caption strings (not wordpiece).
- Provides encode/decode helpers.
- Deterministic ordering: sort by frequency desc, then lexicographic.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class CaptionVocab:
    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token: List[str] = [""] * len(token_to_id)
        for tok, idx in token_to_id.items():
            self.id_to_token[idx] = tok
        self.pad_token_id = self.token_to_id[PAD_TOKEN]
        self.unk_token_id = self.token_to_id[UNK_TOKEN]

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str) -> int:
        return self.token_to_id.get(text, self.unk_token_id)

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self.id_to_token):
            return self.id_to_token[idx]
        return UNK_TOKEN

    def to_dict(self) -> Dict[str, int]:
        return self.token_to_id

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "CaptionVocab":
        return cls(token_to_id=data)


def _discover_train_ids(dataset_root: Path, split_ratio: float = 0.8, seed: int = 42) -> List[str]:
    ids: List[str] = []
    for item in sorted(dataset_root.iterdir()):
        if not item.is_dir():
            continue
        vid = item.name
        if (item / f"vid2seq_{vid}.json").exists():
            ids.append(vid)
    if not ids:
        raise FileNotFoundError(f"No vid2seq_*.json under {dataset_root}")
    rng = random.Random(seed)
    rng.shuffle(ids)
    cut = int(len(ids) * split_ratio)
    return ids[:cut]  # train split


def _load_captions(json_path: Path) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    captions: List[str] = []
    for line in data:
        parts = str(line).strip().split()
        if len(parts) < 4:
            continue
        caption = " ".join(parts[2:-1]).strip()
        if caption:
            captions.append(caption)
    return captions


def build_vocab_from_dataset(dataset_root: str, split_ratio: float = 0.8, seed: int = 42,
                             min_freq: int = 1, max_size: int | None = None) -> CaptionVocab:
    root = Path(dataset_root)
    train_ids = _discover_train_ids(root, split_ratio=split_ratio, seed=seed)
    counter: Counter[str] = Counter()
    for vid in train_ids:
        json_path = root / vid / f"vid2seq_{vid}.json"
        counter.update(_load_captions(json_path))

    # Always include pad/unk
    tokens: List[Tuple[str, int]] = counter.most_common()
    if min_freq > 1:
        tokens = [(t, c) for t, c in tokens if c >= min_freq]
    if max_size is not None:
        tokens = tokens[:max_size]

    # deterministic secondary sort by token text
    tokens = sorted(tokens, key=lambda x: (-x[1], x[0]))

    token_to_id: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    next_id = 2
    for tok, _ in tokens:
        if tok in token_to_id:
            continue
        token_to_id[tok] = next_id
        next_id += 1

    return CaptionVocab(token_to_id)


def save_vocab(vocab: CaptionVocab, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab.to_dict(), f, ensure_ascii=False, indent=2)


def load_vocab(path: Path) -> CaptionVocab:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CaptionVocab.from_dict(data)
