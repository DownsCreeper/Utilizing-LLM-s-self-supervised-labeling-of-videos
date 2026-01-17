"""
Minimal v2q training script that trains on fine-grained videos only.
- Uses breakfast_progress.json to read fine_grained.completed video IDs.
- Loads per-video JSON labels inside each video folder (e.g., doubao_results_{video_id}.json or spark_labels_* etc.).
- Trains a tiny bag-of-words classifier with synthetic targets (hash-based) for demonstration.
- Saves parameters and training history.
Dependencies: numpy only.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import random
import re
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def load_progress(data_root: Path) -> List[str]:
    progress_file = data_root / "breakfast_progress.json"
    if not progress_file.exists():
        logger.warning("progress file not found: %s", progress_file)
        return []
    try:
        data = json.loads(progress_file.read_text(encoding="utf-8"))
        fine_list = data.get("fine_grained", {}).get("completed", [])
        if not fine_list:
            logger.warning("fine_grained.completed is empty")
        else:
            logger.info("fine-grained videos from progress: %s", fine_list)
        return fine_list
    except Exception as exc:
        logger.error("failed to load progress file: %s", exc)
        return []


def pick_label_file(video_dir: Path, video_id: str) -> Path:
    patterns = [
        f"spark_labels_{video_id}.json",
        f"fine_labels_{video_id}.json",
        f"fine_grained_{video_id}.json",
        f"labels_{video_id}.json",
        f"doubao_results_{video_id}.json",
    ]
    for name in patterns:
        candidate = video_dir / name
        if candidate.exists():
            return candidate
    return None


def normalize_labels(raw) -> List[str]:
    # dict with nested label dict
    if isinstance(raw, dict) and any(isinstance(v, dict) and "label" in v for v in raw.values()):
        out = []
        for frame_name, item in raw.items():
            if isinstance(item, dict):
                text = item.get("label", {}).get("raw_response") or item.get("raw_response") or ""
                if text:
                    out.append(str(text))
        return out
    # dict of frame -> text
    if isinstance(raw, dict):
        return [str(v) for v in raw.values()]
    # list of strings
    if isinstance(raw, list):
        return [str(v) for v in raw]
    return []


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_PATTERN.findall(text)]


def build_dataset(data_root: Path, fine_ids: List[str]) -> Tuple[List[List[str]], List[int], Dict[str, int]]:
    texts: List[List[str]] = []
    targets: List[int] = []
    vocab: Dict[str, int] = {}

    for vid in fine_ids:
        video_dir = data_root / vid
        if not video_dir.is_dir():
            logger.warning("video dir not found: %s", video_dir)
            continue
        label_file = pick_label_file(video_dir, vid)
        if not label_file:
            logger.warning("label file not found in %s", video_dir)
            continue
        try:
            raw = json.loads(label_file.read_text(encoding="utf-8"))
            labels = normalize_labels(raw)
            if not labels:
                logger.warning("no labels in %s", label_file)
                continue
            for lbl in labels:
                tokens = tokenize(lbl)
                if not tokens:
                    continue
                for t in tokens:
                    if t not in vocab:
                        vocab[t] = len(vocab)
                texts.append(tokens)
                targets.append(hash(lbl) % 8)  # synthetic target class
        except Exception as exc:
            logger.error("failed to load labels from %s: %s", label_file, exc)
            continue

    logger.info("dataset size: %d samples, vocab size: %d", len(texts), len(vocab))
    return texts, targets, vocab


def vectorize_batch(batch_tokens: List[List[str]], vocab: Dict[str, int]) -> np.ndarray:
    mat = np.zeros((len(batch_tokens), len(vocab)), dtype=np.float32)
    for i, toks in enumerate(batch_tokens):
        for t in toks:
            j = vocab.get(t)
            if j is not None:
                mat[i, j] += 1.0
    return mat


def softmax_logits(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def train(texts: List[List[str]], targets: List[int], vocab: Dict[str, int], epochs: int, batch_size: int, lr: float, seed: int):
    if not texts:
        raise ValueError("no training samples")
    rng = np.random.default_rng(seed)
    num_classes = 8
    W = rng.normal(scale=0.02, size=(len(vocab), num_classes)).astype(np.float32)
    history = []

    indices = list(range(len(texts)))
    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        total_loss = 0.0
        total_batches = 0
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_tokens = [texts[i] for i in batch_idx]
            batch_targets = np.array([targets[i] for i in batch_idx], dtype=np.int64)
            X = vectorize_batch(batch_tokens, vocab)  # [B, V]
            logits = X @ W  # [B, C]
            probs = softmax_logits(logits)
            # cross-entropy
            y_onehot = np.zeros_like(probs)
            y_onehot[np.arange(len(batch_idx)), batch_targets] = 1.0
            loss = -np.sum(y_onehot * np.log(probs + 1e-8)) / len(batch_idx)
            # gradient
            grad_logits = (probs - y_onehot) / len(batch_idx)
            grad_W = X.T @ grad_logits
            W -= lr * grad_W
            total_loss += loss
            total_batches += 1
        avg_loss = total_loss / max(total_batches, 1)
        history.append({"epoch": epoch, "loss": float(avg_loss)})
        logger.info("epoch %d/%d - loss %.4f", epoch, epochs, avg_loss)
    return W, history


def save_checkpoint(output_dir: Path, W: np.ndarray, vocab: Dict[str, int], history: List[Dict]):
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint_latest.npz"
    np.savez(ckpt_path, W=W, vocab_tokens=np.array(list(vocab.keys())))
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("saved checkpoint to %s", ckpt_path)


def parse_args():
    ap = argparse.ArgumentParser(description="Train v2q on fine-grained breakfast data")
    ap.add_argument("--train_data", required=True, help="Path to dataset root, e.g., dataset/breakfast/breakfast_5")
    ap.add_argument("--num_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--output_dir", default="outputs/v2q")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.train_data)
    fine_ids = load_progress(data_root)
    if not fine_ids:
        logger.error("no fine-grained videos listed; aborting")
        return
    texts, targets, vocab = build_dataset(data_root, fine_ids)
    if not texts:
        logger.error("no training samples; aborting")
        return
    W, history = train(texts, targets, vocab, args.num_epochs, args.batch_size, args.lr, args.seed)
    save_checkpoint(Path(args.output_dir), W, vocab, history)


if __name__ == "__main__":
    main()
