"""
Autoregressive prediction on coarse-labeled dataset using trained Vid2Seq (PyTorch path).
- Loads checkpoint saved by Trainer (breakfast_vid2seq_train.py)
- Samples a subset of videos (--sample_ratio or --sample_num)
- Extracts CLIP ViT-L/14 features per video (must have CLIP installed)
- Greedy autoregressive decode for captions; timestamps from event head
- Saves to vid2seq_{video_id}_predict.json under each video folder
  (format similar to input: "<start> <end>  token_{id}  .")

Note: The toy model hashes captions; decoding yields token ids (token_{k}) rather than real text.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "main"))

from main.breakfast_vid2seq_train import FeatureExtractor, SimpleVid2SeqModel
from main.caption_vocab import CaptionVocab
from main.local_vid2seq_loader import LocalVid2SeqDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_checkpoint(ckpt_path: Path, device: str) -> Tuple[dict, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    vocab_dict = ckpt.get("vocab")
    if vocab_dict is None:
        raise RuntimeError("checkpoint does not contain vocab; retrain with new pipeline")
    logger.info("checkpoint loaded from %s", ckpt_path)
    return ckpt, vocab_dict


def greedy_decode(model: SimpleVid2SeqModel, features: torch.Tensor, max_len: int) -> torch.Tensor:
    """Greedy autoregressive decode; returns token ids (B, L).
    Recomputes model forward each step to stay consistent with feature input.
    """
    device = features.device
    B = features.shape[0]
    bos = torch.full((B, 1), model.bos_token_id, dtype=torch.long, device=device)
    seq = bos
    for _ in range(max_len - 1):
        logits = model(features, label_hints=seq)["caption_logits"]  # (B, L, vocab)
        next_logits = logits[:, -1, :]
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        seq = torch.cat([seq, next_token], dim=1)
        if torch.all(next_token.squeeze(1) == model.eos_token_id):
            break
    return seq


def tokens_to_lines(tokens: torch.Tensor, event_logits: torch.Tensor, model: SimpleVid2SeqModel,
                   vocab: CaptionVocab) -> List[str]:
    """Convert token ids and timestamps to vid2seq-like lines with decoded captions."""
    lines: List[str] = []
    B, E, _ = event_logits.shape
    pad = model.pad_token_id
    mask_id = model.mask_token_id
    bos = model.bos_token_id
    eos = model.eos_token_id

    def ids_to_words(row: torch.Tensor) -> List[str]:
        ws: List[str] = []
        for t in row.tolist():
            if t in (pad, mask_id, bos, eos):
                continue
            ws.append(vocab.decode(t))
        return ws if ws else ["<empty>"]

    words_per_b = [ids_to_words(tokens[b]) for b in range(tokens.shape[0])]

    for b in range(B):
        caption_text = " ".join(words_per_b[b])
        for e in range(min(E, tokens.shape[1])):
            start = max(0.0, float(event_logits[b, e, 0].item()))
            end = max(start, float(event_logits[b, e, 1].item()))
            lines.append(f"<{int(start)}> <{int(end)}>  {caption_text}  .")
    return lines


def process_video(model: SimpleVid2SeqModel, feat_extractor: FeatureExtractor, video_dir: Path,
                  num_frames: int, max_len: int, device: str, suffix: str, vocab: CaptionVocab) -> bool:
    # frames
    frames = sorted(video_dir.glob("*.jpg"))
    if not frames:
        logger.warning("no frames in %s", video_dir)
        return False
    idxs = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    imgs = []
    for i in idxs:
        from PIL import Image
        img = Image.open(frames[i]).convert("RGB")
        img = img.resize((224, 224), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        imgs.append(arr)
    batch_frames = torch.from_numpy(np.stack(imgs, axis=0)).unsqueeze(0).to(device)  # (1, T, 3, H, W)

    with torch.no_grad():
        feats = feat_extractor.extract_features(batch_frames)  # (1, T, D)
        out = model(feats, label_hints=torch.full((1, 1), model.bos_token_id, dtype=torch.long, device=device))
        event_logits = out["event_logits"]
        decoded = greedy_decode(model, features=feats, max_len=max_len)

    lines = tokens_to_lines(decoded, event_logits, model, vocab)
    out_path = video_dir / f"vid2seq_{video_dir.name}{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(lines, f, ensure_ascii=False, indent=2)
    logger.info("saved prediction: %s", out_path)
    return True


def main():
    ap = argparse.ArgumentParser(description="Autoregressive predict vid2seq captions on coarse set")
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_frames", type=int, default=100)
    ap.add_argument("--sample_ratio", type=float, default=None)
    ap.add_argument("--sample_num", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--suffix", type=str, default="_predict", help="suffix for output file")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint + vocab, then build model
    ckpt, vocab_dict = load_checkpoint(Path(args.checkpoint), device)
    vocab = CaptionVocab.from_dict(vocab_dict)
    ckpt_cfg = ckpt.get("config", {})
    model = SimpleVid2SeqModel(
        max_events=ckpt_cfg.get("max_events", 20),
        vocab_size=vocab.size,
    ).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    feat_extractor = FeatureExtractor(device=device)
    model.eval()

    ds = LocalVid2SeqDataset(dataset_root=args.dataset_root, split="train", split_ratio=1.0,
                             num_frames=args.num_frames, image_size=(224, 224), seed=args.seed)
    vids = ds.video_ids
    if args.sample_num is not None:
        np.random.seed(args.seed)
        vids = list(np.random.choice(vids, size=min(args.sample_num, len(vids)), replace=False))
    elif args.sample_ratio is not None:
        np.random.seed(args.seed)
        k = max(1, int(len(vids) * args.sample_ratio))
        vids = list(np.random.choice(vids, size=k, replace=False))
    logger.info("processing %d videos", len(vids))

    for vid in vids:
        process_video(model, feat_extractor, Path(args.dataset_root) / vid,
                  num_frames=args.num_frames, max_len=args.max_len,
                  device=device, suffix=args.suffix, vocab=vocab)


if __name__ == "__main__":
    main()
