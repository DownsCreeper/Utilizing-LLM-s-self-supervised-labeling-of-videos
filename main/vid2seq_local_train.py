"""
最小训练入口：使用本地帧 + vid2seq 标注 JSON，复用现有 PyTorch 模型/训练器。
- 不调用外部 API；直接读取 JPG 帧与 vid2seq_*.json。
- 采用 main/local_vid2seq_loader.py 生成数据。
- 训练逻辑复用 main/breakfast_vid2seq_train.py 中的 Trainer（包含两阶段/混合训练能力）。
"""

import sys
from pathlib import Path
import argparse
import logging

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "main"))  # ensure main package imports

from main.breakfast_config import get_config  # type: ignore
from main.breakfast_vid2seq_train import Trainer  # type: ignore
from main.local_vid2seq_loader import get_local_dataloaders  # type: ignore
from main.caption_vocab import build_vocab_from_dataset, save_vocab, load_vocab, CaptionVocab  # type: ignore


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace):
    cfg_obj = get_config()
    config = cfg_obj.to_dict() if hasattr(cfg_obj, "to_dict") else dict(cfg_obj)
    # 覆盖配置
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.epochs:
        config["num_training_epochs"] = args.epochs
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.stage:
        config["train_stage"] = args.stage
    if args.mask_ratio is not None:
        config["mask_ratio"] = args.mask_ratio
    if args.mixed_stage2_ratio is not None:
        config["mixed_stage2_ratio"] = args.mixed_stage2_ratio
    if args.caption_loss_weight is not None:
        config["caption_loss_weight"] = args.caption_loss_weight
    if args.max_events is not None:
        config["max_events"] = args.max_events

    dataset_root = args.dataset_root
    batch_size = config.get("batch_size", 4)

    logger.info("加载数据集... root=%s", dataset_root)

    # 词表：优先读取工作目录已有 vocab.json，否则基于训练集构建
    vocab_path = Path(args.workdir) / "vocab.json"
    split_ratio = args.split_ratio
    if vocab_path.exists():
        vocab = load_vocab(vocab_path)
        logger.info("已从 %s 读取词表，大小=%d", vocab_path, vocab.size)
    else:
        vocab = build_vocab_from_dataset(dataset_root, split_ratio=split_ratio, seed=args.seed)
        save_vocab(vocab, vocab_path)
        logger.info("已构建并保存词表到 %s，大小=%d", vocab_path, vocab.size)
    train_loader, val_loader = get_local_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=0,
        num_frames=args.num_frames,
        split_ratio=args.split_ratio,
        seed=args.seed,
        image_size=(args.image_size, args.image_size),
        max_events=config.get("max_events", 20),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("使用设备: %s", device)

    trainer = Trainer(
        config=config,
        workdir=args.workdir,
        vocab=vocab,
        device=device,
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get("num_training_epochs", 10),
        resume_from_checkpoint=not args.no_resume,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vid2Seq 本地帧+JSON 训练入口")
    parser.add_argument("--dataset_root", type=str, required=True, help="数据集根目录，例如 dataset/breakfast/breakfast_5")
    parser.add_argument("--workdir", type=str, default=str(PROJECT_ROOT / "outputs" / "vid2seq_local"), help="工作目录")
    parser.add_argument("--batch-size", type=int, default=None, help="批大小")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮次")
    parser.add_argument("--learning-rate", type=float, default=None, help="学习率")
    parser.add_argument("--num-frames", type=int, default=100, help="每视频采样帧数")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--image-size", type=int, default=224, help="帧缩放尺寸")
    parser.add_argument("--seed", type=int, default=42, help="随机种子用于切分")
    parser.add_argument("--stage", type=str, choices=["1", "2", "mixed"], default=None, help="训练阶段选择")
    parser.add_argument("--mask-ratio", type=float, default=None, help="Stage2 遮挡比例，默认0.3")
    parser.add_argument("--mixed-stage2-ratio", type=float, default=None, help="mixed 模式下 Stage2 占比，默认0.5")
    parser.add_argument("--caption-loss-weight", type=float, default=None, help="caption 损失权重，默认1.0")
    parser.add_argument("--max-events", type=int, default=None, help="事件上限，默认20")
    parser.add_argument("--no-resume", action="store_true", help="不从检查点恢复")
    args = parser.parse_args()

    main(args)
