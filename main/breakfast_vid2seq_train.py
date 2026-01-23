"""
Breakfast数据集上的vid2seq模型训练脚本

功能：
1. 从本地Breakfast数据集加载视频帧和标注
2. 使用vid2seq模型进行训练
3. 支持中断恢复和模型检查点保存
4. 记录训练过程和验证结果
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'models' / 'vid2seq'))

from main.breakfast_dataset_loader import get_breakfast_dataloaders
from main.breakfast_config import get_config
from main.caption_vocab import CaptionVocab, build_vocab_from_dataset, save_vocab, load_vocab

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('breakfast_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingState:
    """训练状态管理"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.checkpoint_dir / 'training_state.json'
        
        self.state = {
            'epoch': 0,
            'step': 0,
            'best_val_loss': float('inf'),
            'best_epoch': 0,
            'last_checkpoint': None,
            'start_time': datetime.now().isoformat(),
        }
        
        if self.state_file.exists():
            self._load_state()
    
    def _load_state(self):
        """加载保存的状态"""
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                loaded_state = json.load(f)
            self.state.update(loaded_state)
            logger.info(f"加载训练状态: epoch={self.state['epoch']}, step={self.state['step']}")
        except Exception as e:
            logger.warning(f"加载训练状态失败: {e}")
    
    def save(self):
        """保存状态"""
        self.state['last_update'] = datetime.now().isoformat()
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)
    
    def update_epoch(self, epoch: int):
        self.state['epoch'] = epoch
    
    def update_step(self, step: int):
        self.state['step'] = step
    
    def update_best_loss(self, loss: float, epoch: int):
        if loss < self.state['best_val_loss']:
            self.state['best_val_loss'] = loss
            self.state['best_epoch'] = epoch
            return True
        return False


class FeatureExtractor:
    """视频特征提取器 - 从视频帧提取CLIP特征"""
    
    def __init__(self, device: str = 'cuda'):
        """初始化CLIP特征提取器"""
        self.device = device
        try:
            import clip
            self.model, self.preprocess = clip.load("ViT-L/14", device=device)
            self.model.eval()
            logger.info("CLIP ViT-L/14 加载成功（必须依赖）")
        except Exception as exc:
            raise RuntimeError(
                "please install CLIP via pip install git+https://github.com/openai/CLIP.git"
            ) from exc
    
    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        提取视频帧特征
        
        Args:
            frames: 形状为 (batch, num_frames, 3, H, W) 的张量
            
        Returns:
            形状为 (batch, num_frames, feature_dim) 的特征张量
        """
        batch_size, num_frames = frames.shape[:2]

        features_list = []
        with torch.no_grad():
            for i in range(num_frames):
                frame_batch = frames[:, i]  # (batch, 3, H, W)
                # CLIP 预处理: 输入需在 [0,1]; preprocess 负责归一化/resize
                imgs = []
                for j in range(frame_batch.shape[0]):
                    img = frame_batch[j]
                    img = torch.clamp(img, 0, 1)
                    # clip.preprocess expects PIL; convert then back to float32 tensor
                    import torchvision.transforms.functional as TF
                    pil = TF.to_pil_image(img.cpu())
                    img = self.preprocess(pil)
                    imgs.append(img)
                frame_batch = torch.stack(imgs, dim=0).to(self.device).float()
                image_features = self.model.encode_image(frame_batch).float()
                features_list.append(image_features)

        features = torch.stack(features_list, dim=1)
        return features


class SimpleVid2SeqModel(nn.Module):
    """简化的Vid2Seq模型 - 用于演示"""
    
    def __init__(self, 
                 feature_dim: int = 768,
                 hidden_dim: int = 512,
                 vocab_size: int = 32128,
                 max_events: int = 20,
                 max_caption_len: int = 256):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_events = max_events
        self.max_caption_len = max_caption_len
        self.pad_token_id = 0
        self.mask_token_id = vocab_size  # 额外的mask token
        self.bos_token_id = vocab_size + 1
        self.eos_token_id = vocab_size + 2
        
        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # 标签嵌入（包含pad/mask/bos/eos）
        self.label_embedding = nn.Embedding(vocab_size + 3, hidden_dim, padding_idx=self.pad_token_id)
        self.pos_embedding = nn.Embedding(max_caption_len + max_events + 4, hidden_dim)
        
        # 时间编码器（Transformer）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4,
        )
        
        # 事件检测头（用于预测事件数和时间戳）
        self.event_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_events * 2),  # 预测start和end时间戳
        )

        # 文本解码器（自回归TransformerDecoder）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.text_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.decoder_out = nn.Linear(hidden_dim, vocab_size + 3)
    
    def forward(self, 
                features: torch.Tensor,
                label_hints: Optional[torch.Tensor] = None,
                timestamps: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: 形状为 (batch, num_frames, feature_dim) 的特征
            label_hints: 可选的标签提示（包含mask/pad），形状为(batch, max_events)
            timestamps: 可选的时间戳
            
        Returns:
            包含输出的字典
        """
        batch_size, num_frames, _ = features.shape
        
        # 投影特征
        projected = self.feature_projection(features)  # (batch, num_frames, hidden_dim)
        
        # 通过时间编码器
        encoded = self.temporal_encoder(projected)  # (batch, num_frames, hidden_dim)
        
        # 取最后一帧的编码作为全局表征
        global_repr = encoded[:, -1, :]  # (batch, hidden_dim)
        
        # 事件检测
        event_logits = self.event_detector(global_repr)  # (batch, max_events * 2)
        event_logits = event_logits.view(batch_size, self.max_events, 2)
        
        # 文本生成（自回归解码）
        if label_hints is None:
            raise ValueError("label_hints (decoder input tokens) cannot be None for decoding")
        positions = torch.arange(label_hints.size(1), device=features.device).unsqueeze(0)
        dec_emb = self.label_embedding(label_hints) + self.pos_embedding(positions)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(label_hints.size(1)).to(features.device)
        # memory 是 encoded (batch, num_frames, hidden)
        dec_out = self.text_decoder(tgt=dec_emb, memory=encoded, tgt_mask=tgt_mask)
        caption_logits = self.decoder_out(dec_out)  # (batch, seq_len, vocab_size+3)
        
        return {
            'event_logits': event_logits,
            'caption_logits': caption_logits,
            'encoded_features': encoded,
        }


class Trainer:
    """训练器类"""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 workdir: str,
                 vocab: CaptionVocab,
                 device: str = 'cuda'):
        """初始化训练器
        
        Args:
            config: 配置字典
            workdir: 工作目录
            device: 计算设备
        """
        self.config = config
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.vocab = vocab
        
        logger.info(f"工作目录: {self.workdir}")
        logger.info(f"计算设备: {device}")
        
        # 保存配置
        self._save_config()
        
        # 初始化组件
        self.train_state = TrainingState(str(self.workdir / 'checkpoints'))
        self.feature_extractor = FeatureExtractor(device=device)
        self.model = SimpleVid2SeqModel(
            max_events=config.get('max_events', 20),
            vocab_size=self.vocab.size
        ).to(device)
        self.pad_token_id = self.vocab.pad_token_id
        self.mask_token_id = self.model.mask_token_id  # 与模型中的mask保持一致
        self.stage = config.get('train_stage', '1')  # '1', '2', 'mixed'
        self.mask_ratio = float(config.get('mask_ratio', 0.3))
        self.mixed_stage2_ratio = float(config.get('mixed_stage2_ratio', 0.5))
        self.caption_loss_weight = float(config.get('caption_loss_weight', 1.0))
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.0001),
            weight_decay=config.get('weight_decay', 0.)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_training_epochs', 10) * 100,
        )
        
        # TensorBoard
        self.writer = SummaryWriter(str(self.workdir / 'logs'))
        
        # 损失函数
        self.caption_loss_fn = nn.CrossEntropyLoss(ignore_index=self.model.pad_token_id)
        self.timestamp_loss_fn = nn.MSELoss()
        
        logger.info(f"模型参数数量: {self._count_parameters():,}")
    
    def _count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _save_config(self):
        """保存配置文件"""
        config_file = self.workdir / 'config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            # 转换config中的不可序列化对象
            config_serializable = {}
            for k, v in self.config.items():
                if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    config_serializable[k] = v
            json.dump(config_serializable, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"配置已保存到: {config_file}")
    
    def _collate_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """处理批次数据
        
        Returns:
            (features, labels)
        """
        frames = batch['frames'].to(self.device)  # (batch, num_frames, 3, H, W) in [0,1]
        
        # 提取特征
        with torch.no_grad():
            features = self.feature_extractor.extract_features(frames)
        
        # 准备标签
        labels = {
            'captions': batch['captions'],
            'timestamps': batch['timestamps'],
            'video_ids': batch['video_ids'],
        }
        
        return features, labels

    def _encode_captions(self, captions: Any) -> torch.Tensor:
        """将caption列表编码为token序列，形状(batch, seq_len)。
        序列格式: [BOS] cap1 cap2 ... capN [EOS] 并pad到最大长度(max_events+2)
        使用自然语言词表（caption级别）。
        """
        batch_size = len(captions)
        max_seq_len = self.model.max_events + 2  # bos + caps + eos
        seq = torch.full(
            (batch_size, max_seq_len),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        for b, cap_list in enumerate(captions):
            seq[b, 0] = self.model.bos_token_id
            write_pos = 1
            for text in cap_list[:self.model.max_events]:
                if write_pos >= max_seq_len - 1:
                    break
                seq[b, write_pos] = self.vocab.encode(text)
                write_pos += 1
            seq[b, min(write_pos, max_seq_len - 1)] = self.model.eos_token_id
        return seq

    def _make_stage2_hints(self, caption_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成带mask的hint以及mask位置"""
        mask = (torch.rand_like(caption_ids.float()) < self.mask_ratio) & (caption_ids != self.pad_token_id)
        label_hints = caption_ids.clone()
        label_hints[mask] = self.mask_token_id
        return label_hints, mask
    
    def _compute_loss(self, 
                     model_output: Dict[str, torch.Tensor],
                     labels: Dict[str, Any],
                     caption_ids: torch.Tensor,
                     stage: str,
                     mask_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算损失"""
        total_loss = 0.
        
        # 事件损失（暂时使用L2损失来近似时间戳预测）
        event_logits = model_output['event_logits']  # (batch, max_events, 2)
        batch_size = event_logits.shape[0]
        
        # 将时间戳转换为张量
        timestamps = torch.zeros(batch_size, 20, 2, device=self.device)
        for i, ts_list in enumerate(labels['timestamps']):
            for j, (start, end) in enumerate(ts_list[:20]):
                timestamps[i, j, 0] = start
                timestamps[i, j, 1] = end
        
        timestamp_loss = self.timestamp_loss_fn(event_logits, timestamps)
        total_loss += timestamp_loss
        
        # caption损失
        caption_logits = model_output['caption_logits']  # (batch, seq_len, vocab)
        target = caption_ids

        if stage == '1' or mask_positions is None:
            # 全部非pad token为目标
            valid_mask = (target != self.pad_token_id)
        else:
            # 仅对被mask的token计算损失
            valid_mask = mask_positions & (target != self.pad_token_id)

        if valid_mask.any():
            caption_loss = self.caption_loss_fn(
                caption_logits[valid_mask],
                target[valid_mask],
            )
            total_loss += self.caption_loss_weight * caption_loss
        
        return total_loss
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: epoch编号
            
        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0.
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 前向传播
            features, labels = self._collate_batch(batch)
            caption_ids = self._encode_captions(labels['captions'])

            # 阶段选择
            current_stage = self.stage
            if self.stage == 'mixed':
                if np.random.rand() < self.mixed_stage2_ratio:
                    current_stage = '2'
                else:
                    current_stage = '1'

            if current_stage == '2':
                label_hints, mask_positions = self._make_stage2_hints(caption_ids)
            else:
                label_hints, mask_positions = caption_ids, None

            model_output = self.model(features, label_hints=label_hints)
            
            # 计算损失
            loss = self._compute_loss(model_output, labels, caption_ids, stage=current_stage, mask_positions=mask_positions)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            max_grad_norm = self.config.get('max_grad_norm', 1.0)
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # 记录到TensorBoard
            global_step = self.train_state.state['step']
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            
            self.train_state.update_step(global_step + 1)
        
        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1} 训练损失: {avg_loss:.4f}")
        
        return avg_loss
    
    def eval_epoch(self, val_loader: DataLoader, epoch: int) -> float:
        """验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
            epoch: epoch编号
            
        Returns:
            平均损失
        """
        self.model.eval()
        total_loss = 0.
        num_batches = 0
        
        progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        with torch.no_grad():
            for batch in progress_bar:
                features, labels = self._collate_batch(batch)
                caption_ids = self._encode_captions(labels['captions'])
                # 验证阶段使用teacher forcing输入，评估完整序列损失
                model_output = self.model(features, label_hints=caption_ids)
                loss = self._compute_loss(model_output, labels, caption_ids, stage='1', mask_positions=None)
                
                total_loss += loss.item()
                num_batches += 1
                
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1} 验证损失: {avg_loss:.4f}")
        
        # 记录到TensorBoard
        global_step = self.train_state.state['step']
        self.writer.add_scalar('Val/Loss', avg_loss, global_step)
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点
        
        Args:
            epoch: epoch编号
            is_best: 是否为最佳模型
        """
        checkpoint_dir = self.workdir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_state': self.train_state.state,
            'vocab': self.vocab.to_dict(),
        }
        
        # 保存最新检查点
        latest_path = checkpoint_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        logger.info(f"检查点已保存: {latest_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = checkpoint_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"最佳模型已保存: {best_path}")
        
        # 定期保存epoch检查点
        if (epoch + 1) % 5 == 0:
            epoch_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1:03d}.pth'
            torch.save(checkpoint, epoch_path)
            logger.info(f"Epoch检查点已保存: {epoch_path}")
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> int:
        """加载检查点
        
        Args:
            checkpoint_path: 检查点路径，如果为None则加载最新检查点
            
        Returns:
            开始的epoch
        """
        if checkpoint_path is None:
            checkpoint_path = self.workdir / 'checkpoints' / 'checkpoint_latest.pth'
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"检查点不存在: {checkpoint_path}")
            return 0
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"检查点已加载，继续从epoch {start_epoch}")
            return start_epoch
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return 0
    
    def train(self, 
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_epochs: int,
             resume_from_checkpoint: bool = True):
        """训练循环
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 总epoch数
            resume_from_checkpoint: 是否从检查点恢复
        """
        start_epoch = 0
        
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint()
        
        logger.info(f"开始训练: epoch {start_epoch} 到 {num_epochs-1}")
        
        for epoch in range(start_epoch, num_epochs):
            self.train_state.update_epoch(epoch)
            
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss = self.eval_epoch(val_loader, epoch)
            
            # 检查是否是最佳模型
            is_best = self.train_state.update_best_loss(val_loss, epoch)
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            
            # 保存状态
            self.train_state.save()
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} 完成 "
                       f"(Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f})")
        
        logger.info(f"训练完成！最佳模型在epoch {self.train_state.state['best_epoch']+1}")
        self.writer.close()


def main(args: argparse.Namespace):
    """主训练函数"""
    
    # 加载配置
    cfg_obj = get_config()
    config = cfg_obj.to_dict() if hasattr(cfg_obj, "to_dict") else dict(cfg_obj)
    
    # 覆盖配置中的参数
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['num_training_epochs'] = args.epochs
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.stage:
        config['train_stage'] = args.stage
    if args.mask_ratio is not None:
        config['mask_ratio'] = args.mask_ratio
    if args.mixed_stage2_ratio is not None:
        config['mixed_stage2_ratio'] = args.mixed_stage2_ratio
    if args.caption_loss_weight is not None:
        config['caption_loss_weight'] = args.caption_loss_weight
    
    logger.info("=" * 80)
    logger.info("Breakfast数据集 - Vid2Seq模型训练")
    logger.info("=" * 80)
    logger.info(f"配置摘要:")
    logger.info(f"  数据集: {config.get('dataset_configs', {}).get('base_dir')}")
    logger.info(f"  批大小: {config.get('batch_size')}")
    logger.info(f"  学习率: {config.get('learning_rate')}")
    logger.info(f"  训练轮次: {config.get('num_training_epochs')}")
    logger.info(f"  训练阶段: {config.get('train_stage', '1')} (mask_ratio={config.get('mask_ratio', 0.3)}, mix_stage2={config.get('mixed_stage2_ratio', 0.5)})")
    logger.info(f"  工作目录: {args.workdir}")
    logger.info("=" * 80)
    
    # 创建数据加载器 & 词表
    logger.info("加载数据集...")
    dataset_root = config.get('dataset_configs', {}).get('base_dir')
    batch_size = config.get('batch_size', 4)
    split_ratio = config.get('dataset_configs', {}).get('split_ratio', 0.8)

    vocab_path = Path(args.workdir) / "vocab.json"
    if vocab_path.exists():
        vocab = load_vocab(vocab_path)
        logger.info(f"已从 {vocab_path} 读取词表，大小: {vocab.size}")
    else:
        vocab = build_vocab_from_dataset(dataset_root, split_ratio=split_ratio, seed=42)
        save_vocab(vocab, vocab_path)
        logger.info(f"已构建词表并保存到 {vocab_path}，大小: {vocab.size}")
    
    train_loader, val_loader = get_breakfast_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=0,  # Windows不支持多进程
        num_frames=100,
        split_ratio=split_ratio,
    )
    
    logger.info(f"训练集批数: {len(train_loader)}")
    logger.info(f"验证集批数: {len(val_loader)}")
    
    # 创建训练器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    trainer = Trainer(
        config=config,
        workdir=args.workdir,
        vocab=vocab,
        device=device,
    )
    
    # 开始训练
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get('num_training_epochs', 10),
        resume_from_checkpoint=not args.no_resume,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Breakfast数据集上的Vid2Seq训练脚本')
    parser.add_argument('--workdir', 
                       type=str, 
                       default=r'D:\ULLV\Utilizing-LLM-s-self-supervised-labeling-of-videos\outputs\vid2seq_breakfast',
                       help='工作目录')
    parser.add_argument('--batch-size', 
                       type=int, 
                       default=None,
                       help='批大小')
    parser.add_argument('--epochs', 
                       type=int, 
                       default=None,
                       help='训练轮次')
    parser.add_argument('--learning-rate', 
                       type=float, 
                       default=None,
                       help='学习率')
    parser.add_argument('--stage',
                       type=str,
                       choices=['1', '2', 'mixed'],
                       default=None,
                       help='训练阶段: 1=仅标签作为目标; 2=遮挡部分标签作为输入; mixed=按比例混合')
    parser.add_argument('--mask-ratio',
                       type=float,
                       default=None,
                       help='Stage2遮挡比例，默认0.3')
    parser.add_argument('--mixed-stage2-ratio',
                       type=float,
                       default=None,
                       help='mixed模式下使用Stage2批次的比例，默认0.5')
    parser.add_argument('--caption-loss-weight',
                       type=float,
                       default=None,
                       help='caption损失的权重，默认1.0')
    parser.add_argument('--no-resume', 
                       action='store_true',
                       help='不从检查点恢复')
    
    args = parser.parse_args()
    
    main(args)
