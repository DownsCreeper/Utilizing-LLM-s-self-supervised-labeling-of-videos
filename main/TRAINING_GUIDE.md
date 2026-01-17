# Vid2Seq 模型训练指南

## 概述

本目录包含两个脚本：
- `train_v2q.py`：轻量演示版，仅使用精细标注视频文件夹内的 JSON。
- `train_vid2seq.py`：完整训练流程，需要补充真实模型实现。

## 数据集格式

**重要：训练脚本仅使用精细打标（fine-grained）数据进行训练，且精细标注文件应位于对应的视频文件夹内。**

### 数据集结构

```
dataset/breakfast/breakfast_5/
├── breakfast_progress.json          # 记录标注进度（必需）
├── 000001/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   ├── ...
│   └── doubao_results_000001.json  # 精细/精略标注（使用）
├── 000004/                          # 已完成精细打标的视频
│   ├── 000001.jpg
│   ├── 000002.jpg
│   ├── ...
│   ├── doubao_results_000004.json    # 精细标注（使用）
│   └── groups_of_five.json
└── ...
```

### 必需文件

1. **breakfast_progress.json**：记录哪些视频已完成精细打标
   ```json
   {
     "fine_grained": {
       "completed": ["000004", "000007"],
       "failed": []
     }
   }
   ```

2. **精细标注文件**：每个精细打标的视频需要以下文件之一（都放在对应视频文件夹内）
  - `doubao_results_{video_id}.json`
  - `spark_labels_{video_id}.json`
  - `fine_labels_{video_id}.json`
  - `fine_grained_{video_id}.json`
  - `labels_{video_id}.json`

3. **帧图像**：按顺序编号的 .jpg 文件

### 精细标注格式示例

```json
{
  "000001.jpg": {
    "label": {
      "raw_response": "{男人，倒水，咖啡机，厨房}"
    },
    "status": "success",
    "timestamp": "2026-01-16 17:23:23"
  },
  "000002.jpg": {
    "label": {
      "raw_response": "{男人，倒咖啡，咖啡壶，咖啡店}"
    },
    "status": "success",
    "timestamp": "2026-01-16 17:23:30"
  }
}
```

## 使用方法

### 基本用法

**快速演示：train_v2q.py（精细标注，轻量示例）**

```bash
python main/train_v2q.py \
  --train_data dataset/breakfast/breakfast_5 \
  --num_epochs 10 \
  --batch_size 8 \
  --lr 1e-2 \
  --output_dir outputs/v2q
```

**完整流程：train_vid2seq.py（需补充模型实现）**

```bash
python main/train_vid2seq.py \
  --train_data dataset/breakfast/breakfast_5 \
  --num_epochs 10 \
  --batch_size 4 \
  --output_dir outputs/vid2seq_training
```

### 包含验证集

```bash
python main/train_vid2seq.py \
    --train_data dataset/breakfast/breakfast_5 \
    --val_data dataset/breakfast/breakfast_val \
    --num_epochs 20 \
    --batch_size 8 \
    --output_dir outputs/vid2seq_training
```

### 完整参数说明

#### 数据相关参数
- `--train_data`: 训练数据路径（必需）
- `--val_data`: 验证数据路径（可选）
- `--output_dir`: 输出目录，用于保存检查点和日志（默认：`outputs/vid2seq_training`）

#### 训练参数
- `--num_epochs`: 训练轮次（默认：10）
- `--batch_size`: 批次大小（默认：4）
- `--learning_rate`: 学习率（默认：1e-4）
- `--weight_decay`: 权重衰减（默认：0.01）
- `--log_interval`: 日志记录间隔，以批次为单位（默认：10）
- `--val_interval`: 验证间隔，以轮次为单位（默认：1）
- `--save_interval`: 检查点保存间隔，以轮次为单位（默认：5）

#### 模型参数
- `--num_layers`: Transformer层数（默认：12）
- `--hidden_size`: 隐藏层大小（默认：768）
- `--num_heads`: 注意力头数（默认：12）
- `--vocab_size`: 词汇表大小（默认：32128）
- `--num_bins`: 时间分箱数量（默认：100）

## 输出结构

训练完成后，输出目录将包含：

```
outputs/vid2seq_training/
├── checkpoints/
│   ├── checkpoint_epoch_1.pt
│   ├── checkpoint_epoch_5.pt
│   ├── checkpoint_epoch_10.pt
│   ├── ...
│   └── best_model.pt
└── logs/
    ├── training_history.json
    └── config.json
```

### 检查点文件
- `checkpoint_epoch_*.pt`: 每隔指定轮次保存的检查点
- `best_model.pt`: 验证损失最低的模型

### 日志文件
- `training_history.json`: 训练历史记录，包含每个epoch的损失和学习率
- `config.json`: 训练配置的完整记录

## 训练历史记录格式

`training_history.json` 文件包含：

```json
{
  "epoch": [1, 2, 3, ...],
  "train_loss": [0.45, 0.38, 0.32, ...],
  "val_loss": [0.50, 0.42, 0.35, ...],
  "learning_rate": [1e-4, 1e-4, 1e-4, ...]
}
```

## 示例

### 快速开始（10轮训练）

```bash
python main/train_vid2seq.py \
    --train_data dataset/breakfast/breakfast_5 \
    --num_epochs 10 \
    --batch_size 4
```

### 长时间训练（50轮，使用验证集）

```bash
python main/train_vid2seq.py \
    --train_data dataset/breakfast/breakfast_5 \
    --val_data dataset/breakfast/breakfast_val \
    --num_epochs 50 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --log_interval 20 \
    --save_interval 10 \
    --output_dir outputs/vid2seq_long_training
```

### 自定义模型配置

```bash
python main/train_vid2seq.py \
    --train_data dataset/breakfast/breakfast_5 \
    --num_epochs 20 \
    --batch_size 4 \
    --num_layers 16 \
    --hidden_size 1024 \
    --num_heads 16
```

## 注意事项

1. **当前版本说明**：当前脚本提供了完整的训练框架和数据加载逻辑，但模型的实际前向/反向传播需要根据 `models/vid2seq/` 中的具体实现来完善。

2. **内存需求**：根据 `batch_size` 和模型大小，训练可能需要大量GPU内存。建议从较小的batch_size开始。

3. **数据预处理**：确保数据集格式正确，所有视频序列都包含帧图像和对应的标注文件。

4. **检查点恢复**：未来版本将支持从检查点恢复训练。

## 下一步开发

要完善这个训练脚本，需要：

1. 集成 `models/vid2seq/models.py` 中的实际模型
2. 实现正确的前向传播和损失计算
3. 添加实际的优化器（如AdamW）
4. 实现检查点的加载和恢复功能
5. 添加更多的评估指标
6. 支持多GPU训练

## 问题反馈

如有问题或建议，请查看项目文档或提交issue。
