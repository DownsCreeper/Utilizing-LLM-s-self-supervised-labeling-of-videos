# Vid2Seq 模型训练模块

本目录包含用于训练 Vid2Seq (Video-to-Sequence) 模型的脚本和配置文件。

## 文件说明

### 核心文件

- **train_v2q.py**: 轻量演示版训练脚本，仅使用精细标注视频，读取视频文件夹内的 JSON（如 `doubao_results_{id}.json`），训练一个 Bag-of-Words 分类示例并保存权重与训练历史。
- **train_vid2seq.py**: 主训练脚本，完整训练流程（依赖更重，需自行补充模型实现）。
- **run_training.py**: 使用配置文件的启动脚本。
- **config_default.json**: 默认训练配置。

### 文档

- **TRAINING_GUIDE.md**: 详细的训练指南
  - 数据集格式说明
  - 使用方法和示例
  - 参数说明
  - 输出结构

## 快速开始

### 方法1：使用命令行参数

**快速演示（train_v2q.py，使用精细标注视频文件夹内 JSON）**

```bash
python main/train_v2q.py \
  --train_data dataset/breakfast/breakfast_5 \
  --num_epochs 10 \
  --batch_size 8 \
  --lr 1e-2 \
  --output_dir outputs/v2q
```

**完整流程（train_vid2seq.py，需要补充模型实现）**

```bash
# 基本训练（10轮）
python main/train_vid2seq.py \
  --train_data dataset/breakfast/breakfast_5 \
  --num_epochs 10 \
  --batch_size 4

# 包含验证集的训练
python main/train_vid2seq.py \
  --train_data dataset/breakfast/breakfast_5 \
  --val_data dataset/breakfast/breakfast_val \
  --num_epochs 20 \
  --batch_size 8 \
  --output_dir outputs/my_training
```

### 方法2：使用配置文件

1. 复制并修改配置文件：
```bash
cp main/config_default.json main/my_config.json
# 编辑 my_config.json 以修改训练参数
```

2. 运行训练：
```bash
# 使用默认配置
python main/run_training.py

# 使用自定义配置
python main/run_training.py --config main/my_config.json

# 使用配置文件并覆盖部分参数
python main/run_training.py \
    --config main/my_config.json \
    --num_epochs 30 \
    --batch_size 16
```

## 数据集要求

训练脚本**仅使用精细打标（fine-grained）数据**进行训练；精细标注文件应在对应视频文件夹内。

### 数据集结构

```
dataset/breakfast/breakfast_5/
├── breakfast_progress.json  # 记录精细打标进度的文件
├── 000001/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   ├── ...
│   └── doubao_results_000002.json  # 粗略标注（不使用）
├── 000004/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   ├── ...
│   ├── spark_labels_000004.json    # 精细标注（使用）
│   └── groups_of_five.json
└── ...
```

### breakfast_progress.json 格式

此文件记录哪些视频已完成精细打标：

```json
{
  "timestamp": "2026-01-17T20:53:11.715322",
  "fine_grained": {
    "completed": ["000004", "000007", ...],
    "failed": []
  },
  "coarse_grained": {
    "completed": ["000001", "000002", ...],
    "failed": []
  }
}
```

训练脚本会：
1. 读取 `breakfast_progress.json` 中的 `fine_grained.completed` 列表
2. 只加载这些视频序号的数据
3. 查找精细标注文件（如 `spark_labels_*.json`）
4. 跳过没有精细标注的视频

## 训练参数说明

### 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--train_data` | 训练数据路径 | 必需 |
| `--val_data` | 验证数据路径 | None |
| `--num_epochs` | 训练轮次 | 10 |
| `--batch_size` | 批次大小 | 4 |
| `--learning_rate` | 学习率 | 1e-4 |
| `--output_dir` | 输出目录 | outputs/vid2seq_training |

完整参数列表请参考 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

## 输出文件

训练完成后，输出目录结构：

```
outputs/vid2seq_training/
├── checkpoints/
│   ├── checkpoint_epoch_5.pt    # 周期性保存的检查点
│   ├── checkpoint_epoch_10.pt
│   └── best_model.pt            # 最佳模型
└── logs/
    ├── training_history.json    # 训练历史（loss, lr等）
    └── config.json              # 训练配置记录
```

## 训练监控

训练过程中，脚本会输出：

1. **实时日志**：每隔指定批次显示训练损失和学习率
2. **Epoch总结**：每个epoch结束时显示平均训练损失
3. **验证结果**：如有验证集，显示验证损失
4. **最佳模型**：自动保存验证损失最低的模型

示例输出：
```
==================================================
Starting training
Total epochs: 10
Batch size: 4
Training samples: 50
==================================================
Epoch [1/10] Batch [10] Loss: 0.3245 LR: 0.000100
Epoch [1/10] Batch [20] Loss: 0.2987 LR: 0.000100
Epoch 1 average training loss: 0.3012
Validating model...
Validation loss: 0.3156
New best validation loss: 0.3156
--------------------------------------------------
```

## 使用示例

### 示例1：快速测试（小数据集，少轮次）

```bash
python main/train_vid2seq.py \
    --train_data dataset/breakfast/breakfast_5 \
    --num_epochs 5 \
    --batch_size 2 \
    --log_interval 5
```

### 示例2：完整训练（大数据集，多轮次）

```bash
python main/train_vid2seq.py \
    --train_data dataset/breakfast/breakfast_5 \
    --val_data dataset/breakfast/breakfast_val \
    --num_epochs 50 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --save_interval 10 \
    --output_dir outputs/full_training
```

### 示例3：使用配置文件

创建 `my_config.json`:
```json
{
  "dataset": {
    "train_data": "dataset/breakfast/breakfast_5",
    "val_data": "dataset/breakfast/breakfast_val",
    "batch_size": 8
  },
  "training": {
    "num_epochs": 30,
    "learning_rate": 5e-5,
    "save_interval": 5
  },
  "output": {
    "output_dir": "outputs/my_experiment"
  }
}
```

运行：
```bash
python main/run_training.py --config my_config.json
```

## 进阶功能

### 自定义模型结构

```bash
python main/train_vid2seq.py \
    --train_data dataset/breakfast/breakfast_5 \
    --num_layers 16 \
    --hidden_size 1024 \
    --num_heads 16
```

### 调整训练策略

```bash
python main/train_vid2seq.py \
    --train_data dataset/breakfast/breakfast_5 \
    --learning_rate 2e-4 \
    --weight_decay 0.05 \
    --log_interval 20 \
    --val_interval 2
```

## 注意事项

1. **当前版本**：这是训练框架的初始版本，提供完整的数据加载和训练流程框架
2. **模型集成**：实际的模型前向/反向传播需要与 `models/vid2seq/` 中的实现集成
3. **GPU支持**：建议使用GPU进行训练，根据GPU内存调整batch_size
4. **数据验证**：训练前请确保数据集格式正确

## 故障排除

### 问题：找不到训练数据

```
Loaded 0 samples with fine-grained labels
```

**解决方案**：
1. 检查 `breakfast_progress.json` 文件是否存在
2. 确认 `fine_grained.completed` 列表中有视频序号
3. 确保对应视频文件夹包含精细标注文件（如 `spark_labels_*.json`）
4. 查看日志中的警告信息，了解哪些视频被跳过

**示例检查**：
```bash
# 检查进度文件
cat dataset/breakfast/breakfast_5/breakfast_progress.json

# 检查某个视频的标注文件
ls dataset/breakfast/breakfast_5/000004/spark_labels_*.json
```

### 问题：内存不足

```
Out of memory error
```

**解决方案**：
- 减小 `--batch_size`
- 减小模型大小（`--hidden_size`, `--num_layers`）

## 后续开发

计划添加的功能：
- [ ] 集成实际的Vid2Seq模型实现
- [ ] 添加更多评估指标（BLEU, METEOR等）
- [ ] 支持从检查点恢复训练
- [ ] 多GPU并行训练支持
- [ ] TensorBoard可视化
- [ ] 数据增强功能

## 相关文档

- [训练指南](TRAINING_GUIDE.md) - 详细的参数说明和使用示例
- [Vid2Seq论文](https://arxiv.org/abs/2302.14115) - 模型原理
- [项目README](../README.md) - 项目整体说明

## 问题反馈

如有问题或建议，请查看项目文档或提交issue。
