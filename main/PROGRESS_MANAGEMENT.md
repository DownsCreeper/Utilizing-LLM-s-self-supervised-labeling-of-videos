# 精细打标进度管理工具

`manage_progress.py` 是一个用于管理 `breakfast_progress.json` 文件的工具脚本。

## 快速开始

### 查看当前进度

```bash
python main/manage_progress.py show
```

输出示例：
```
============================================================
标注进度
============================================================
更新时间: 2026-01-17T20:53:11.715322

精细打标 (Fine-grained):
  ✓ 完成: 1 个
     ['000004']
  ✗ 失败: 0 个

粗略打标 (Coarse-grained):
  ✓ 完成: 4 个
     ['000001', '000002', '000003', '000005']
  ✗ 失败: 0 个
============================================================
```

### 扫描目录并自动更新

自动扫描数据集目录，查找包含精细标注文件的视频并添加到进度文件：

```bash
python main/manage_progress.py scan
```

这会查找所有包含以下文件的视频文件夹：
- `spark_labels_*.json`
- `fine_labels_*.json`
- `fine_grained_*.json`

### 手动添加视频

添加一个已完成精细打标的视频：

```bash
python main/manage_progress.py add 000007
```

### 移除视频

从精细打标完成列表中移除视频：

```bash
python main/manage_progress.py remove 000004
```

### 标记失败视频

标记打标失败的视频：

```bash
python main/manage_progress.py fail 000008
```

## 使用场景

### 场景1：完成新的精细打标

当你完成了某个视频的精细打标（生成了 `spark_labels_*.json`）后：

```bash
# 方法1：手动添加
python main/manage_progress.py add 000009

# 方法2：自动扫描
python main/manage_progress.py scan
```

### 场景2：批量扫描并更新

如果你已经完成了多个视频的精细打标，想一次性更新进度文件：

```bash
python main/manage_progress.py scan
```

### 场景3：准备训练前检查

在开始训练前，检查有多少精细打标数据可用：

```bash
# 1. 查看进度
python main/manage_progress.py show

# 2. 扫描并更新（如果有新标注）
python main/manage_progress.py scan

# 3. 再次查看确认
python main/manage_progress.py show
```

### 场景4：修正错误

如果某个视频被错误地添加到列表中：

```bash
python main/manage_progress.py remove 000010
```

## 工作流程示例

### 完整的标注和训练流程

```bash
# 1. 完成视频打标（假设生成了 spark_labels_000011.json）

# 2. 更新进度文件
python main/manage_progress.py add 000011

# 3. 查看当前可用于训练的数据
python main/manage_progress.py show

# 4. 开始训练
python main/train_vid2seq.py \
    --train_data dataset/breakfast/breakfast_5 \
    --num_epochs 10

# 训练脚本会自动读取 breakfast_progress.json 
# 并只使用 fine_grained.completed 中的视频
```

## 命令参考

### 通用参数

- `--data_root`: 数据集根目录（默认：`dataset/breakfast/breakfast_5`）

### 子命令

| 命令 | 说明 | 参数 |
|------|------|------|
| `show` | 显示当前进度 | 无 |
| `scan` | 扫描目录并自动更新 | 无 |
| `add` | 添加精细打标视频 | `video_id` |
| `remove` | 移除精细打标视频 | `video_id` |
| `fail` | 标记失败视频 | `video_id` |

### 完整示例

```bash
# 指定不同的数据目录
python main/manage_progress.py --data_root dataset/breakfast/breakfast_cat_5 show

# 扫描特定目录
python main/manage_progress.py --data_root dataset/breakfast/breakfast_cat_5 scan

# 添加到特定目录
python main/manage_progress.py --data_root dataset/breakfast/breakfast_cat_5 add 000020
```

## 进度文件格式

`breakfast_progress.json` 文件结构：

```json
{
  "timestamp": "2026-01-17T20:53:11.715322",
  "fine_grained": {
    "completed": ["000004", "000007", "000009"],
    "failed": []
  },
  "coarse_grained": {
    "completed": ["000001", "000002", "000003"],
    "failed": []
  }
}
```

- `timestamp`: 最后更新时间
- `fine_grained.completed`: 已完成精细打标的视频列表（训练时使用）
- `fine_grained.failed`: 精细打标失败的视频列表
- `coarse_grained.completed`: 已完成粗略打标的视频列表
- `coarse_grained.failed`: 粗略打标失败的视频列表

## 注意事项

1. **自动备份**：修改前建议备份 `breakfast_progress.json`
2. **视频ID格式**：视频ID应该是6位数字格式，如 `000004`
3. **文件存在性**：`add` 命令不会检查实际的标注文件是否存在，请确保添加前已完成标注
4. **扫描范围**：`scan` 命令只检查精细标注文件的存在，不验证文件内容

## 常见问题

### Q: 扫描命令没有找到任何视频？

A: 确保：
1. 数据路径正确
2. 视频文件夹内包含精细标注文件（如 `spark_labels_*.json`）
3. 标注文件命名格式正确

### Q: 训练时加载了0个样本？

A: 检查：
```bash
# 1. 查看进度文件
python main/manage_progress.py show

# 2. 扫描并更新
python main/manage_progress.py scan

# 3. 确认有精细标注数据
python main/manage_progress.py show
```

### Q: 如何批量添加多个视频？

A: 使用 `scan` 命令自动扫描，或者编写简单的 shell 脚本：

```bash
# Bash/PowerShell
for i in 000010 000011 000012; do
    python main/manage_progress.py add $i
done
```

## 相关文档

- [训练指南](TRAINING_GUIDE.md) - 如何使用数据进行训练
- [README](README.md) - 训练模块总体说明
