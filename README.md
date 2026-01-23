# Utilizing-LLM-s-self-supervised-labeling-of-videos
Leveraging Large Language Models for Cascaded Refinement of Video Annotation: Combining Small-Batch Precise Labeling with Large-Scale Coarse Labeling to Train Models for Full-Scale Accurate Prediction, Thereby Significantly Reducing Annotation Costs.

## Requirements

This project is developed and tested under the following environment.  
We strongly recommend using **conda** to manage dependencies, especially for CUDA-enabled deep learning libraries.

### System Requirements
- **Operating System**: Windows / Linux  
- **Python**: 3.10  
- **CUDA**: 11.8 (required for GPU acceleration, optional for CPU-only usage)

---

### Conda Dependencies

The following core libraries should be installed via **conda** to ensure binary compatibility and stable CUDA support:

- PyTorch 1.7.1
- torchvision  
- torchaudio  
- pytorch-cuda (11.8)  
- numpy  
- pandas  
- scikit-learn  
- scikit-image  

Example:
```bash
conda create -n vid2seq python=3.10
conda activate vid2seq
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy pandas scikit-learn scikit-image
```
### Pip Dependencies
```bash
pip install -r requirements.txt
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git  # 必须安装，训练/推理未装会直接报错
```

## 使用tool/process_frames.py对视频进行预处理
```bash
python tool/process_frames.py --source-base .../BreakfastII_15fps_qvga_sync --target-base .../dataset/breakfast --dataset-name breakfast --sample-interval 5 --clear-target
```

## 调用api对视频打标

### 单个视频（以豆包为例）
```bash
python api/doubao_image_api.py
```

### 批量视频
```bash
# 自定义比例（10% 精细）
python tool/batch_process_videos.py --dataset-dir "D:\ULLV\Utilizing-LLM-s-self-supervised-labeling-of-videos\dataset\breakfast\breakfast_5" --dataset-name breakfast --fine-ratio 0.1
```

## 对齐vid2seq输入

### 语音对齐
```bash
python tool/convert_doubao_to_vid2seq.py --dataset-root dataset/breakfast/breakfast_5 --overwrite
```

## 训练

### 预测模型的训练
第一阶段
```bash
python main/vid2seq_local_train.py --dataset_root dataset/breakfast/breakfast_5 --workdir outputs/vid2seq_local_stage1 --stage 1 --epochs 5 --batch-size 4
```

第二阶段
```bash
python main/vid2seq_local_train.py --dataset_root dataset/breakfast/breakfast_5 --workdir outputs/vid2seq_local_stage2 --stage 2 --mask-ratio 0.3 --epochs 5 --batch-size 4
```

### 预测模型的推理
> 说明：现在训练会自动基于训练集字幕构建词表（保存到 workdir/vocab.json，并一起写入 checkpoint），推理会用词表解码成自然语言。请使用新代码重新训练获得包含词表的 checkpoint。
一定比率
```bash
python main/vid2seq_predict_autoreg.py --dataset_root dataset/breakfast/breakfast_5 --checkpoint outputs/vid2seq_local_stage1/checkpoints/checkpoint_latest.pth --sample_ratio 0.2 --max_len 32 --num_frames 100 --suffix _predict
```
固定个数
```bash
python main/vid2seq_predict_autoreg.py \
  --dataset_root dataset/breakfast/breakfast_5 \
  --checkpoint outputs/vid2seq_local_stage1/checkpoints/checkpoint_latest.pth \
  --sample_num 10
```