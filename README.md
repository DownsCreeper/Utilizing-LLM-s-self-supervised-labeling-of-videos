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

- PyTorch  
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