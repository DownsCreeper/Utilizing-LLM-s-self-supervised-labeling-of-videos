"""
将视频每5帧取1帧，将这5帧水平拼接成一张图片
即：每25帧产生一张拼接图片
"""

import os
import cv2
import numpy as np
from pathlib import Path
import re
import shutil


def concat_frames_horizontally(frames: list) -> np.ndarray:
    """
    水平拼接多张图片
    
    Args:
        frames: 图片数组列表
        
    Returns:
        拼接后的图片数组
    """
    if not frames:
        return None
    
    # 获取图片高度（以第一张为参考）
    height = frames[0].shape[0]
    
    # 统一所有图片的高度
    resized_frames = []
    for frame in frames:
        if frame.shape[0] != height:
            # 调整高度
            aspect_ratio = frame.shape[1] / frame.shape[0]
            new_width = int(height * aspect_ratio)
            resized = cv2.resize(frame, (new_width, height))
        else:
            resized = frame
        resized_frames.append(resized)
    
    # 水平拼接
    concatenated = np.hstack(resized_frames)
    return concatenated


def process_video_to_concat_frames(video_path: str, 
                                   output_folder: str,
                                   sample_interval: int = 5) -> int:
    """
    从视频中每sample_interval帧取1帧，将5帧拼接成一张图片
    即：每 sample_interval * 5 = 25 帧产生一张拼接图片
    
    Args:
        video_path: 视频文件路径
        output_folder: 输出文件夹
        sample_interval: 采样间隔（每N帧取1帧）
        
    Returns:
        生成的拼接图片数量
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    错误：无法打开视频文件 {video_path}")
        return 0
    
    os.makedirs(output_folder, exist_ok=True)
    
    frame_count = 0
    concat_count = 0
    sampled_frames = []  # 采样后的帧buffer
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # 视频结束，处理剩余的帧
            if sampled_frames:
                concatenated = concat_frames_horizontally(sampled_frames)
                if concatenated is not None:
                    concat_filename = f"{concat_count + 1:06d}.jpg"
                    concat_path = os.path.join(output_folder, concat_filename)
                    cv2.imwrite(concat_path, concatenated)
                    concat_count += 1
            break
        
        # 每sample_interval帧取1帧
        if frame_count % sample_interval == 0:
            sampled_frames.append(frame)
        
        frame_count += 1
        
        # 当采样了5帧后，拼接成一张图片
        if len(sampled_frames) == 5:
            concatenated = concat_frames_horizontally(sampled_frames)
            if concatenated is not None:
                concat_filename = f"{concat_count + 1:06d}.jpg"
                concat_path = os.path.join(output_folder, concat_filename)
                cv2.imwrite(concat_path, concatenated)
                concat_count += 1
            
            sampled_frames = []
            
            # 显示进度
            if frame_count % (sample_interval * 5 * 20) == 0:
                print(f"    已处理 {frame_count}/{total_frames} 帧...", end='\r')
    
    cap.release()
    print(f"    成功生成 {concat_count} 张拼接图片          ")
    return concat_count


def main():
    """主程序"""
    
    # 定义路径
    source_base = r"d:\ULLV\Utilizing-LLM-s-self-supervised-labeling-of-videos\dataset\breakfast\BreakfastII_15fps_qvga_sync"
    target_base = r"d:\ULLV\Utilizing-LLM-s-self-supervised-labeling-of-videos\dataset\breakfast\breakfast_cat_5"
    
    # 创建目标基础目录
    os.makedirs(target_base, exist_ok=True)
    
    # 清空目标目录（如果存在）
    if os.path.exists(target_base):
        print("清空目标目录...")
        for item in os.listdir(target_base):
            item_path = os.path.join(target_base, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    
    # 计数器用于生成6位数字文件夹名，从1开始
    folder_counter = 1
    total_concat_images = 0
    total_videos = 0
    
    # 遍历所有Pxx目录
    for p_dir in sorted(os.listdir(source_base)):
        p_path = os.path.join(source_base, p_dir)
        
        # 检查是否是目录，且符合Pxx模式
        if not os.path.isdir(p_path) or not re.match(r'^P\d+$', p_dir):
            continue
        
        cam01_path = os.path.join(p_path, "cam01")
        
        # 检查cam01目录是否存在
        if not os.path.exists(cam01_path):
            print(f"跳过：{p_dir} 没有 cam01 目录")
            continue
        
        print(f"处理：{p_dir}")
        
        # 遍历cam01目录中的所有.avi文件
        for file in sorted(os.listdir(cam01_path)):
            if file.endswith('.avi'):
                source_file = os.path.join(cam01_path, file)
                
                # 创建目标文件夹（6位数字格式，从000001开始）
                folder_name = f"{folder_counter:06d}"
                target_folder = os.path.join(target_base, folder_name)
                os.makedirs(target_folder, exist_ok=True)
                
                print(f"  视频：{file} -> {folder_name}/")
                
                # 从视频中每5帧取1帧，将这5帧拼接成一张图片（每25帧产生一张）
                concat_count = process_video_to_concat_frames(
                    source_file, 
                    target_folder,
                    sample_interval=5
                )
                
                total_concat_images += concat_count
                total_videos += 1
                
                # 如果存在.labels文件，也复制过去
                labels_file = source_file + '.labels'
                if os.path.exists(labels_file):
                    target_labels = os.path.join(target_folder, os.path.basename(labels_file))
                    shutil.copy2(labels_file, target_labels)
                
                folder_counter += 1
    
    print(f"\n完成！")
    print(f"共处理了 {total_videos} 个视频")
    print(f"生成了 {total_concat_images} 张拼接图片")
    print(f"保存路径：{target_base}")


if __name__ == "__main__":
    main()
