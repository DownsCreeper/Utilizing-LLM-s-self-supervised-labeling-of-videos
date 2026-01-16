import os
import shutil
from pathlib import Path
import re
import cv2

def extract_frames(video_path, output_folder):
    """
    从视频中提取所有帧并保存为图片
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    错误：无法打开视频文件 {video_path}")
        return 0
    
    frame_count = 0
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 保存帧为jpg图片，帧编号从1开始
        if frame_count % 5 == 0:
            frame_filename = f"{frame_idx + 1:06d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_idx += 1
        
        frame_count += 1
        
        # 每100帧显示一次进度
        if frame_idx % 100 == 0:
            print(f"    已提取 {frame_idx}/{total_frames//5} 帧...", end='\r')
    
    cap.release()
    print(f"    成功提取 {frame_idx} 帧          ")
    return frame_idx

def main():
    # 定义源和目标路径
    source_base = r"d:\ULLV\Utilizing-LLM-s-self-supervised-labeling-of-videos\dataset\breakfast\BreakfastII_15fps_qvga_sync"
    target_base = r"d:\ULLV\Utilizing-LLM-s-self-supervised-labeling-of-videos\dataset\breakfast\Breakfast"
    
    # 创建目标基础目录
    os.makedirs(target_base, exist_ok=True)
    
    # 计数器用于生成6位数字文件夹名，从1开始
    folder_counter = 1
    total_frames = 0
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
                
                # 提取视频的所有帧
                frame_count = extract_frames(source_file, target_folder)
                total_frames += frame_count
                total_videos += 1
                
                # 如果存在.labels文件，也复制过去
                labels_file = source_file + '.labels'
                if os.path.exists(labels_file):
                    target_labels = os.path.join(target_folder, os.path.basename(labels_file))
                    shutil.copy2(labels_file, target_labels)
                
                folder_counter += 1
    
    print(f"\n完成！")
    print(f"共处理了 {total_videos} 个视频")
    print(f"提取了 {total_frames} 帧图片")
    print(f"保存路径：{target_base}")

if __name__ == "__main__":
    main()
