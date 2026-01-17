"""
管理 breakfast_progress.json 文件的工具脚本
用于添加、删除和查看精细打标进度
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

def load_progress(progress_file: Path) -> Dict:
    """加载进度文件"""
    if not progress_file.exists():
        return {
            "timestamp": datetime.now().isoformat(),
            "fine_grained": {
                "completed": [],
                "failed": []
            },
            "coarse_grained": {
                "completed": [],
                "failed": []
            }
        }
    
    with open(progress_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_progress(progress_file: Path, data: Dict):
    """保存进度文件"""
    data["timestamp"] = datetime.now().isoformat()
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ 进度文件已保存: {progress_file}")


def add_fine_grained_video(progress_file: Path, video_id: str):
    """添加精细打标完成的视频"""
    data = load_progress(progress_file)
    
    if video_id in data["fine_grained"]["completed"]:
        print(f"⚠️  视频 {video_id} 已在精细打标完成列表中")
        return
    
    # 从失败列表中移除（如果存在）
    if video_id in data["fine_grained"]["failed"]:
        data["fine_grained"]["failed"].remove(video_id)
        print(f"✓ 从失败列表移除 {video_id}")
    
    data["fine_grained"]["completed"].append(video_id)
    data["fine_grained"]["completed"].sort()
    
    save_progress(progress_file, data)
    print(f"✓ 添加精细打标视频: {video_id}")


def remove_fine_grained_video(progress_file: Path, video_id: str):
    """从精细打标列表中移除视频"""
    data = load_progress(progress_file)
    
    if video_id not in data["fine_grained"]["completed"]:
        print(f"⚠️  视频 {video_id} 不在精细打标完成列表中")
        return
    
    data["fine_grained"]["completed"].remove(video_id)
    
    save_progress(progress_file, data)
    print(f"✓ 移除精细打标视频: {video_id}")


def mark_failed_video(progress_file: Path, video_id: str, label_type: str = 'fine_grained'):
    """标记失败的视频"""
    data = load_progress(progress_file)
    
    if video_id not in data[label_type]["failed"]:
        data[label_type]["failed"].append(video_id)
        data[label_type]["failed"].sort()
        
        save_progress(progress_file, data)
        print(f"✓ 标记失败视频: {video_id} ({label_type})")
    else:
        print(f"⚠️  视频 {video_id} 已在失败列表中")


def show_progress(progress_file: Path):
    """显示当前进度"""
    if not progress_file.exists():
        print(f"⚠️  进度文件不存在: {progress_file}")
        return
    
    data = load_progress(progress_file)
    
    print("=" * 60)
    print("标注进度")
    print("=" * 60)
    print(f"更新时间: {data.get('timestamp', 'Unknown')}")
    print()
    
    fine_completed = data["fine_grained"]["completed"]
    fine_failed = data["fine_grained"]["failed"]
    coarse_completed = data["coarse_grained"]["completed"]
    coarse_failed = data["coarse_grained"]["failed"]
    
    print("精细打标 (Fine-grained):")
    print(f"  ✓ 完成: {len(fine_completed)} 个")
    if fine_completed:
        print(f"     {fine_completed}")
    print(f"  ✗ 失败: {len(fine_failed)} 个")
    if fine_failed:
        print(f"     {fine_failed}")
    print()
    
    print("粗略打标 (Coarse-grained):")
    print(f"  ✓ 完成: {len(coarse_completed)} 个")
    if coarse_completed:
        print(f"     {coarse_completed}")
    print(f"  ✗ 失败: {len(coarse_failed)} 个")
    if coarse_failed:
        print(f"     {coarse_failed}")
    print("=" * 60)


def scan_and_update(data_root: Path, progress_file: Path):
    """扫描数据集目录并自动更新进度文件"""
    print("扫描数据集目录...")
    print(f"数据路径: {data_root}")
    
    data = load_progress(progress_file)
    added_count = 0
    
    for video_dir in sorted(data_root.glob('*')):
        if not video_dir.is_dir():
            continue
        
        video_id = video_dir.name
        
        # 检查是否有精细标注文件
        fine_label_patterns = [
            f'spark_labels_{video_id}.json',
            f'fine_labels_{video_id}.json',
            f'fine_grained_{video_id}.json',
        ]
        
        has_fine_label = False
        for pattern in fine_label_patterns:
            if (video_dir / pattern).exists():
                has_fine_label = True
                break
        
        if has_fine_label and video_id not in data["fine_grained"]["completed"]:
            data["fine_grained"]["completed"].append(video_id)
            added_count += 1
            print(f"  + 发现精细标注: {video_id}")
    
    if added_count > 0:
        data["fine_grained"]["completed"].sort()
        save_progress(progress_file, data)
        print(f"\n✓ 添加了 {added_count} 个精细打标视频")
    else:
        print("\n没有发现新的精细打标视频")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='管理 breakfast_progress.json 文件')
    parser.add_argument('--data_root', type=str, default='dataset/breakfast/breakfast_5',
                       help='数据集根目录')
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 添加视频
    add_parser = subparsers.add_parser('add', help='添加精细打标视频')
    add_parser.add_argument('video_id', type=str, help='视频序号，如 000004')
    
    # 移除视频
    remove_parser = subparsers.add_parser('remove', help='移除精细打标视频')
    remove_parser.add_argument('video_id', type=str, help='视频序号')
    
    # 标记失败
    fail_parser = subparsers.add_parser('fail', help='标记失败的视频')
    fail_parser.add_argument('video_id', type=str, help='视频序号')
    
    # 显示进度
    subparsers.add_parser('show', help='显示当前进度')
    
    # 扫描并更新
    subparsers.add_parser('scan', help='扫描目录并自动更新进度')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    progress_file = data_root / 'breakfast_progress.json'
    
    if args.command == 'add':
        add_fine_grained_video(progress_file, args.video_id)
    elif args.command == 'remove':
        remove_fine_grained_video(progress_file, args.video_id)
    elif args.command == 'fail':
        mark_failed_video(progress_file, args.video_id)
    elif args.command == 'show':
        show_progress(progress_file)
    elif args.command == 'scan':
        scan_and_update(data_root, progress_file)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
