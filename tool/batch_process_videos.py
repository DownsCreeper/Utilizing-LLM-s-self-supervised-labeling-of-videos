"""
批量处理视频剪辑的工具
- 扫描数据集中的所有视频
- 按比例随机分割为精细标注和粗略标注
- 保存划分结果到 txt 文件
- 批量调用 doubao_image_api.py 处理视频
"""

import os
import json
import random
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime


class VideoBatchProcessor:
    """视频批量处理器"""
    
    def __init__(self, dataset_dir: str, dataset_name: str, sample_interval: int = 5):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_name = dataset_name
        self.sample_interval = sample_interval
        
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.dataset_dir}")
        
        self.split_file = self.dataset_dir / f"{dataset_name}_split_info.txt"
        self.progress_file = self.dataset_dir / f"{dataset_name}_progress.json"
    
    def scan_videos(self) -> List[str]:
        """扫描数据集中的所有视频目录"""
        video_ids = []
        
        for item in self.dataset_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                groups_file = item / "groups_of_five.json"
                if groups_file.exists():
                    video_ids.append(item.name)
        
        video_ids.sort()
        
        if video_ids:
            print(f"✓ 找到 {len(video_ids)} 个视频 ({video_ids[0]} ~ {video_ids[-1]})")
        else:
            print("警告：未找到任何视频目录")
        
        return video_ids
    
    def split_videos(self, video_ids: List[str], fine_ratio: float = 0.8, seed: int = 42) -> Tuple[List[str], List[str]]:
        """将视频分割为精细标注和粗略标注两部分"""
        if not 0 < fine_ratio < 1:
            raise ValueError("fine_ratio 必须在 0 到 1 之间")
        
        random.seed(seed)
        shuffled = video_ids.copy()
        random.shuffle(shuffled)
        
        split_point = int(len(shuffled) * fine_ratio)
        fine_grained = sorted(shuffled[:split_point])
        coarse_grained = sorted(shuffled[split_point:])
        
        return fine_grained, coarse_grained
    
    def save_split_info(self, fine_grained: List[str], coarse_grained: List[str], seed: int = 42) -> None:
        """保存划分信息到 txt 文件"""
        with open(self.split_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("视频标注类型划分\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据集: {self.dataset_name}\n")
            f.write(f"数据路径: {self.dataset_dir}\n")
            f.write(f"随机种子: {seed}\n\n")
            
            total = len(fine_grained) + len(coarse_grained)
            f.write(f"精细标注数量: {len(fine_grained)} ({len(fine_grained)/total*100:.1f}%)\n")
            f.write(f"粗略标注数量: {len(coarse_grained)} ({len(coarse_grained)/total*100:.1f}%)\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("精细标注视频\n")
            f.write("-" * 70 + "\n")
            for vid in fine_grained:
                f.write(f"{vid}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("粗略标注视频\n")
            f.write("-" * 70 + "\n")
            for vid in coarse_grained:
                f.write(f"{vid}\n")
        
        print(f"✓ 划分信息已保存: {self.split_file}")
    
    def init_progress(self) -> None:
        """初始化进度跟踪文件"""
        progress = {
            "timestamp": datetime.now().isoformat(),
            "fine_grained": {"completed": [], "failed": []},
            "coarse_grained": {"completed": [], "failed": []}
        }
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
    
    def load_progress(self) -> Dict:
        """加载进度信息"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def save_progress(self, progress: Dict) -> None:
        """保存进度信息"""
        progress["timestamp"] = datetime.now().isoformat()
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
    
    def process_video(self, video_id: str, annotation_type: str = "fine") -> bool:
        """处理单个视频"""
        video_dir = self.dataset_dir / video_id
        groups_file = video_dir / "groups_of_five.json"
        
        if not groups_file.exists():
            print(f"  ✗ 缺少 groups_of_five.json，跳过")
            return False
        
        output_file = video_dir / f"doubao_results_{video_id}.json"
        if output_file.exists():
            print(f"  ⊘ 已处理，跳过")
            return True
        
        cmd = [
            "python",
            str(Path(__file__).parent.parent / "api" / "doubao_image_api.py"),
            "--dataset-name", self.dataset_name,
            "--sample-interval", str(self.sample_interval),
            "--video-id", video_id,
            "--base-dir", str(self.dataset_dir.parent.parent)
        ]
        
        if annotation_type == "coarse":
            # 粗略标注：只标注动作，不标注人物、物体、环境
            coarse_prompt = "请识别这张拼接图片中主体人物的动作，只需要返回动作名称。格式：{动作：XXX}。例如：{动作：切菜}、{动作：倒水}、{动作：搅拌}、{动作：站立}等。"
            cmd.extend(["--prompt", coarse_prompt])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                print(f"  ✓ 处理成功")
                return True
            else:
                print(f"  ✗ 处理失败: {result.stderr[:100]}")
                return False
        except subprocess.TimeoutExpired:
            print(f"  ✗ 超时（>1小时）")
            return False
        except Exception as e:
            print(f"  ✗ 异常: {str(e)}")
            return False
    
    def process_batch(self, fine_grained: List[str], coarse_grained: List[str], dry_run: bool = False) -> None:
        """批量处理视频"""
        self.init_progress()
        progress = self.load_progress()
        
        total = len(fine_grained) + len(coarse_grained)
        processed = 0
        failed = 0
        
        print("\n" + "=" * 70)
        print(f"批量处理: 精细标注 {len(fine_grained)} 个 + 粗略标注 {len(coarse_grained)} 个 = 总计 {total} 个")
        print("=" * 70)
        
        if dry_run:
            print("[干运行模式] 只显示计划，不实际处理\n")
        
        # 精细标注
        for i, vid in enumerate(fine_grained, 1):
            print(f"[{i}/{len(fine_grained)}] 精细标注: {vid}", end=" ")
            if dry_run:
                print("(DRY RUN)")
                processed += 1
            else:
                if self.process_video(vid, "fine"):
                    processed += 1
                    progress["fine_grained"]["completed"].append(vid)
                else:
                    failed += 1
                    progress["fine_grained"]["failed"].append(vid)
                self.save_progress(progress)
        
        # 粗略标注
        for i, vid in enumerate(coarse_grained, 1):
            print(f"[{i}/{len(coarse_grained)}] 粗略标注: {vid}", end=" ")
            if dry_run:
                print("(DRY RUN)")
                processed += 1
            else:
                if self.process_video(vid, "coarse"):
                    processed += 1
                    progress["coarse_grained"]["completed"].append(vid)
                else:
                    failed += 1
                    progress["coarse_grained"]["failed"].append(vid)
                self.save_progress(progress)
        
        print("\n" + "=" * 70)
        print(f"处理完成: 成功 {processed} 个, 失败 {failed} 个, 成功率 {processed/total*100:.1f}%")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="批量处理视频，区分精细和粗略标注")
    
    parser.add_argument('--dataset-dir', type=str, required=True, help='数据集路径')
    parser.add_argument('--dataset-name', type=str, required=True, help='数据集名称')
    parser.add_argument('--sample-interval', type=int, default=5, help='采样间隔（默认: 5）')
    parser.add_argument('--fine-ratio', type=float, default=0.8, help='精细标注比例（默认: 0.8）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认: 42）')
    parser.add_argument('--dry-run', action='store_true', help='干运行模式')
    parser.add_argument('--no-split', action='store_true', help='使用已有的分割，不重新分割')
    
    args = parser.parse_args()
    
    processor = VideoBatchProcessor(args.dataset_dir, args.dataset_name, args.sample_interval)
    
    all_videos = processor.scan_videos()
    if not all_videos:
        return 1
    
    # 分割或加载已有分割
    if args.no_split and processor.split_file.exists():
        print(f"使用已有分割: {processor.split_file}")
        with open(processor.split_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        fine_grained = []
        coarse_grained = []
        in_fine = False
        in_coarse = False
        
        for line in lines:
            line = line.strip()
            if '精细标注' in line:
                in_fine, in_coarse = True, False
            elif '粗略标注' in line:
                in_coarse, in_fine = True, False
            elif line and line.isdigit():
                if in_fine:
                    fine_grained.append(line)
                elif in_coarse:
                    coarse_grained.append(line)
    else:
        print(f"随机分割 (精细比例: {args.fine_ratio:.0%}, 种子: {args.seed})")
        fine_grained, coarse_grained = processor.split_videos(all_videos, args.fine_ratio, args.seed)
        processor.save_split_info(fine_grained, coarse_grained, args.seed)
    
    processor.process_batch(fine_grained, coarse_grained, dry_run=args.dry_run)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
