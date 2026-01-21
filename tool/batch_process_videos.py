"""
批量处理视频剪辑的工具
- 扫描数据集中的所有视频
- 按比例随机分割为精细标注和粗略标注
- 保存划分结果到 txt 文件
- 批量调用 doubao_image_api.py 处理视频
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

# 添加 api 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
from doubao_image_api import DoubaoImageUnderstandingAPI, BatchImageProcessor
from standardize_response import parse_dict_format


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
        self.vocab_file = self.dataset_dir / f"{dataset_name}_vocab.json"

    def load_vocab(self) -> Dict[str, List[str]]:
        """加载或初始化词典（英文值）"""
        default_vocab = {
            "person": [],
            "action": ["still"],
            "object": [],
            "environment": []
        }
        if self.vocab_file.exists():
            try:
                with open(self.vocab_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 确保四个键都存在
                for k in default_vocab:
                    data.setdefault(k, [])
                return data
            except Exception:
                return default_vocab
        else:
            return default_vocab

    def save_vocab(self, vocab: Dict[str, List[str]]) -> None:
        """保存词典到文件"""
        try:
            with open(self.vocab_file, 'w', encoding='utf-8') as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)
            print(f"✓ 词典已更新: {self.vocab_file}")
        except Exception as e:
            print(f"警告：词典保存失败 - {str(e)}")

    def build_prompt(self, annotation_type: str, vocab: Dict[str, List[str]]) -> str:
        """基于注释类型与词典构建提示词（英文键/值）"""
        # 将列表转成逗号分隔的简短文本
        def join(lst: List[str]) -> str:
            if not lst:
                return "(empty)"
            # 控制长度，避免提示词过长
            return ", ".join(sorted(set(lst)))

        if annotation_type == "coarse":
            return (
                "Identify the main person's action in this stitched image. "
                "Prefer the base/core verb from the existing action vocabulary; avoid synonyms that only change tense (e.g., pour not pouring). "
                "If none fits, generate one concise base-form verb (lemma) and output it. Do not return empty/blank/empty string. "
                "Return JSON with English key/value: {action: XXX}. ONE action, ONE word, NO phrases, NO multiple values. "
                "Lowercase, no brackets, single-line response. If unsure, give your best guess; if truly unknown, use 'unknown' instead of empty. If standing still, use {action: still}. "
                "Deduplicate choices: if a synonym already exists (e.g., pour vs pouring), pick the existing base form. "
                f"Action vocabulary: [{join(vocab.get('action', []))}]"
            )
        else:
            return (
                "Perform structured action recognition for the main person in this stitched image. "
                "Prefer base/core labels from the existing vocabulary; avoid tense/inflection synonyms (e.g., pour not pouring). "
                "If none fits, generate concise new single-word labels in base form. Do not leave any field empty/blank; if truly unknown, use 'unknown' instead of empty. "
                "Return JSON using English keys and English single-word values: {person: XXX, action: XXX, object: XXX, environment: XXX}. "
                "ONE value per key, ONE word each, NO phrases, NO multiple values, lowercase, no brackets, single-line response. Provide your best guess based on the image. "
                "If the person is standing still, set action: still. Deduplicate choices by picking existing base forms when synonyms appear. "
                f"Person vocabulary: [{join(vocab.get('person', []))}] "
                f"Action vocabulary: [{join(vocab.get('action', []))}] "
                f"Object vocabulary: [{join(vocab.get('object', []))}] "
                f"Environment vocabulary: [{join(vocab.get('environment', []))}]"
            )

    def update_vocab_from_results(self, vocab: Dict[str, List[str]], results: List[str], annotation_type: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """根据标准化结果更新词典；返回(更新后的词典, 新增项字典)"""
        added: Dict[str, List[str]] = {"person": [], "action": [], "object": [], "environment": []}

        # 标准化结果是字符串，需要解析为字典
        for resp in results:
            try:
                data = parse_dict_format(resp)
            except Exception:
                data = {}

            # 中文键映射到英文键
            cn_to_en = {
                "人物": "person",
                "动作": "action",
                "物体": "object",
                "环境": "environment",
                # 兼容英文键
                "person": "person",
                "action": "action",
                "object": "object",
                "environment": "environment",
            }

            for cn_key, en_key in cn_to_en.items():
                if cn_key in data:
                    val = str(data.get(cn_key, '')).strip()
                    if not val:
                        continue
                    # 粗略标注仅关注 action
                    if annotation_type == "coarse" and en_key != "action":
                        continue
                    # 加入词典（英文值）
                    lst = vocab.setdefault(en_key, [])
                    if val not in lst:
                        lst.append(val)
                        added[en_key].append(val)

        return vocab, added
    
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
        print(f"\n{'='*70}")
        print(f"  处理视频: {video_id} [{annotation_type.upper()}]")
        print(f"{'='*70}")
        
        video_dir = self.dataset_dir / video_id
        groups_file = video_dir / "groups_of_five.json"
        
        # 检查 groups_of_five.json
        print(f"  [1/4] 检查分组文件...", end=" ")
        if not groups_file.exists():
            print("✗ 缺少 groups_of_five.json")
            return False
        print("✓")
        
        # 检查是否已处理
        output_file = video_dir / f"doubao_results_{video_id}.json"
        print(f"  [2/4] 检查输出文件...", end=" ")
        if output_file.exists():
            print("⊘ 已存在，跳过处理")
            return True
        print("✓ 需要处理")
        
        # 准备参数
        print(f"  [3/4] 准备调用 API...")
        
        # 加载词典并构造提示词
        vocab = self.load_vocab()
        if annotation_type == "coarse":
            print(f"        标注类型: 粗略（仅动作，优先使用词典）")
        else:
            print(f"        标注类型: 精细（人物+动作+物体+环境，优先使用词典）")
        prompt = self.build_prompt(annotation_type, vocab)
        
        print(f"  [4/4] 执行 API 调用...")
        print(f"        {'-'*66}")
        
        try:
            # 初始化 API 客户端
            api_key = "241d5b8f-6403-46ec-8c25-7fcb4eb4e489"
            model = "ep-20260116222301-2pdc2"
            base_url = "https://ark.cn-beijing.volces.com/api/v3"
            
            api_client = DoubaoImageUnderstandingAPI(api_key, model, base_url)
            processor = BatchImageProcessor(api_client)
            
            # 处理文件夹
            results = processor.process_folder(
                str(video_dir),
                prompt,
                output_file=str(output_file),
                temperature=0.5,
                max_tokens=2048,
                delay=1.0
            )
            
            print(f"        {'-'*66}")
            print(f"  ✓ 视频 {video_id} 处理成功")
            print(f"  ✓ 共 {len(results)} 条标准化记录")
            # 根据结果更新词典
            updated_vocab, added = self.update_vocab_from_results(vocab, results, annotation_type)
            # 仅在有新增时保存
            if any(added[k] for k in added):
                self.save_vocab(updated_vocab)
                print("  ✓ 新增词条:")
                for k in ("person", "action", "object", "environment"):
                    if added[k]:
                        print(f"    - {k}: {', '.join(sorted(set(added[k])))}")
            return True
                
        except Exception as e:
            print(f"        {'-'*66}")
            print(f"  ✗ 处理异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_batch(self, fine_grained: List[str], coarse_grained: List[str], dry_run: bool = False) -> None:
        """批量处理视频"""
        self.init_progress()
        progress = self.load_progress()
        
        total = len(fine_grained) + len(coarse_grained)
        processed = 0
        failed = 0
        
        print("\n" + "=" * 70)
        print("批量处理开始".center(70))
        print("=" * 70)
        print(f"  精细标注: {len(fine_grained)} 个")
        print(f"  粗略标注: {len(coarse_grained)} 个")
        print(f"  总计: {total} 个")
        print("=" * 70)
        
        if dry_run:
            print("\n[干运行模式] 只显示计划，不实际处理\n")
        
        # 精细标注
        if fine_grained:
            print(f"\n{'#'*70}")
            print(f"  开始处理精细标注视频 ({len(fine_grained)} 个)")
            print(f"{'#'*70}")
            
            for i, vid in enumerate(fine_grained, 1):
                print(f"\n>>> 进度: [{i}/{len(fine_grained)}] 精细标注")
                if dry_run:
                    print(f"    [DRY RUN] 视频 {vid} - 将使用标准提示词")
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
        if coarse_grained:
            print(f"\n{'#'*70}")
            print(f"  开始处理粗略标注视频 ({len(coarse_grained)} 个)")
            print(f"{'#'*70}")
            
            for i, vid in enumerate(coarse_grained, 1):
                print(f"\n>>> 进度: [{i}/{len(coarse_grained)}] 粗略标注")
                if dry_run:
                    print(f"    [DRY RUN] 视频 {vid} - 将使用简化提示词")
                    processed += 1
                else:
                    if self.process_video(vid, "coarse"):
                        processed += 1
                        progress["coarse_grained"]["completed"].append(vid)
                    else:
                        failed += 1
                        progress["coarse_grained"]["failed"].append(vid)
                    self.save_progress(progress)
        
        # 总结
        print("\n" + "=" * 70)
        print("批量处理完成".center(70))
        print("=" * 70)
        print(f"  总计视频: {total} 个")
        print(f"  成功处理: {processed} 个  ✓")
        print(f"  处理失败: {failed} 个  ✗")
        if total > 0:
            success_rate = processed/total*100
            status = "优秀" if success_rate >= 90 else "良好" if success_rate >= 70 else "一般"
            print(f"  成功率: {success_rate:.1f}% ({status})")
        print("=" * 70 + "\n")


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
