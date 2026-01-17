"""
豆包图片理解API - 基于火山方舟平台
支持使用OpenAI兼容的SDK进行图片理解
支持命令行参数配置处理任务
支持标准化保存标注结果
"""

import argparse
import base64
import json
import os
import time
import _thread as thread
from datetime import datetime
from typing import List, Dict, Any, Callable
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI

from standardize_response import standardize_response, standardize_batch


class DoubaoImageUnderstandingAPI:
    """豆包图片理解API客户端"""
    
    def __init__(self, api_key: str, model: str = "doubao-1-5-thinking-vision-pro-250428", 
                 base_url: str = "https://ark.cn-beijing.volces.com/api/v3"):
        """
        初始化Doubao图片理解API客户端
        
        Args:
            api_key: 火山方舟API Key
            model: 模型名称/推理接入点ID
            base_url: API基础URL
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
    def _prepare_image_content(self, image_path: str) -> Dict[str, Any]:
        """
        准备图片内容，支持URL和本地文件
        
        Args:
            image_path: 图片路径或URL
            
        Returns:
            OpenAI格式的图片内容字典
        """
        if image_path.startswith('http://') or image_path.startswith('https://'):
            # URL图片
            return {
                "type": "image_url",
                "image_url": {
                    "url": image_path
                }
            }
        else:
            # 本地文件，转换为base64
            with open(image_path, 'rb') as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # 判断图片格式
            ext = Path(image_path).suffix.lower()
            mime_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.webp': 'image/webp'
            }
            mime_type = mime_type_map.get(ext, 'image/jpeg')
            
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}"
                }
            }
    
    def call(self, images: List[str], prompt: str, **kwargs) -> str:
        """
        调用图片理解API
        
        Args:
            images: 图片路径或URL列表
            prompt: 提示词
            **kwargs: 其他参数 (temperature, max_tokens等)
            
        Returns:
            API响应文本
        """
        # 构建消息内容
        content = []
        
        # 添加所有图片
        for image_path in images:
            content.append(self._prepare_image_content(image_path))
        
        # 添加文本提示
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        # 提取参数
        temperature = kwargs.get("temperature", 0.5)
        max_tokens = kwargs.get("max_tokens", 2048)
        top_p = kwargs.get("top_p", 0.95)
        
        # 调用API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
            
            # 提取响应文本
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"API调用失败: {str(e)}")
    
    def call_async(self, 
                   images: List[str], 
                   prompt: str, 
                   callback: Callable[[str], None] = None,
                   **kwargs) -> None:
        """
        异步调用API
        
        Args:
            images: 图片路径或URL列表
            prompt: 提示词
            callback: 完成后的回调函数，接收响应文本
            **kwargs: 其他参数
        """
        def _run_async():
            try:
                response = self.call(images, prompt, **kwargs)
                if callback:
                    callback(response)
            except Exception as e:
                if callback:
                    callback(f"Error: {str(e)}")
        
        thread.start_new_thread(_run_async, ())
    
    def call_stream(self, images: List[str], prompt: str, **kwargs) -> None:
        """
        流式调用API（如果支持）
        
        Args:
            images: 图片路径或URL列表
            prompt: 提示词
            **kwargs: 其他参数
        """
        # 构建消息内容
        content = []
        
        # 添加所有图片
        for image_path in images:
            content.append(self._prepare_image_content(image_path))
        
        # 添加文本提示
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        # 提取参数
        temperature = kwargs.get("temperature", 0.5)
        max_tokens = kwargs.get("max_tokens", 2048)
        top_p = kwargs.get("top_p", 0.95)
        
        # 流式调用
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end='', flush=True)
            print()  # 换行
            
        except Exception as e:
            raise Exception(f"流式API调用失败: {str(e)}")


class BatchImageProcessor:
    """批量处理图片的类"""
    
    def __init__(self, api_client: DoubaoImageUnderstandingAPI):
        """
        初始化批量处理器
        
        Args:
            api_client: DoubaoImageUnderstandingAPI实例
        """
        self.api_client = api_client
        self.results = []
    
    def _concat_images(self, image_paths: List[str], temp_dir: Path, group_idx: int) -> str:
        """将多张图片水平拼接为一张，并保存到临时目录，返回路径。"""
        images = []
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(f"无法读取图片: {p}")
            images.append(img)

        # 统一高度
        target_h = min(img.shape[0] for img in images)
        resized = []
        for img in images:
            h, w = img.shape[:2]
            if h != target_h:
                scale = target_h / h
                img = cv2.resize(img, (int(w * scale), target_h))
            resized.append(img)

        concat = np.hstack(resized)
        temp_path = temp_dir / f"group_{group_idx:06d}.jpg"
        cv2.imwrite(str(temp_path), concat)
        return str(temp_path)

    def process_folder(self,
                      folder_path: str,
                      prompt: str,
                      output_file: str = None,
                      delay: float = 1.0,
                      **kwargs) -> List[str]:
        """
        基于 groups_of_five.json 将 5 张拼 1 张发给模型，结果再展开到 5 张。

        folder_path: 对应 数据集_x 的某个视频目录，例如 .../数据集_5/000001
        需要存在 groups_of_five.json，由 process_frames.py 生成。
        """
        groups_file = Path(folder_path) / "groups_of_five.json"
        if not groups_file.exists():
            raise FileNotFoundError(f"找不到分组文件: {groups_file}")

        with open(groups_file, 'r', encoding='utf-8') as f:
            groups = json.load(f)

        # 路径补全
        groups_full = [[str(Path(folder_path) / img) for img in group] for group in groups]
        total_groups = len(groups_full)
        total_frames = sum(len(g) for g in groups_full)
        print(f"找到 {total_frames} 张图片，{total_groups} 组（每组最多5张）")

        self.results = []  # 顺序与帧顺序一致

        temp_dir = Path(folder_path) / "_concat_temp"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            for gi, group in enumerate(groups_full):
                print(f"\n处理分组 [{gi+1}/{total_groups}]，包含 {len(group)} 张")

                # 拼接
                concat_path = self._concat_images(group, temp_dir, gi)

                # 调用模型
                try:
                    response = self.api_client.call([concat_path], prompt, **kwargs)
                    print(f"回复: {response[:120]}...")
                except Exception as e:
                    response = f"Error: {str(e)}"
                    print(response)

                # 展开到 5 张（或不足 5 张的实际数量）
                self.results.extend([response] * len(group))

                if gi < total_groups - 1:
                    time.sleep(delay)

        finally:
            # 清理临时拼接文件
            for f in temp_dir.glob("group_*.jpg"):
                try:
                    os.remove(f)
                except OSError:
                    pass
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass

        if len(self.results) != total_frames:
            print("警告：结果数量与帧数不一致")

        # 标准化结果
        print("\n开始标准化回复...")
        standardized_results = standardize_batch(self.results)
        
        # 保存结果（按帧顺序的列表）
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(standardized_results, f, ensure_ascii=False, indent=2)
            print(f"✓ 结果已保存到: {output_file}")
            print(f"✓ 共 {len(standardized_results)} 条标准化记录")

        return standardized_results
    
    def process_batch(self,
                     image_list: List[str],
                     prompt: str,
                     output_file: str = None,
                     delay: float = 0.01,
                     **kwargs) -> List[Dict[str, Any]]:
        """
        处理图片列表
        
        Args:
            image_list: 图片路径列表
            prompt: 提示词
            output_file: 保存结果的JSON文件
            delay: 每次请求之间的延迟（秒）
            **kwargs: 其他参数
            
        Returns:
            处理结果列表
        """
        self.results = []
        total = len(image_list)
        
        for i, image_path in enumerate(image_list):
            batch_num = i + 1
            print(f"\n处理 [{batch_num}/{total}] {Path(image_path).name}")
            
            try:
                response = self.api_client.call([image_path], prompt, **kwargs)
                
                result = {
                    "index": batch_num,
                    "image": image_path,
                    "prompt": prompt,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.results.append(result)
                print(f"回复: {response[:100]}...")
                
                # 延迟避免限流
                if i < total - 1:  # 最后一张不需要延迟
                    time.sleep(delay)
                
            except Exception as e:
                print(f"处理失败: {str(e)}")
                self.results.append({
                    "index": batch_num,
                    "image": image_path,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # 保存结果
        if output_file:
            # 为 process_batch 的每个结果标准化 response 字段
            for result in self.results:
                if "response" in result:
                    result["response"] = standardize_response(result["response"])
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"\n✓ 结果已保存到: {output_file}")
            print(f"✓ 共 {len(self.results)} 条记录（已标准化）")
        
        return self.results


def main():
    """命令行工具：使用 Doubao API 处理视频帧"""
    
    # ==================== 命令行参数配置 ====================
    parser = argparse.ArgumentParser(
        description="使用 Doubao 大模型处理视频帧，支持 5 张拼接 1 张后调用 API，再展开结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python doubao_image_api.py --dataset-name breakfast --sample-interval 5 --video-id 000003
  python doubao_image_api.py --dataset-name youcook --sample-interval 10 --video-id 001 --api-key YOUR_KEY
        """
    )
    
    # 数据集相关参数
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='breakfast',
        help='数据集名称前缀 (默认: breakfast)'
    )
    
    parser.add_argument(
        '--sample-interval',
        type=int,
        default=5,
        help='采样间隔 (默认: 5)'
    )
    
    parser.add_argument(
        '--video-id',
        type=str,
        default='000003',
        help='视频 ID (默认: 000003)'
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        default=r'd:\ULLV\Utilizing-LLM-s-self-supervised-labeling-of-videos\dataset',
        help='数据集基础目录 (默认: d:\\ULLV\\...\\dataset)'
    )
    
    # API 相关参数
    parser.add_argument(
        '--api-key',
        type=str,
        default='241d5b8f-6403-46ec-8c25-7fcb4eb4e489',
        help='Doubao API Key'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='ep-20260116222301-2pdc2',
        help='Doubao 模型 ID'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        default='https://ark.cn-beijing.volces.com/api/v3',
        help='API 服务地址'
    )
    
    # 处理参数
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.5,
        help='温度参数 (默认: 0.5)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=2048,
        help='最大输出 token 数 (默认: 2048)'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='API 调用间隔（秒） (默认: 1.0)'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        default='请结构化识别这张拼接图片中的主体人物动作，返回JSON格式。格式为"{人物，动作，物体，环境}"，如{人物：男人，动作：切，物体：菜，环境：厨房}。标签简洁，不要加括号，多行不要，用中文键。若只是站立无动作，输出{still}。',
        help='处理提示词'
    )
    
    # 输出参数
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='输出 JSON 文件路径（默认：<folder_path>/doubao_results_<video_id>.json）'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='安静模式，减少输出'
    )
    # =====================================================
    
    args = parser.parse_args()
    
    # 构建路径
    base_dir = Path(args.base_dir) / args.dataset_name
    folder_path = base_dir / f"{args.dataset_name}_{args.sample_interval}" / args.video_id
    
    # 验证输入目录
    if not folder_path.exists():
        print(f"错误：输入目录不存在: {folder_path}")
        return 1
    
    # 检查 groups_of_five.json 是否存在
    groups_file = folder_path / "groups_of_five.json"
    if not groups_file.exists():
        print(f"错误：找不到分组索引文件: {groups_file}")
        print(f"请先运行 tool/process_frames.py 生成分组数据")
        return 1
    
    # 确定输出文件
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = str(folder_path / f"doubao_results_{args.video_id}.json")
    
    if not args.quiet:
        print("=" * 60)
        print("Doubao 图像处理工具")
        print("=" * 60)
        print(f"数据集: {args.dataset_name}")
        print(f"采样间隔: {args.sample_interval}")
        print(f"视频 ID: {args.video_id}")
        print(f"输入目录: {folder_path}")
        print(f"输出文件: {output_file}")
        print(f"模型: {args.model}")
        print("-" * 60)
        print("按分组处理（5 张拼 1 张）\n")
    
    try:
        # 初始化 API 客户端
        api_client = DoubaoImageUnderstandingAPI(args.api_key, args.model, args.base_url)
        
        # 初始化批处理器
        processor = BatchImageProcessor(api_client)
        
        # 处理文件夹
        results = processor.process_folder(
            str(folder_path),
            args.prompt,
            output_file=output_file,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            delay=args.delay
        )
        
        if not args.quiet:
            print(f"\n✓ 处理完成，共 {len(results)} 条结果（按帧展开）")
            print(f"✓ 结果已保存到: {output_file}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"错误：文件不存在 - {str(e)}")
        return 1
    except Exception as e:
        print(f"错误：处理失败 - {str(e)}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
