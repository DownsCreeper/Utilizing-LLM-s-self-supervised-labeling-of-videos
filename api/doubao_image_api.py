"""
豆包图片理解API - 基于火山方舟平台
支持使用OpenAI兼容的SDK进行图片理解
"""

import base64
import json
import time
import _thread as thread
from datetime import datetime
from typing import List, Dict, Any, Callable
from pathlib import Path
from openai import OpenAI


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
    
    def process_folder(self,
                      folder_path: str,
                      prompt: str,
                      batch_size: int = 1,
                      output_file: str = None,
                      delay: float = 1.0,
                      **kwargs) -> List[Dict[str, Any]]:
        """
        批量处理文件夹中的图片
        
        Args:
            folder_path: 图片文件夹路径
            prompt: 提示词
            batch_size: 批量大小（通常设为1）
            output_file: 保存结果的JSON文件
            delay: 每次请求之间的延迟（秒）
            **kwargs: 其他参数
            
        Returns:
            处理结果列表
        """
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f"*{ext}"))
        
        image_files = sorted([str(f) for f in image_files])
        total_images = len(image_files)
        
        print(f"找到 {total_images} 张图片")
        
        self.results = []
        
        # 按批处理
        for i, image_path in enumerate(image_files):
            batch_num = i + 1
            print(f"\n处理 [{batch_num}/{total_images}] {Path(image_path).name}")
            
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
                if i < total_images - 1:  # 最后一张不需要延迟
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
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {output_file}")
        
        return self.results
    
    def process_batch(self,
                     image_list: List[str],
                     prompt: str,
                     output_file: str = None,
                     delay: float = 1.0,
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
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {output_file}")
        
        return self.results


def main():
    """使用示例"""
    
    # ==================== 配置区域 ====================
    # 从环境变量获取API Key，或直接填写
    api_key = "241d5b8f-6403-46ec-8c25-7fcb4eb4e489"
    model = "ep-20260116222301-2pdc2"
    base_url = "https://ark.cn-beijing.volces.com/api/v3"
    # =============================================
    
    # 初始化API客户端
    api_client = DoubaoImageUnderstandingAPI(api_key, model, base_url)
    
    # 处理单张图片
    # try:
    #     image_path = r"d:\ULLV\Utilizing-LLM-s-self-supervised-labeling-of-videos\dataset\breakfast\Breakfast\000001\000001.jpg"
    #     prompt = "请结构化识别这张图片中的单个主体人物的动作，返回JSON格式。格式为\"{人物，动作，物体，环境}\"，如{人物：男人，动作：切，物体：菜，环境：厨房}。标签尽量简略。如果人物只是站立，无其他动作，请输出{still}。"
        
    #     # 取消注释以运行
    #     response = api_client.call([image_path], prompt)
    #     print(f"回复: {response}")
    # except FileNotFoundError:
    #     print("图片文件不存在，请修改路径")
    
    # 批量处理文件夹
    print("\n=== 批量处理文件夹 ===")
    try:
        folder_path = r"d:\ULLV\Utilizing-LLM-s-self-supervised-labeling-of-videos\dataset\breakfast\breakfast_cat_5\000001"
        prompt = "请结构化识别这张图片中的单个主体人物的动作，返回JSON格式。格式为\"{人物，动作，物体，环境}\"，如{人物：男人，动作：切，物体：菜，环境：厨房}。标签尽量简略，不要添加括号，一行内输出。如果人物只是站立，无其他动作，请输出{still}。"
        output_file = "doubao_results.json"
        
        processor = BatchImageProcessor(api_client)
        
        results = processor.process_folder(
            folder_path, 
            prompt, 
            output_file=output_file,
            temperature=0.5,
            max_tokens=2048,
            delay=1.0  # 每次请求间隔1秒
        )
        print(f"\n处理完成，共 {len(results)} 张图片")
    except Exception as e:
        print(f"批量处理失败: {str(e)}")


if __name__ == "__main__":
    main()
