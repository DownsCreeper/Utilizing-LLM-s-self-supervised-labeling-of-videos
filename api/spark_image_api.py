"""
讯飞星火图片理解API - 参考官方实现
支持使用WebSocket连接进行图片理解
"""

import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from websocket import WebSocketApp
import time
from typing import List, Dict, Any, Callable
from pathlib import Path


class SparkImageUnderstandingAPI:
    """讯飞星火图片理解API客户端"""
    
    def __init__(self, app_id: str, api_key: str, api_secret: str, region: str = "cn-huabei-1"):
        """
        初始化Spark图片理解API客户端
        
        Args:
            app_id: 讯飞APPID
            api_key: 讯飞APIKey
            api_secret: 讯飞APISecret
            region: 地区代码 ("cn-huabei-1", "cn-huadong-1" 等)
        """
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.region = region
        
        # WebSocket URL
        self.image_understanding_url = f"wss://spark-api.{region}.xf-yun.com/v2.1/image"
        
        # 响应存储
        self.response_text = ""
        self.ws = None
        self.finished = False
        
    def _create_auth_url(self) -> str:
        """
        生成带认证信息的WebSocket URL
        参考官方实现
        """
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        
        # 解析URL获取host和path
        url_obj = urlparse(self.image_understanding_url)
        host = url_obj.netloc
        path = url_obj.path
        
        # 拼接签名字符串
        signature_origin = f"host: {host}\n"
        signature_origin += f"date: {date}\n"
        signature_origin += f"GET {path} HTTP/1.1"
        
        # 进行hmac-sha256加密
        signature_sha = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_origin.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        
        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')
        
        # 构建Authorization头
        authorization_origin = (
            f'api_key="{self.api_key}", '
            f'algorithm="hmac-sha256", '
            f'headers="host date request-line", '
            f'signature="{signature_sha_base64}"'
        )
        
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        
        # 构建最终URL
        auth_params = {
            "authorization": authorization,
            "date": date,
            "host": host
        }
        
        final_url = self.image_understanding_url + '?' + urlencode(auth_params)
        return final_url
    
    def _gen_params(self, text: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        生成请求参数
        """
        return {
            "header": {
                "app_id": self.app_id
            },
            "parameter": {
                "chat": {
                    "domain": kwargs.get("domain", "imagev3"),
                    "temperature": kwargs.get("temperature", 0.5),
                    "top_k": kwargs.get("top_k", 4),
                    "max_tokens": kwargs.get("max_tokens", 2048),
                    "auditing": "default"
                }
            },
            "payload": {
                "message": {
                    "text": text
                }
            }
        }
    
    def _on_message(self, ws, message: str):
        """处理WebSocket消息"""
        data = json.loads(message)
        code = data['header']['code']
        
        if code != 0:
            print(f'请求错误: {code}, {data}')
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            self.response_text += content
            
            # 状态为2表示消息接收完成
            if status == 2:
                ws.close()
                self.finished = True
    
    def _on_error(self, ws, error):
        """处理WebSocket错误"""
        print(f"### WebSocket error: {error}")
    
    def _on_close(self, ws, one, two):
        """处理WebSocket关闭"""
        pass
    
    def _on_open(self, ws):
        """处理WebSocket连接建立"""
        # 在后台线程中发送数据
        thread.start_new_thread(self._run, (ws,))
    
    def _run(self, ws):
        """发送请求数据"""
        data = json.dumps(self.request_data)
        ws.send(data)
    
    def call(self, images: List[str], prompt: str, **kwargs) -> str:
        """
        调用图片理解API
        
        Args:
            images: 图片路径或URL列表
            prompt: 提示词
            **kwargs: 其他参数 (domain, temperature, max_tokens等)
            
        Returns:
            API响应文本
        """
        # 构建消息文本
        text = []
        
        # 添加图片
        for image_path in images:
            if image_path.startswith('http'):
                # URL图片
                image_base64 = image_path
            else:
                # 本地文件
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            text.append({
                "role": "user",
                "content": image_base64,
                "content_type": "image"
            })
        
        # 添加文本提示
        text.append({
            "role": "user",
            "content": prompt,
            "content_type": "text"
        })
        
        # 生成请求参数
        self.request_data = self._gen_params(text, **kwargs)
        
        # 重置响应
        self.response_text = ""
        self.finished = False
        
        # 创建WebSocket连接
        ws_url = self._create_auth_url()
        self.ws = WebSocketApp(
            ws_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        
        # 运行WebSocket（阻塞）
        self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        
        return self.response_text
    
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
            response = self.call(images, prompt, **kwargs)
            if callback:
                callback(response)
        
        thread.start_new_thread(_run_async, ())


class BatchImageProcessor:
    """批量处理图片的类"""
    
    def __init__(self, api_client: SparkImageUnderstandingAPI):
        """
        初始化批量处理器
        
        Args:
            api_client: SparkImageUnderstandingAPI实例
        """
        self.api_client = api_client
        self.results = []
    
    def process_folder(self,
                      folder_path: str,
                      prompt: str,
                      batch_size: int = 1,
                      output_file: str = None,
                      **kwargs) -> List[Dict[str, Any]]:
        """
        批量处理文件夹中的图片
        
        Args:
            folder_path: 图片文件夹路径
            prompt: 提示词
            batch_size: 批量大小（通常设为1，因为WebSocket一次一个）
            output_file: 保存结果的JSON文件
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
                time.sleep(1)
                
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
                     **kwargs) -> List[Dict[str, Any]]:
        """
        处理图片列表
        
        Args:
            image_list: 图片路径列表
            prompt: 提示词
            output_file: 保存结果的JSON文件
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
                time.sleep(1)
                
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
    app_id = "96dbd267"
    api_key = "19391a089226307b3239bbee8e24da2f"
    api_secret = "OGVkYWFmMWNkNWIzZGU3NjM0MjI1ZWVm"
    region = "cn-huabei-1"  # 或 "cn-huadong-1"
    # =============================================
    
    # 初始化API客户端
    api_client = SparkImageUnderstandingAPI(app_id, api_key, api_secret, region)
    
    # 示例1：处理单张图片
    print("=== 处理单张图片 ===")
    try:
        image_path = r"d:\ULLV\Utilizing-LLM-s-self-supervised-labeling-of-videos\dataset\breakfast\Breakfast\000001\000001.jpg"
        prompt = "请结构化识别这张图片中的动作，返回JSON格式。格式为“{人物，动作，物体，环境}”，如{男人，切，菜，厨房}。如果人物只是站立，无其他动作，请输出{still}。"
        
        # 取消注释以运行
        response = api_client.call([image_path], prompt)
        print(f"回复: {response}")
    except FileNotFoundError:
        print("图片文件不存在，请修改路径")

if __name__ == "__main__":
    main()
