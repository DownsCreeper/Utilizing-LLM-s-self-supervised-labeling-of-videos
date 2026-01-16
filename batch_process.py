"""
批量处理文件夹内的所有图片，使用讯飞星火API进行动作识别
将结果汇总为一个JSON文件
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import time
import re

# 添加api目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.spark_image_api import SparkImageUnderstandingAPI


def parse_json_from_response(response: str) -> Dict[str, Any]:
    """
    从API响应中提取JSON
    
    Args:
        response: API返回的响应文本
        
    Returns:
        解析后的字典，如果解析失败返回原始文本
    """
    # 尝试直接解析JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # 尝试查找JSON格式的内容
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, response)
    
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass
    
    # 如果解析失败，返回原始文本
    return {"raw_response": response}


def process_folder_images(folder_path: str,
                          app_id: str,
                          api_key: str,
                          api_secret: str,
                          output_file: str = None,
                          region: str = "cn-huabei-1",
                          delay: float = 1.0) -> Dict[str, Any]:
    """
    处理文件夹内的所有图片
    
    Args:
        folder_path: 图片文件夹路径
        app_id: 讯飞APPID
        api_key: 讯飞APIKey
        api_secret: 讯飞APISecret
        output_file: 输出JSON文件路径
        region: 服务地区
        delay: 每次请求后的延迟时间（秒）
        
    Returns:
        包含所有结果的字典
    """
    
    # 提示词
    prompt = '请结构化识别这张图片中的动作，返回JSON格式。格式为"{人物，动作，物体，环境}"，如{男人，切，菜，厨房}。如果人物只是站立，无其他动作，请输出{still}。'
    
    # 初始化API客户端
    print(f"初始化讯飞星火API...")
    api_client = SparkImageUnderstandingAPI(app_id, api_key, api_secret, region)
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    image_files = []
    
    folder_path_obj = Path(folder_path)
    for ext in image_extensions:
        image_files.extend(folder_path_obj.glob(f"*{ext}"))
        image_files.extend(folder_path_obj.glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files, key=lambda x: x.name)
    total_images = len(image_files)
    
    if total_images == 0:
        print(f"错误：文件夹 {folder_path} 中没有找到图片文件")
        return {}
    
    print(f"找到 {total_images} 张图片")
    print(f"提示词: {prompt}")
    print(f"开始处理...\n")
    
    # 结果字典
    results = {}
    successful_count = 0
    failed_count = 0
    
    # 处理每张图片
    for i, image_path in enumerate(image_files):
        image_name = image_path.name
        batch_num = i + 1
        
        print(f"[{batch_num}/{total_images}] 处理: {image_name}")
        
        try:
            # 调用API
            response = api_client.call([str(image_path)], prompt, max_tokens=200)
            
            # 解析响应
            parsed_result = parse_json_from_response(response)
            
            # 保存结果
            results[image_name] = {
                "label": parsed_result,
                "raw_response": response,
                "status": "success",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            successful_count += 1
            print(f"  结果: {parsed_result}")
            
        except Exception as e:
            print(f"  错误: {str(e)}")
            results[image_name] = {
                "label": None,
                "error": str(e),
                "status": "failed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            failed_count += 1
        
        # 延迟避免限流
        if batch_num < total_images:
            time.sleep(delay)
        
        print()
    
    # 生成输出文件名
    if output_file is None:
        folder_name = folder_path_obj.name
        output_file = f"labels_{folder_name}.json"
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 统计信息
    print(f"\n{'='*50}")
    print(f"处理完成！")
    print(f"总计: {total_images} 张")
    print(f"成功: {successful_count} 张")
    print(f"失败: {failed_count} 张")
    print(f"结果已保存到: {output_file}")
    print(f"{'='*50}")
    
    return results


def process_multiple_folders(base_path: str,
                             app_id: str,
                             api_key: str,
                             api_secret: str,
                             folder_pattern: str = "00*",
                             region: str = "cn-huabei-1",
                             delay: float = 1.0):
    """
    批量处理多个文件夹
    
    Args:
        base_path: 基础路径
        app_id: 讯飞APPID
        api_key: 讯飞APIKey
        api_secret: 讯飞APISecret
        folder_pattern: 文件夹匹配模式
        region: 服务地区
        delay: 每次请求后的延迟时间（秒）
    """
    
    base_path_obj = Path(base_path)
    folders = sorted([f for f in base_path_obj.glob(folder_pattern) if f.is_dir()])
    
    total_folders = len(folders)
    print(f"找到 {total_folders} 个文件夹")
    print(f"基础路径: {base_path}\n")
    
    all_results = {}
    
    for i, folder in enumerate(folders):
        folder_name = folder.name
        print(f"\n{'#'*50}")
        print(f"处理文件夹 [{i+1}/{total_folders}]: {folder_name}")
        print(f"{'#'*50}\n")
        
        try:
            results = process_folder_images(
                folder_path=str(folder),
                app_id=app_id,
                api_key=api_key,
                api_secret=api_secret,
                output_file=f"labels_{folder_name}.json",
                region=region,
                delay=delay
            )
            
            all_results[folder_name] = {
                "total": len(results),
                "successful": sum(1 for r in results.values() if r.get("status") == "success"),
                "failed": sum(1 for r in results.values() if r.get("status") == "failed"),
                "output_file": f"labels_{folder_name}.json"
            }
            
        except Exception as e:
            print(f"文件夹处理失败: {str(e)}")
            all_results[folder_name] = {
                "error": str(e)
            }
    
    # 保存汇总结果
    summary_file = f"summary_{base_path_obj.name}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n{'='*60}")
    print(f"全部处理完成！")
    print(f"汇总结果保存到: {summary_file}")
    print(f"{'='*60}")


def main():
    """主函数"""
    
    # ==================== 配置区域 ====================
    # 替换为你的讯飞API凭证
    APP_ID = "96dbd267"
    API_KEY = "19391a089226307b3239bbee8e24da2f"
    API_SECRET = "OGVkYWFmMWNkNWIzZGU3NjM0MjI1ZWVm"
    REGION = "cn-huabei-1"
    # =============================================
    
    import argparse
    
    parser = argparse.ArgumentParser(description="批量处理图片进行动作识别")
    parser.add_argument("--folder", type=str, help="要处理的图片文件夹路径")
    parser.add_argument("--base", type=str, help="批量处理的基础路径")
    parser.add_argument("--output", type=str, help="输出JSON文件名")
    parser.add_argument("--delay", type=float, default=1.0, help="请求延迟时间（秒）")
    parser.add_argument("--pattern", type=str, default="00*", help="文件夹匹配模式")
    
    args = parser.parse_args()
    
    if args.folder:
        # 处理单个文件夹
        process_folder_images(
            folder_path=args.folder,
            app_id=APP_ID,
            api_key=API_KEY,
            api_secret=API_SECRET,
            output_file=args.output,
            region=REGION,
            delay=args.delay
        )
    
    elif args.base:
        # 批量处理多个文件夹
        process_multiple_folders(
            base_path=args.base,
            app_id=APP_ID,
            api_key=API_KEY,
            api_secret=API_SECRET,
            folder_pattern=args.pattern,
            region=REGION,
            delay=args.delay
        )
    
    else:
        # 默认示例
        print("使用示例：\n")
        print("1. 处理单个文件夹:")
        print("   python batch_process.py --folder <folder_path> --output labels.json")
        print("\n   例如:")
        print("   python batch_process.py --folder \"d:\\ULLV\\...\\Breakfast\\000001\" --output labels_000001.json")
        
        print("\n2. 批量处理多个文件夹:")
        print("   python batch_process.py --base <base_path> --pattern \"00*\"")
        print("\n   例如:")
        print("   python batch_process.py --base \"d:\\ULLV\\...\\Breakfast\" --pattern \"00*\"")
        
        print("\n3. 调整请求延迟:")
        print("   python batch_process.py --folder <folder_path> --delay 2.0")
        
        print("\n请使用 --help 查看所有参数")


if __name__ == "__main__":
    main()
