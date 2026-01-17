"""
标准化 Doubao LLM 回复的工具

将各种格式的 LLM 回复（JSON、中文冒号、包含括号等）标准化为统一格式：
  "{人物：xxx，动作：xxx，物体：xxx，环境：xxx}"

标准化规则：
1. 删除所有括号及其内容（例如 "拿（含伸臂、持瓶、持袋）" → "拿"）
2. 移除所有 \n 换行符和多余空白
3. 统一为中文冒号和逗号分隔符
4. 保存为纯字符串（无 \" 转义）
"""

import json
import re
from typing import Union, Dict, Any


def remove_parentheses_content(text: str) -> str:
    """
    删除文本中所有括号及其内容
    
    例如：
      "拿（含伸臂、持瓶、持袋）" → "拿"
      "抬手（从抬起至放下的挥动类动作）" → "抬手"
    """
    # 删除所有括号内容（支持圆括号、方括号、花括号）
    text = re.sub(r'[（\(].*?[）\)]', '', text)
    text = re.sub(r'[【\[].*?[】\]]', '', text)
    
    return text.strip()


def normalize_whitespace(text: str) -> str:
    """
    标准化空白符：删除 \n 和多余空格
    """
    # 删除所有换行符和回车符
    text = text.replace('\n', '').replace('\r', '')
    # 删除多余空格（保留单个空格）
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def parse_dict_format(text: str) -> Dict[str, str]:
    """
    解析不同格式的字典文本
    
    支持的格式：
      - JSON: {"人物":"男人","动作":"取"}
      - 中文冒号: {人物：男人，动作：取}
      - 带引号: {"人物":"男人","动作":"取"}
      - 纯字符串: 人物：男人，动作：取
    """
    # 删除引号转义
    text = text.replace('\\"', '"')
    # 标准化空白
    text = normalize_whitespace(text)
    
    # 尝试 JSON 解析
    try:
        # 移除前后的花括号
        text_stripped = text.strip('{}').strip()
        
        # 如果包含 " 或 ':
        if '"' in text or '":' in text:
            # 尝试直接 JSON 解析
            try:
                return json.loads(text)
            except:
                pass
        
        # 尝试用中文冒号解析：{人物：xxx，动作：xxx}
        if '：' in text or '，' in text:
            result = {}
            # 移除引号
            text_stripped = text.strip('{}').strip('"\'')
            
            # 按逗号分割（支持中文和英文逗号）
            pairs = re.split('[,，]', text_stripped)
            for pair in pairs:
                pair = pair.strip()
                if '：' in pair:
                    key, value = pair.split('：', 1)
                    key = key.strip().strip('"\'')
                    value = value.strip().strip('"\'')
                    result[key] = value
                elif ':' in pair:
                    key, value = pair.split(':', 1)
                    key = key.strip().strip('"\'')
                    value = value.strip().strip('"\'')
                    result[key] = value
            
            if result:
                return result
    
    except Exception as e:
        print(f"警告：解析失败 - {text[:50]}... - {str(e)}")
    
    return {}


def standardize_response(response: Union[str, Dict, Any]) -> str:
    """
    将 LLM 回复标准化为统一格式
    
    输入：
      - 字符串（可能是 JSON、纯文本等）
      - 字典
    
    输出：
      "{人物：xxx，动作：xxx，物体：xxx，环境：xxx}"
    
    示例：
      输入：'{"人物":"男人","动作":"拿（含伸臂、持瓶、持袋）","物体":"瓶子、袋子","环境":"厨房"}'
      输出："{人物：男人，动作：拿，物体：瓶子、袋子，环境：厨房}"
    """
    
    # 1. 转换为字典
    if isinstance(response, str):
        data = parse_dict_format(response)
    elif isinstance(response, dict):
        data = response
    else:
        return ""
    
    # 2. 标准化各字段
    # 定义标准字段（中文键）
    standard_keys = {
        '人物': ['character', 'person', '人物'],
        '动作': ['action', 'motion', '动作'],
        '物体': ['object', 'thing', '物体'],
        '环境': ['environment', 'scene', '环境']
    }
    
    # 归一化键名和值
    normalized = {}
    for cn_key, variants in standard_keys.items():
        for variant in variants:
            if variant in data:
                value = str(data[variant]).strip()
                
                # 删除括号内容
                value = remove_parentheses_content(value)
                
                # 删除不必要的引号
                value = value.strip('\"\'').strip()
                
                # 删除所有 \n 和多余空白
                value = normalize_whitespace(value)
                
                normalized[cn_key] = value
                break
        
        # 如果找不到该字段，使用默认值
        if cn_key not in normalized:
            normalized[cn_key] = ""
    
    # 3. 格式化输出
    # 按顺序拼接：人物，动作，物体，环境
    result = "{"
    result += "人物：" + normalized.get('人物', '')
    result += "，动作：" + normalized.get('动作', '')
    result += "，物体：" + normalized.get('物体', '')
    result += "，环境：" + normalized.get('环境', '')
    result += "}"
    
    return result


def standardize_batch(responses: list) -> list:
    """
    批量标准化多个回复
    
    输入：
      [response1, response2, ...]
    
    输出：
      [standardized1, standardized2, ...]
    """
    return [standardize_response(resp) for resp in responses]


# 测试用例
if __name__ == "__main__":
    test_cases = [
        '{"人物":"男人","动作":"拿（含伸臂、持瓶、持袋）","物体":"瓶子、袋子","环境":"厨房"}',
        '{人物：男人，动作：取，物体：厨具，环境：厨房}',
        '{\n"人物":"男人",\n"动作":"拿、抬手、开",\n"物体":"杯子、柜子",\n"环境":"厨房"\n}',
        '{"人物":"男人","动作":"抬手（从抬起至放下的挥动类动作）","物体":"无","环境":"厨房"}',
        "{人物：男人，动作：倒，物体：液体，环境：厨房}",
    ]
    
    print("标准化测试:")
    print("=" * 70)
    for test in test_cases:
        result = standardize_response(test)
        print(f"输入：  {test[:60]}...")
        print(f"输出：  {result}")
        print("-" * 70)
