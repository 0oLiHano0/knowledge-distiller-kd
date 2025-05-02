"""
工具函数模块。

此模块提供项目中使用的通用工具函数。
"""

# [DEPENDENCIES]
# 1. Python Standard Library: logging, re, pathlib
# 2. 需要安装：mistune # 用于 Markdown 解析
# 3. 同项目模块: constants (使用绝对导入)

import logging
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable
import mistune # 确保已安装
from knowledge_distiller_kd.core import constants
from knowledge_distiller_kd.core.error_handler import (
    FileOperationError,
    handle_error,
    safe_file_operation
)

# --- 配置日志记录 ---
def setup_logger(log_level: int = logging.INFO) -> logging.Logger:
    """
    配置并返回一个 logger 实例。

    Args:
        log_level (int): 日志级别，默认为 logging.INFO

    Returns:
        logging.Logger: 配置好的 logger 实例
    """
    logger = logging.getLogger('KDToolLogger')
    # 如果 logger 已经有 handlers，假设它已经被配置过，直接返回
    if logger.handlers:
        # 如果需要根据新的 log_level 调整现有 handler 的级别
        for handler in logger.handlers:
            handler.setLevel(log_level)
        logger.setLevel(log_level)
        return logger

    logger.setLevel(log_level) # 设置 logger 的级别

    # 创建一个控制台处理器 (handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level) # 设置 handler 的级别

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)

    # 将处理器添加到 logger
    logger.addHandler(console_handler)

    # 防止日志消息传播到根 logger (如果根 logger 有自己的 handler)
    logger.propagate = False

    return logger

# 在模块加载时获取 logger 实例
logger = setup_logger()

# --- 辅助函数 ---

def create_decision_key(file_path: Union[str, Path], block_index: Union[int, str], block_type: str) -> str:
    """
    为内容块创建唯一的决策键。

    Args:
        file_path (Union[str, Path]): 文件路径
        block_index (Union[int, str]): 块的索引
        block_type (str): 块的类型

    Returns:
        str: 格式为 "file_path::block_index::block_type" 的唯一键

    Raises:
        Exception: 当路径解析失败时
    """
    # 确保 file_path 是字符串形式，并且是绝对路径以保证唯一性
    try:
        file_path_str = str(Path(file_path).resolve())
    except Exception: # 处理可能的路径错误
        file_path_str = str(file_path) # 回退到原始字符串

    # 确保 block_index 是字符串
    index_str = str(block_index)
    type_str = str(block_type)

    # 使用常量中定义的分隔符
    key = f"{file_path_str}{constants.DECISION_KEY_SEPARATOR}{index_str}{constants.DECISION_KEY_SEPARATOR}{type_str}"
    return key

def parse_decision_key(key: str) -> Tuple[Optional[str], Optional[Union[int, str]], Optional[str]]:
    """
    从决策键解析出文件路径、块索引和块类型。

    Args:
        key (str): 决策键字符串

    Returns:
        Tuple[Optional[str], Optional[Union[int, str]], Optional[str]]: 
            (文件路径, 块索引, 块类型) 的元组，解析失败时返回 (None, None, None)

    Example:
        >>> parse_decision_key("/path/to/file::1::paragraph")
        ("/path/to/file", 1, "paragraph")
    """
    try:
        parts = key.split(constants.DECISION_KEY_SEPARATOR)
        if len(parts) < 3:
            logger.error(f"决策键格式不正确，部分太少: {key}")
            return None, None, None

        # 类型是最后一部分
        type_str = parts[-1]
        # 索引是倒数第二部分
        index_str = parts[-2]
        # 文件路径是前面所有部分的组合 (以防路径本身包含分隔符)
        path_str = constants.DECISION_KEY_SEPARATOR.join(parts[:-2])

        # 尝试将索引转换为整数，如果失败则保持为字符串
        try:
            index_val = int(index_str)
        except ValueError:
            index_val = index_str # 保持原样，例如 "1_0"

        return path_str, index_val, type_str

    except Exception as e:
        logger.error(f"无法解析决策键: {key} - {e}", exc_info=True)
        return None, None, None

def extract_text_from_children(children: Optional[List[Dict[str, Any]]]) -> str:
    """
    从 mistune 令牌子项中递归提取文本内容。

    Args:
        children (Optional[List[Dict[str, Any]]]): mistune 解析出的子令牌列表

    Returns:
        str: 提取出的纯文本内容

    Note:
        处理以下类型的令牌：
        - text: 纯文本
        - codespan: 代码片段
        - link/image: 链接和图片的文本描述
        - emphasis/strong/strikethrough: 强调文本
        - softbreak/linebreak: 换行符
        - inline_html: 内联 HTML（当前忽略）
    """
    text = ""
    if children is None:
        return ""
    for child in children:
        child_type = child.get('type')
        if child_type == 'text':
            text += child.get('raw', '')
        elif child_type == 'codespan':
            # 保留代码片段的原始标记
            text += child.get('raw', '')
        elif child_type in ['link', 'image']:
             # 对于链接和图片，提取其文本描述部分
             text += extract_text_from_children(child.get('children', []))
        elif child_type in ['emphasis', 'strong', 'strikethrough']:
            # 对于强调、加粗、删除线，提取其内部文本
            text += extract_text_from_children(child.get('children', []))
        elif child_type == 'softbreak' or child_type == 'linebreak':
             # 将换行符转换为空格，以模拟阅读时的连续性
             text += ' '
        elif child_type == 'inline_html':
             # 通常忽略内联 HTML 标签本身，但可以根据需要提取内容
             pass # 或者提取 child.get('raw', '')

    # 移除多余的空格并将多个空格合并为一个
    text = ' '.join(text.split())
    return text

def display_block_preview(text: str, max_len: int = 80) -> str:
    """
    生成用于控制台显示的块内容预览。

    Args:
        text (str): 原始文本内容
        max_len (int): 预览文本的最大长度，默认为80

    Returns:
        str: 处理后的预览文本，如果超过最大长度会添加省略号
    """
    if not text:
        return "[空内容]"

    # 替换换行符为空格，并移除首尾空格
    preview = text.replace('\n', ' ').strip()
    
    # 如果预览文本过长，进行截断
    if len(preview) > max_len:
        return preview[:max_len-3] + "..."
    
    # 如果文本太短，添加长度信息
    if len(preview) < 20:
        return f"{preview} [长度: {len(text)}字符]"
    
    return preview

def get_markdown_parser() -> mistune.Markdown:
    """
    创建并返回一个配置好的 mistune Markdown 解析器实例。

    Returns:
        mistune.Markdown: 配置好的 Markdown 解析器实例

    Note:
        启用了以下 Markdown 插件：
        - strikethrough: 删除线
        - footnotes: 脚注
        - table: 表格
        - list: 列表（默认启用）
    """
    # 创建一个新的解析器实例
    renderer = mistune.HTMLRenderer()
    markdown = mistune.Markdown(renderer)
    return markdown

def sort_blocks_key(block_info: Tuple[Union[str, Path], Union[int, str], str, str]) -> Tuple[str, Union[int, float], Union[int, str]]:
    """
    为块信息提供排序键，确保块按原始文件中的顺序排列。

    Args:
        block_info (Tuple[Union[str, Path], Union[int, str], str, str]): 
            包含 (文件路径, 索引, 类型, 内容) 的块信息元组

    Returns:
        Tuple[str, Union[int, float], Union[int, str]]: 
            用于排序的键，格式为 (文件路径, 主索引, 子索引)

    Note:
        处理以下索引格式：
        - 普通整数索引
        - 列表项的 '主_子' 格式索引
        - 其他无法解析的索引格式（会被放在排序末尾）
    """
    file_path, index_val, block_type, _ = block_info
    # 使用文件路径的字符串形式进行排序的第一级比较
    file_path_str = str(file_path)

    if isinstance(index_val, str) and '_' in index_val:
        try:
            # 尝试按数字分割和排序
            main_idx, sub_idx = map(int, index_val.split('_'))
            return (file_path_str, main_idx, sub_idx)
        except ValueError:
             # 如果分割或转换失败，使用字符串排序作为后备
             # 将无法解析的放在后面 (使用 float('inf'))
             return (file_path_str, float('inf'), index_val)
    try:
        # 对于普通整数索引
        return (file_path_str, int(index_val), 0)
    except ValueError:
         # 如果索引不是整数也不是 '主_子' 格式，使用字符串排序
         # 将无法解析的放在后面
         return (file_path_str, float('inf'), index_val)

def read_file_content(file_path: Path) -> str:
    """
    读取文件内容。

    Args:
        file_path: 文件路径

    Returns:
        文件内容字符串

    Raises:
        FileOperationError: 如果文件读取失败
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise FileOperationError(f"读取文件 {file_path} 失败: {str(e)}")

def save_decisions(decisions: Dict[str, str], file_path: Path) -> None:
    """
    保存决策到文件。

    Args:
        decisions: 决策字典
        file_path: 保存路径

    Raises:
        FileOperationError: 如果保存失败
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(decisions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise FileOperationError(f"保存决策到文件 {file_path} 失败: {str(e)}")

def load_decisions(file_path: Path) -> Dict[str, str]:
    """
    从文件加载决策。

    Args:
        file_path: 决策文件路径

    Returns:
        决策字典

    Raises:
        FileOperationError: 如果加载失败
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise FileOperationError(f"从文件 {file_path} 加载决策失败: {str(e)}")
