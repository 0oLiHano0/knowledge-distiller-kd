# [DEPENDENCIES]
# 1. Python Standard Library: logging, re, pathlib
# 2. 需要安装：mistune # 用于 Markdown 解析
# 3. 同项目模块: constants (使用绝对导入)

import logging
import re
from pathlib import Path
import mistune # 确保已安装
import constants # 使用绝对导入

# --- 配置日志记录 ---
def setup_logger(log_level=logging.INFO):
    """配置并返回一个 logger 实例。"""
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

def create_decision_key(file_path, block_index, block_type):
    """为块创建唯一的字符串键。"""
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

def parse_decision_key(key):
    """
    从键解析出文件路径字符串、块索引和块类型。
    返回 (path_str, index_val, type_str) 或 (None, None, None) 如果解析失败。
    索引可能是 int 或 str。
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

def extract_text_from_children(children):
    """
    (来自原代码) 辅助函数，用于从 mistune 令牌子项中递归提取文本。
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
        # 可以根据需要添加对其他内联类型的处理
        # else:
        #     logger.debug(f"Unhandled inline token type: {child_type}")

    # 移除多余的空格并将多个空格合并为一个
    text = ' '.join(text.split())
    return text


def display_block_preview(text, max_len=80):
    """生成用于控制台显示的块预览。"""
    # 替换换行符为空格，并移除首尾空格
    preview = text.replace('\n', ' ').strip()
    # 如果预览文本过长，进行截断
    if len(preview) > max_len:
        return preview[:max_len-3] + "..."
    return preview

def get_markdown_parser():
    """创建并返回一个配置好的 mistune Markdown 解析器实例。"""
    # 启用常用的 Markdown 插件
    # renderer=None 表示我们只需要 AST (抽象语法树)，不需要渲染成 HTML
    return mistune.create_markdown(
        renderer=None,
        plugins=['strikethrough', 'footnotes', 'table', 'url', 'task_lists']
    )

def sort_blocks_key(block_info):
    """
    为块信息提供排序键，确保块按原始文件中的顺序排列。
    处理普通整数索引和列表项的 '主_子' 索引。
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
