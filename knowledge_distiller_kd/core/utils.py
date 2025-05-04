# knowledge_distiller_kd/core/utils.py
"""
通用工具函数模块。
"""

import logging
import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Generator, Set

# 导入常量模块 (使用相对导入)
from . import constants
from .error_handler import handle_error, FileOperationError

# --- 日志设置 ---
# 使用常量中定义的 Logger 名称
logger = logging.getLogger(constants.LOGGER_NAME)

def setup_logger(
    log_level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_dir: str = constants.DEFAULT_LOG_DIR
) -> None:
    """
    配置日志记录器。

    Args:
        log_level: 日志级别 (例如 logging.INFO)。
        log_file: 可选的日志文件名。如果未提供，则使用默认文件名。
        log_dir: 日志文件存放目录。
    """
    # 移除现有的 handlers，防止重复添加
    # 获取 logger 实例进行操作
    current_logger = logging.getLogger(constants.LOGGER_NAME)
    for handler in current_logger.handlers[:]:
        current_logger.removeHandler(handler)
        handler.close()

    current_logger.setLevel(log_level)
    formatter = logging.Formatter(constants.LOG_FORMAT, datefmt=constants.LOG_DATE_FORMAT)

    # 控制台处理器
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    current_logger.addHandler(ch)

    # 文件处理器 (如果需要)
    log_file_path: Optional[Path] = None
    if log_dir:
        log_dir_path = Path(log_dir)
        try:
            log_dir_path.mkdir(parents=True, exist_ok=True)
            # 确定最终日志文件路径
            if log_file:
                 log_file_path = log_dir_path / Path(log_file).name
            else:
                 log_file_path = log_dir_path / constants.DEFAULT_LOG_FILE

            fh = logging.FileHandler(log_file_path, encoding=constants.DEFAULT_ENCODING)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            current_logger.addHandler(fh)
            current_logger.info(f"日志将记录到文件: {log_file_path}") # 使用 current_logger
        except OSError as e:
            current_logger.error(f"无法创建日志目录或文件 '{log_dir_path}' 或 '{log_file_path}': {e}", exc_info=True)
        except Exception as e:
             current_logger.error(f"设置文件日志处理器时发生意外错误: {e}", exc_info=True)

    current_logger.propagate = False
    current_logger.info(f"日志级别设置为: {logging.getLevelName(log_level)}")
    if not log_file_path:
         current_logger.info("未配置日志文件，日志仅输出到控制台。")


# --- 决策键处理 ---
def create_decision_key(file_path: str, block_id: str, block_type: str) -> str:
    """
    根据文件路径、块ID和块类型创建唯一的决策键。
    """
    file_part = str(file_path) if file_path else "UNKNOWN_FILE"
    id_part = str(block_id) if block_id else "UNKNOWN_ID"
    type_part = str(block_type) if block_type else "UNKNOWN_TYPE"
    return f"{file_part}{constants.DECISION_KEY_SEPARATOR}{id_part}{constants.DECISION_KEY_SEPARATOR}{type_part}"

# --- 修正后的 parse_decision_key ---
def parse_decision_key(key: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    解析决策键，提取文件路径、块ID和块类型。
    使用 rsplit 确保能处理路径或ID中包含分隔符的情况。
    """
    if not key:
        return None, None, None
    try:
        # 从右边分割，最多分割 2 次
        parts = key.rsplit(constants.DECISION_KEY_SEPARATOR, 2)
        if len(parts) == 3:
            # parts[0] = 文件路径 (可能包含分隔符)
            # parts[1] = 块 ID
            # parts[2] = 块类型
            return parts[0], parts[1], parts[2]
        else:
            # 如果不能正好分割成 3 部分，则认为格式无效
            logger.warning(f"无法解析决策键: '{key}'，格式不符合预期（需要两个 '{constants.DECISION_KEY_SEPARATOR}' 分隔符）。")
            return None, None, None
    except Exception as e:
        # 使用模块级的 logger 记录错误
        logger.error(f"解析决策键 '{key}' 时出错: {e}")
        return None, None, None
# --- 修正结束 ---


# --- 文件和目录操作 ---
def find_markdown_files(directory: Union[str, Path], recursive: bool = True) -> Generator[Path, None, None]:
    """
    查找指定目录下的 Markdown 文件。
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"指定的目录不存在: {directory}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"指定的路径不是一个目录: {directory}")

    # 使用常量中定义的扩展名集合
    patterns = [f"**/*{ext}" for ext in constants.SUPPORTED_EXTENSIONS] if recursive else [f"*{ext}" for ext in constants.SUPPORTED_EXTENSIONS]

    logger.info(f"开始在目录 '{dir_path}' 中搜索 Markdown 文件 (递归: {recursive})...")
    count = 0
    processed_files : Set[Path] = set() # 防止因模式重叠重复 yield
    for pattern in patterns:
        for file_path in dir_path.glob(pattern):
            # 确保是文件且未被处理过
            if file_path.is_file() and file_path not in processed_files:
                logger.debug(f"找到 Markdown 文件: {file_path}")
                yield file_path
                processed_files.add(file_path)
                count += 1
    logger.info(f"搜索完成，共找到 {count} 个 Markdown 文件。")


# --- 文本处理和显示 ---
def display_block_preview(text: Optional[str], max_len: int = constants.PREVIEW_LENGTH) -> str:
    """
    生成用于显示的文本块预览。
    """
    if text is None:
        return "[无内容]"
    preview = text.replace('\n', ' ').strip()
    if len(preview) > max_len:
        return preview[:max_len] + "..."
    return preview


# --- 其他通用工具 ---
def calculate_md5(text: str) -> str:
    """计算字符串的 MD5 哈希值。"""
    return hashlib.md5(text.encode(constants.DEFAULT_ENCODING)).hexdigest()

def sort_blocks_key(block: Any) -> Tuple[str, str]:
    """
    为 ContentBlock 提供排序键 (按文件路径和块 ID)。
    警告: 对于非数字或非特定格式的 block_id，排序效果可能不符合预期。
    """
    if hasattr(block, 'file_path') and hasattr(block, 'block_id'):
        try:
            # 尝试提取 block_id 中的数字部分进行排序
            block_id_str = str(block.block_id)
            numeric_part = "".join(filter(str.isdigit, block_id_str))
            if numeric_part: # 如果提取到了数字
                sort_id = int(numeric_part)
                # 使用固定长度的零填充字符串保证数字排序正确
                return (str(block.file_path), f"{sort_id:010d}")
            else: # 没有数字部分，按字符串排序
                 return (str(block.file_path), block_id_str)
        except (ValueError, TypeError):
             # 发生异常，按字符串排序
             return (str(block.file_path), str(block.block_id))
    # 如果对象没有所需属性，返回空元组，排在最前面
    return ("", "")

# --- 导入依赖检查 ---
def check_optional_dependency(dependency_name: str) -> bool:
    """检查可选依赖项是否已安装。"""
    try:
        __import__(dependency_name)
        return True
    except ImportError:
        return False

# --- get_markdown_parser 函数已被移除 ---