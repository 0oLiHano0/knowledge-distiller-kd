# knowledge_distiller_kd/core/utils.py
"""
通用工具函数模块。
"""

import logging
import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Generator

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
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(log_level)
    formatter = logging.Formatter(constants.LOG_FORMAT, datefmt=constants.LOG_DATE_FORMAT)

    # 控制台处理器
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件处理器 (如果需要)
    log_file_path: Optional[Path] = None
    if log_dir:
        log_dir_path = Path(log_dir)
        try:
            log_dir_path.mkdir(parents=True, exist_ok=True)
            # 确定最终日志文件路径
            if log_file:
                 # 如果提供了文件名，直接使用（确保在日志目录下）
                 log_file_path = log_dir_path / Path(log_file).name
            else:
                 # 使用默认文件名
                 log_file_path = log_dir_path / constants.DEFAULT_LOG_FILE

            fh = logging.FileHandler(log_file_path, encoding=constants.DEFAULT_ENCODING)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.info(f"日志将记录到文件: {log_file_path}")
        except OSError as e:
            logger.error(f"无法创建日志目录或文件 '{log_dir_path}' 或 '{log_file_path}': {e}", exc_info=True)
        except Exception as e:
             logger.error(f"设置文件日志处理器时发生意外错误: {e}", exc_info=True)

    # 防止日志传播到根 logger (如果根 logger 有自己的 handlers)
    logger.propagate = False

    logger.info(f"日志级别设置为: {logging.getLevelName(log_level)}")
    if not log_file_path:
         logger.info("未配置日志文件，日志仅输出到控制台。")


# --- 决策键处理 ---
def create_decision_key(file_path: str, block_id: str, block_type: str) -> str:
    """
    根据文件路径、块ID和块类型创建唯一的决策键。

    Args:
        file_path: 文件路径字符串。
        block_id: 内容块的唯一标识符。
        block_type: 内容块的类型。

    Returns:
        格式化的决策键字符串。
    """
    # 确保所有部分都是字符串，并处理 None 或空字符串的情况
    file_part = str(file_path) if file_path else "UNKNOWN_FILE"
    id_part = str(block_id) if block_id else "UNKNOWN_ID"
    type_part = str(block_type) if block_type else "UNKNOWN_TYPE"
    return f"{file_part}{constants.DECISION_KEY_SEPARATOR}{id_part}{constants.DECISION_KEY_SEPARATOR}{type_part}"

def parse_decision_key(key: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    解析决策键，提取文件路径、块ID和块类型。

    Args:
        key: 决策键字符串。

    Returns:
        一个包含文件路径、块ID和块类型的元组。如果解析失败，则返回 (None, None, None)。
    """
    try:
        parts = key.split(constants.DECISION_KEY_SEPARATOR)
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        else:
            # 处理可能的文件路径中包含分隔符的情况（虽然不推荐）
            # 假设块ID和类型不包含分隔符
            if len(parts) > 3:
                 block_type = parts[-1]
                 block_id = parts[-2]
                 file_path = constants.DECISION_KEY_SEPARATOR.join(parts[:-2])
                 return file_path, block_id, block_type
            logger.warning(f"无法解析决策键: '{key}'，部分数量不足或过多。")
            return None, None, None
    except Exception as e:
        logger.error(f"解析决策键 '{key}' 时出错: {e}")
        return None, None, None


# --- 文件和目录操作 ---
def find_markdown_files(directory: Union[str, Path], recursive: bool = True) -> Generator[Path, None, None]:
    """
    查找指定目录下的 Markdown 文件。

    Args:
        directory: 要搜索的目录路径。
        recursive: 是否递归搜索子目录。

    Yields:
        Path: 找到的 Markdown 文件的路径对象。

    Raises:
        FileNotFoundError: 如果指定的目录不存在。
        NotADirectoryError: 如果指定的路径不是一个目录。
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"指定的目录不存在: {directory}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"指定的路径不是一个目录: {directory}")

    pattern = "**/*.md" if recursive else "*.md"
    # 包含 .markdown 扩展名
    patterns = ["**/*.md", "**/*.markdown"] if recursive else ["*.md", "*.markdown"]

    logger.info(f"开始在目录 '{dir_path}' 中搜索 Markdown 文件 (递归: {recursive})...")
    count = 0
    for ext_pattern in patterns:
        for file_path in dir_path.glob(ext_pattern):
            if file_path.is_file():
                logger.debug(f"找到 Markdown 文件: {file_path}")
                yield file_path
                count += 1
    logger.info(f"搜索完成，共找到 {count} 个 Markdown 文件。")


# --- 文本处理和显示 ---
# ==================== Correction: Use PREVIEW_LENGTH ====================
def display_block_preview(text: Optional[str], max_len: int = constants.PREVIEW_LENGTH) -> str:
# =======================================================================
    """
    生成用于显示的文本块预览。

    Args:
        text: 要生成预览的文本。
        max_len: 预览的最大长度。

    Returns:
        处理过的预览字符串。
    """
    if text is None:
        return "[无内容]"
    # 移除换行符以便单行显示，并截断
    preview = text.replace('\n', ' ').strip()
    if len(preview) > max_len:
        return preview[:max_len] + "..."
    return preview


# --- 其他通用工具 ---
def calculate_md5(text: str) -> str:
    """计算字符串的 MD5 哈希值。"""
    return hashlib.md5(text.encode(constants.DEFAULT_ENCODING)).hexdigest()

# 用于排序 ContentBlock 的函数 (示例)
def sort_blocks_key(block: Any) -> Tuple[str, str]:
    """
    为 ContentBlock 提供排序键 (按文件路径和块 ID)。
    注意: block_id 当前是哈希值，直接排序意义不大，除非改成数字 ID。
    """
    if hasattr(block, 'file_path') and hasattr(block, 'block_id'):
        # 尝试将 block_id 转为整数排序，如果失败则按字符串排序
        try:
            # 假设 block_id 格式为 "block-数字" 或纯数字
            sort_id = int(str(block.block_id).split('-')[-1]) if isinstance(block.block_id, str) else int(block.block_id)
            return (str(block.file_path), f"{sort_id:010d}") # 补零保证排序正确性
        except (ValueError, TypeError):
             return (str(block.file_path), str(block.block_id)) # 按字符串排序
    return ("", "") # 默认值

# --- 导入依赖检查 ---
# (可以在这里添加检查可选依赖项的功能)
def check_optional_dependency(dependency_name: str) -> bool:
    """检查可选依赖项是否已安装。"""
    try:
        __import__(dependency_name)
        return True
    except ImportError:
        return False

# --- 弃用/旧代码区域 (如果需要) ---
# def old_function(): ...

# --- 获取 Markdown 解析器 ---
# (这个函数可能更适合放在 document_processor.py 中，但暂时放在这里)
def get_markdown_parser():
    """获取 Unstructured 的 Markdown 解析器实例。"""
    try:
        # Unstructured 的导入可能比较深层，根据实际情况调整
        from unstructured.partition.md import partition_md
        # 或者根据需要返回特定的解析器类或函数
        return partition_md
    except ImportError:
        logger.error("无法导入 Unstructured Markdown 解析器。请确保 `unstructured` 已正确安装。")
        return None
    except Exception as e:
         logger.error(f"导入 Markdown 解析器时发生错误: {e}")
         return None
