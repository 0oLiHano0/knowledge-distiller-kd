# knowledge_distiller_kd/core/utils.py
"""
通用工具函数模块。

提供日志设置、键管理、文本预览等辅助功能。
"""

import logging
import sys # 添加导入 sys 用于 StreamHandler
from pathlib import Path
# ==================== 修改：从 typing 导入 Union ====================
from typing import Optional, Tuple, TYPE_CHECKING, Union, Any, TypeAlias
# ============================================================
import re
from knowledge_distiller_kd.core import constants

# 导入 mistune 用于 get_markdown_parser
try:
    import mistune
    MISTUNE_AVAILABLE = True
except ImportError:
    mistune = None # 保持为 None
    MISTUNE_AVAILABLE = False

# 导入 ContentBlock 用于类型提示 (如果需要)
if TYPE_CHECKING:
    from knowledge_distiller_kd.core.document_processor import ContentBlock
    # 在 TYPE_CHECKING 中导入 mistune
    # 这样类型检查器知道 mistune.Markdown，但运行时不会出错
    if MISTUNE_AVAILABLE:
        import mistune


# 全局 logger 实例
logger = logging.getLogger(constants.LOGGER_NAME) # 使用常量定义的名字

def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """
    配置日志记录器。

    Args:
        level: 日志记录级别 (例如 logging.INFO, logging.DEBUG)

    Returns:
        logging.Logger: 配置好的日志记录器实例
    """
    # 检查是否已经有 handlers，避免重复添加
    # (如果 logger 是首次获取，hasHandlers() 可能为 False)
    # 更可靠的方式是移除所有现有的 handlers
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            try:
                # 尝试关闭 handler，忽略可能的错误
                handler.close()
            except Exception:
                pass # 忽略关闭错误
            logger.removeHandler(handler)

    logger.setLevel(level)
    log_format = logging.Formatter(constants.LOG_FORMAT, datefmt=constants.LOG_DATE_FORMAT)

    # --- 文件 Handler ---
    try:
        log_dir = Path(constants.DEFAULT_LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True) # 确保目录存在
        log_file = log_dir / constants.DEFAULT_LOG_FILE
        file_handler = logging.FileHandler(log_file, encoding=constants.DEFAULT_ENCODING)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    except Exception as e:
        # 在无法设置文件日志时，打印警告到 stderr
        print(f"警告：无法配置日志文件处理器 '{constants.DEFAULT_LOG_FILE}': {e}", file=sys.stderr)

    # --- Stream Handler (控制台输出) ---
    # 总是尝试添加 StreamHandler，这样 pytest caplog 才能捕获
    try:
        stream_handler = logging.StreamHandler(sys.stdout) # 输出到标准输出
        stream_handler.setFormatter(log_format)
        # 可以设置不同的级别，例如控制台只显示 INFO
        # stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)
    except Exception as e:
         # 在无法设置流日志时，打印警告到 stderr
         print(f"警告：无法配置日志流处理器: {e}", file=sys.stderr)

    logger.propagate = False # 防止日志向上传播到根 logger
    # 避免在测试中重复打印初始化信息
    # logger.debug(f"Logger '{constants.LOGGER_NAME}' setup complete with level {logging.getLevelName(level)}.")
    return logger

def create_decision_key(file_path: Union[str, Path], block_id: str, block_type: str) -> str:
    """
    根据文件路径、块ID和块类型创建唯一的决策键。

    Args:
        file_path: 文件路径 (字符串或Path对象)
        block_id: 内容块的唯一ID
        block_type: 内容块的类型

    Returns:
        str: 生成的决策键 (格式: "文件路径::块ID::块类型")
    """
    # 确保 file_path 是字符串
    file_path_str = str(file_path)
    # 确保 block_id 是字符串
    block_id_str = str(block_id)
    # 确保 block_type 是字符串
    block_type_str = str(block_type)

    # 使用常量中定义的分隔符
    separator = constants.DECISION_KEY_SEPARATOR
    return f"{file_path_str}{separator}{block_id_str}{separator}{block_type_str}"

def parse_decision_key(key: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    解析决策键，提取文件路径、块ID和块类型。

    Args:
        key: 要解析的决策键字符串

    Returns:
        Tuple[Optional[str], Optional[str], Optional[str]]: 包含文件路径、块ID和块类型的元组。
                                                        如果解析失败，返回 (None, None, None)。
    """
    if not key:
        return None, None, None

    separator = constants.DECISION_KEY_SEPARATOR
    # 使用 rsplit 从右侧分割两次，确保即使文件路径包含分隔符也能正确处理
    parts = key.rsplit(separator, 2)

    if len(parts) == 3:
        file_path, block_id, block_type = parts
        # 做一些基本的健全性检查，例如块类型不能为空
        if not file_path or not block_id or not block_type:
             logger.warning(f"解析决策键时遇到空部分: '{key}' -> {parts}")
             # 根据需要决定是否返回 None 或空字符串
             # return None, None, None
        return file_path, block_id, block_type
    else:
        # 如果分割结果不是3部分，说明格式无效
        logger.warning(f"无法解析无效的决策键格式: '{key}'")
        return None, None, None


def extract_text_from_children(element: Any) -> str:
    """
    递归地从元素的子元素中提取文本内容。
    (注意：此函数可能不再需要，因为 ContentBlock 直接使用 element.text)
    """
    text = ""
    # 检查 element 是否可迭代且非字符串 (避免迭代字符串)
    if hasattr(element, 'children') and isinstance(element.children, (list, tuple)):
        for child in element.children:
            if hasattr(child, 'text') and isinstance(child.text, str):
                text += child.text + " "
            else:
                # 递归处理更深层次的子元素 (如果结构允许)
                text += extract_text_from_children(child) + " "
    elif hasattr(element, 'text') and isinstance(element.text, str):
        text = element.text
    return text.strip()

def display_block_preview(text: Optional[str], max_len: int = constants.PREVIEW_MAX_LEN) -> str:
    """
    生成用于在命令行显示的块内容预览。

    Args:
        text: 要预览的文本内容
        max_len: 预览的最大长度 (指文本部分，不含后缀)

    Returns:
        str: 生成的预览字符串，包含长度信息
    """
    if text is None:
        text = "" # 处理 None 输入

    # 计算原始长度
    original_length = len(text)
    # 清理文本以便单行显示和长度计算（移除换行符等）
    cleaned_text = text.replace('\n', ' ').replace('\r', '').strip()

    preview_text = cleaned_text
    suffix = f" [长度: {original_length}字符]"

    # 只有当实际清理后的文本长度超过限制时才截断
    if len(cleaned_text) > max_len:
        # 截断文本，留出 "..." 的位置 (3个字符)
        trunc_len = max(0, max_len - 3) # 确保截断长度不为负
        preview_text = cleaned_text[:trunc_len] + "..."

    # 最终返回预览文本 + 长度后缀
    return preview_text + suffix


# 仅当 mistune 可用时定义实际函数体
if MISTUNE_AVAILABLE and mistune:
    def get_markdown_parser() -> 'mistune.Markdown': # 使用字符串避免运行时错误
        """
        获取配置好的Markdown解析器实例。
        (注意：随着使用 unstructured，此函数的重要性可能降低)
        """
        # 可以根据需要启用 mistune 的插件
        # 例如：启用表格、脚注、删除线等
        # plugins = ['table', 'footnotes', 'strikethrough', 'url']
        # return mistune.create_markdown(plugins=plugins)
        return mistune.create_markdown() # 默认配置
else:
    def get_markdown_parser() -> None:
        """
        如果 mistune 不可用，返回 None。
        """
        # 不再打印警告，避免重复输出
        # logger.warning("Mistune library not found, cannot provide Markdown parser.")
        return None

# ==================== 修改：使用 Union[int, str] 代替 Any ====================
SortKey: TypeAlias = Tuple[str, Union[int, str]]
# =======================================================================

# 这个函数可能需要更新以处理 ContentBlock 对象
def sort_blocks_key(block: Union[Tuple[str, int, str, str], 'ContentBlock']) -> SortKey: # 使用 TypeAlias
    """
    为内容块列表提供排序键（按文件名和块索引/ID）。

    Args:
        block: 内容块对象或元组

    Returns:
        SortKey: 用于排序的元组 (文件名, 块ID或索引)
    """
    if isinstance(block, tuple) and len(block) >= 2:
        # 处理旧的元组格式 (file_path, block_index, ...)
        file_path = str(block[0])
        try:
            # 尝试将第二个元素转为整数作为索引
            block_index: Union[int, str] = int(block[1]) # 类型明确为 int
        except (ValueError, TypeError):
            # 如果转换失败，使用默认值或记录警告
            logger.debug(f"无法将块元组的第二个元素 '{block[1]}' 转换为整数索引，使用 0 代替。")
            block_index = 0 # 类型为 int
        return (file_path, block_index)
    elif hasattr(block, 'file_path') and hasattr(block, 'block_id'):
        # 处理 ContentBlock 对象
        file_path = str(getattr(block, 'file_path', ''))
        # 直接使用字符串 block_id 进行排序，因为 block_id 通常是哈希值
        block_sort_key: str = str(getattr(block, 'block_id', '')) # 类型明确为 str
        return (file_path, block_sort_key)
    else:
        # 对于未知类型，返回默认值以避免排序错误
        logger.warning(f"无法识别的块类型用于排序: {type(block)}")
        return ("", "") # 类型为 (str, str)

# 注意：不在 utils.py 中直接调用 setup_logger()
# 应由主程序入口 (如 kd_tool_CLI.py 的 main 函数) 调用一次。
