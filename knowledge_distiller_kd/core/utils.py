# knowledge_distiller_kd/core/utils.py
"""
通用工具函数模块。
"""

import logging
import json
import hashlib
import platform
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

# -*- coding: utf-8 -*-
# knowledge_distiller_kd/core/models.py
"""
Defines the core Data Transfer Objects (DTOs) and Enumerations used throughout the application,
particularly for data exchange between layers (e.g., storage, analysis, UI).
(Version confirmed to pass FileStorage tests - incorporates AI feedback)
"""

import datetime
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from typing     import Optional, Dict, Any

logger = logging.getLogger(__name__)

# --- Enumerations (Reflecting version that passed tests) ---

class BlockType(Enum):
    """Enumeration for different types of content blocks."""
    UNKNOWN = "unknown"
    TEXT = "text"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    CODE = "code" # Note: Consider if CODE_SNIPPET was more accurate previously
    TABLE = "table"
    # Add more types as needed based on 'unstructured' or specific needs

class AnalysisType(Enum):
    """Enumeration for different types of analysis performed."""
    UNKNOWN = "unknown"
    MD5_DUPLICATE = "md5_duplicate"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    # Add more types as needed

class DecisionType(Enum):
    """Enumeration for user decisions on analysis results."""
    UNDECIDED = "undecided"
    MERGE = "merge" # Note: Revisit if this accurately reflects needed actions vs. KEEP_*
    IGNORE = "ignore" # Note: Revisit if this accurately reflects needed actions vs. SKIP
    MARK_DUPLICATE = "mark_duplicate"
    MARK_SIMILAR = "mark_similar" # Note: Revisit if this accurately reflects needed actions
    DELETE          = "delete"
    # Add more types as needed

# --- Data Transfer Objects (DTOs) ---

@dataclass
class ContentBlock:
    """Represents a distinct block of content extracted from a file."""
    # Non-Default Fields first
    file_id: str
    text: str
    block_type: BlockType # Uses the Enum defined above

    # Default Fields after
    block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)


    def to_dict(self) -> Dict[str, Any]:
        """Serializes the ContentBlock to a dictionary."""
        return {
            "block_id": self.block_id,
            "file_id": self.file_id,
            "text": self.text,
            "block_type": self.block_type.value, # Store enum value
            "metadata": self.metadata,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentBlock':
        """Deserializes a dictionary into a ContentBlock object."""
        if "file_id" not in data or "text" not in data:
            raise ValueError("Missing required fields 'file_id' or 'text' in ContentBlock data")
        try:
            # Use UNKNOWN as the default if block_type is missing or invalid
            block_type_value = data.get("block_type", BlockType.UNKNOWN.value)
            block_type = BlockType(block_type_value)
        except ValueError:
            logger.warning(f"Invalid BlockType value '{block_type_value}' found for block {data.get('block_id')}. Defaulting to UNKNOWN.")
            block_type = BlockType.UNKNOWN

        return cls(
            block_id=data.get("block_id", str(uuid.uuid4())),
            file_id=data["file_id"],
            text=data["text"],
            block_type=block_type,
            metadata=data.get("metadata", {}),
            )

@dataclass
class AnalysisResult:
    """
    Represents the result of an analysis comparing two content blocks.
    Includes a deterministically generated result_id based on block IDs and type.
    """
    # Non-Default Fields first
    block_id_1: str
    block_id_2: str
    analysis_type: AnalysisType

    # Default Fields after
    score: Optional[float] = None

    # Generated Fields (using UUIDv5 for deterministic ID)
    result_id: str = field(init=False)

    def __post_init__(self):
        """Generate a deterministic result_id after initialization."""
        sorted_ids = sorted([self.block_id_1, self.block_id_2])
        id_string = f"{sorted_ids[0]}_{sorted_ids[1]}_{self.analysis_type.value}"
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8') # Example DNS namespace
        self.result_id = str(uuid.uuid5(namespace, id_string))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the AnalysisResult to a dictionary."""
        return {
            "result_id": self.result_id,
            "block_id_1": self.block_id_1,
            "block_id_2": self.block_id_2,
            "analysis_type": self.analysis_type.value, # Store enum value
            "score": self.score,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Deserializes a dictionary into an AnalysisResult object."""
        if "block_id_1" not in data or "block_id_2" not in data or "analysis_type" not in data:
            raise ValueError("Missing required fields 'block_id_1', 'block_id_2', or 'analysis_type' in AnalysisResult data")
        try:
            # Use UNKNOWN as default if analysis_type is missing or invalid
            analysis_type_value = data.get("analysis_type", AnalysisType.UNKNOWN.value)
            analysis_type = AnalysisType(analysis_type_value)
        except ValueError:
             logger.warning(f"Invalid AnalysisType value '{analysis_type_value}' found for result. Defaulting to UNKNOWN.")
             analysis_type = AnalysisType.UNKNOWN

        return cls(
            block_id_1=data["block_id_1"],
            block_id_2=data["block_id_2"],
            analysis_type=analysis_type,
            score=data.get("score"),
            )

@dataclass
class UserDecision:
    """
    Represents a user's decision regarding a pair of content blocks (AnalysisResult).
    Includes a deterministically generated decision_id based on block IDs and type.
    """
    # Non-Default Fields first
    block_id_1: str
    block_id_2: str
    analysis_type: AnalysisType

    # Default Fields after
    decision: DecisionType = DecisionType.UNDECIDED
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    notes: Optional[str] = None

    # Generated Fields (using UUIDv5 for deterministic ID, same as AnalysisResult's result_id)
    decision_id: str = field(init=False)

    def __post_init__(self):
        """Generate a deterministic decision_id after initialization."""
        sorted_ids = sorted([self.block_id_1, self.block_id_2])
        id_string = f"{sorted_ids[0]}_{sorted_ids[1]}_{self.analysis_type.value}"
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8') # Example DNS namespace
        self.decision_id = str(uuid.uuid5(namespace, id_string))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the UserDecision to a dictionary."""
        return {
            "decision_id": self.decision_id,
            "block_id_1": self.block_id_1,
            "block_id_2": self.block_id_2,
            "analysis_type": self.analysis_type.value, # Store enum value
            "decision": self.decision.value, # Store enum value
            "timestamp": self.timestamp.isoformat(), # Use ISO format for timestamps
            "notes": self.notes,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserDecision':
        """Deserializes a dictionary into a UserDecision object."""
        if "block_id_1" not in data or "block_id_2" not in data or "analysis_type" not in data:
            raise ValueError("Missing required fields 'block_id_1', 'block_id_2', or 'analysis_type' in UserDecision data")

        try:
            # Use UNKNOWN as default if analysis_type is missing or invalid
            analysis_type_value = data.get("analysis_type", AnalysisType.UNKNOWN.value)
            analysis_type = AnalysisType(analysis_type_value)
        except ValueError:
             logger.warning(f"Invalid AnalysisType value '{analysis_type_value}' found for decision. Defaulting to UNKNOWN.")
             analysis_type = AnalysisType.UNKNOWN

        try:
            # Use UNDECIDED as default if decision is missing or invalid
            decision_value = data.get("decision", DecisionType.UNDECIDED.value)
            decision = DecisionType(decision_value)
        except ValueError:
            logger.warning(f"Invalid DecisionType value '{decision_value}' found for decision {data.get('decision_id')}. Defaulting to UNDECIDED.")
            decision = DecisionType.UNDECIDED

        # Handle timestamp deserialization carefully
        timestamp_str = data.get("timestamp")
        timestamp = datetime.datetime.now(datetime.timezone.utc) # Default
        if timestamp_str:
            try:
                if timestamp_str.endswith('Z'):
                    timestamp_str = timestamp_str[:-1] + '+00:00'
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
                if timestamp.tzinfo is None:
                     timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)
            except ValueError:
                 logger.warning(f"Invalid timestamp format '{timestamp_str}'. Using current UTC time.")

        return cls(
            block_id_1=data["block_id_1"],
            block_id_2=data["block_id_2"],
            analysis_type=analysis_type,
            decision=decision,
            timestamp=timestamp,
            notes=data.get("notes"),
            )

@dataclass
class FileRecord:
    """
    Represents metadata associated with a registered file.
    Fields without defaults MUST come before fields with defaults.
    """
    # --- Non-Default Fields ---
    file_id: str          # Unique ID assigned by the storage system
    original_path: str    # The original path provided by the user (Corrected)

    # --- Default Fields ---
    registration_time: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict) # Keep metadata flexible


    def to_dict(self) -> Dict[str, Any]:
        """Serializes the FileRecord to a dictionary."""
        return {
            "file_id": self.file_id,
            "original_path": self.original_path,
            "registration_time": self.registration_time.isoformat(),
            "metadata": self.metadata,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileRecord':
        """Deserializes a dictionary into a FileRecord object."""
        if "file_id" not in data or "original_path" not in data:
            raise ValueError("Missing required fields 'file_id' or 'original_path' in FileRecord data")

        timestamp_str = data.get("registration_time")
        registration_time = datetime.datetime.now(datetime.timezone.utc) # Default
        if timestamp_str:
            try:
                if timestamp_str.endswith('Z'):
                    timestamp_str = timestamp_str[:-1] + '+00:00'
                registration_time = datetime.datetime.fromisoformat(timestamp_str)
                if registration_time.tzinfo is None:
                     registration_time = registration_time.replace(tzinfo=datetime.timezone.utc)
            except ValueError:
                 logger.warning(f"Invalid registration_time format '{timestamp_str}'. Using current UTC time.")

        return cls(
            file_id=data["file_id"],
            original_path=data["original_path"],
            registration_time=registration_time,
            metadata=data.get("metadata", {}),
            )


from pydantic import BaseModel, Field
from typing     import Optional, Dict, Any

class BlockDTO(BaseModel):
    block_id: str
    file_id: str
    block_type: BlockType
    text_content: str
    analysis_text: str
    char_count: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    kd_processing_status: DecisionType = DecisionType.UNDECIDED
    duplicate_of_block_id: Optional[str] = None

def normalize_text_for_analysis(text: str) -> str:
    """
    文本规范化占位函数，目前直接原样返回。
    如果需要更复杂的清洗，可以后续再改。
    """
    return text

# --- 20250509添加以下函数 ---
def get_bundled_czkawka_path() -> str:
    """
    返回捆绑在项目 vendor 目录下的 czkawka_cli 可执行文件路径。
    当前仅支持 macOS ARM64 架构。
    """
    system = platform.system().lower()
    arch = platform.machine().lower()

    # 项目根目录: knowledge_distiller_kd/
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    if system == "darwin" and arch in ("arm64", "aarch64"):
        bin_path = BASE_DIR / "vendor" / "czkawka" / "macos-arm64" / "czkawka_cli"
    else:
        raise RuntimeError(f"Unsupported platform/arch for bundled czkawka: {system}/{arch}")

    if not bin_path.exists():
        raise FileNotFoundError(f"Bundled czkawka_cli not found at {bin_path}")

    return str(bin_path)

# --- 20250509添加函数 End---