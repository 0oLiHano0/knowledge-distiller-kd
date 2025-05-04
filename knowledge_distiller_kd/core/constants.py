# knowledge_distiller_kd/core/constants.py
"""
定义项目中使用的常量。
"""

import logging
from pathlib import Path
import os # 添加 os 模块导入以使用 os.path.join

# --- 日志相关 ---
LOGGER_NAME = "kd_tool" # 定义日志记录器的名称
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = f"{LOGGER_NAME}.log" # 使用定义的名称
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
DEFAULT_LOG_LEVEL = "INFO"

# --- 文件处理相关 ---
DEFAULT_ENCODING = "utf-8"
SUPPORTED_EXTENSIONS = {".md", ".markdown"} # 明确支持的扩展名
DEFAULT_OUTPUT_DIR = "output" # 添加默认输出目录常量
DEFAULT_OUTPUT_SUFFIX = "_deduped"

# --- 决策相关 ---
DECISION_KEEP = "keep"
DECISION_DELETE = "delete"
DECISION_UNDECIDED = "undecided"
DEFAULT_DECISION_DIR = "decisions"
DEFAULT_DECISION_FILE = os.path.join(DEFAULT_DECISION_DIR, "decisions.json") # 使用 os.path.join
DECISION_KEY_SEPARATOR = "::" # 定义决策键分隔符

# --- 分析相关 ---
# MD5
MD5_HASH_ALGORITHM = "md5"

# Semantic
DEFAULT_SEMANTIC_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.80 # 默认相似度阈值
DEFAULT_BATCH_SIZE = 32 # 语义模型编码批处理大小
VECTOR_CACHE_FILE = "cache/vector_cache.pkl" # 向量缓存文件路径 (如果使用持久化缓存)
CACHE_DIR = "cache" # 缓存目录

# --- 错误代码 ---
ERROR_CODES = {
    "FILE_NOT_FOUND": 1001,
    "FILE_PROCESSING_ERROR": 1002,
    "INVALID_DIRECTORY": 1003,
    "DIRECTORY_PROCESSING_ERROR": 1004,
    "CONFIGURATION_ERROR": 2001,
    "MODEL_LOADING_ERROR": 3001,
    "VECTOR_COMPUTATION_ERROR": 3002,
    "SIMILARITY_CALCULATION_ERROR": 3003,
    "ANALYSIS_ERROR": 4001, # 通用分析错误
    "USER_INPUT_ERROR": 5001,
    "UNEXPECTED_ERROR": 9999,
}


# --- 块类型 ---
BLOCK_TYPE_TITLE = "Title"
BLOCK_TYPE_TEXT = "NarrativeText" # 或 Text
BLOCK_TYPE_LIST_ITEM = "ListItem"
BLOCK_TYPE_CODE = "CodeSnippet"
BLOCK_TYPE_TABLE = "Table"
# ... 其他可能的类型

# --- 版本信息 ---
VERSION = "1.0.0" # 假设的版本号

# --- UI Constants ---
PREVIEW_LENGTH = 80 # Max length for block preview in UI

# --- 缓存相关 ---
# (如果需要更复杂的缓存策略)
# CACHE_MAX_SIZE = 10000 # 示例：缓存最大条目数
# CACHE_TTL = 3600 # 示例：缓存过期时间（秒）
DEFAULT_CACHE_BASE_DIR = ".kd_cache"
