"""
常量模块，定义了项目中使用的所有常量。

此模块包含：
1. 文件路径常量
2. 配置常量
3. 错误代码常量
4. 其他常量
"""

from typing import Final

# 文件路径常量
DEFAULT_INPUT_DIR: Final[str] = "input"
DEFAULT_OUTPUT_DIR: Final[str] = "output"
DEFAULT_DECISION_FILE: Final[str] = "decisions/decisions.json"
DEFAULT_LOG_FILE: Final[str] = "logs/kd_tool.log"

# 配置常量
DEFAULT_SEMANTIC_MODEL: Final[str] = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SIMILARITY_THRESHOLD: Final[float] = 0.8
DEFAULT_BATCH_SIZE: Final[int] = 32
DEFAULT_ENCODING: Final[str] = "utf-8"

# 错误代码常量
ERROR_CODE_INIT_FAILED: Final[str] = "INIT_FAILED"
ERROR_CODE_FILE_NOT_FOUND: Final[str] = "FILE_NOT_FOUND"
ERROR_CODE_PERMISSION_DENIED: Final[str] = "PERMISSION_DENIED"
ERROR_CODE_INVALID_PATH: Final[str] = "INVALID_PATH"
ERROR_CODE_READ_FILE_FAILED: Final[str] = "READ_FILE_FAILED"
ERROR_CODE_WRITE_FILE_FAILED: Final[str] = "WRITE_FILE_FAILED"
ERROR_CODE_PARSE_FILE_FAILED: Final[str] = "PARSE_FILE_FAILED"
ERROR_CODE_LOAD_MODEL_FAILED: Final[str] = "LOAD_MODEL_FAILED"
ERROR_CODE_ANALYZE_FAILED: Final[str] = "ANALYZE_FAILED"
ERROR_CODE_SAVE_DECISIONS_FAILED: Final[str] = "SAVE_DECISIONS_FAILED"
ERROR_CODE_LOAD_DECISIONS_FAILED: Final[str] = "LOAD_DECISIONS_FAILED"
ERROR_CODE_APPLY_DECISIONS_FAILED: Final[str] = "APPLY_DECISIONS_FAILED"

# 其他常量
DECISION_KEEP: Final[str] = "keep"
DECISION_DELETE: Final[str] = "delete"
DECISION_UNDECIDED: Final[str] = "undecided"
DECISION_KEY_SEPARATOR: Final[str] = "::"
DEFAULT_OUTPUT_SUFFIX: Final[str] = "_deduped"

BLOCK_TYPE_PARAGRAPH: Final[str] = "paragraph"
BLOCK_TYPE_HEADING: Final[str] = "heading"
BLOCK_TYPE_LIST: Final[str] = "list"
BLOCK_TYPE_CODE: Final[str] = "code"
BLOCK_TYPE_QUOTE: Final[str] = "quote"
BLOCK_TYPE_TABLE: Final[str] = "table"
BLOCK_TYPE_HTML: Final[str] = "html"
BLOCK_TYPE_LINK: Final[str] = "link"
BLOCK_TYPE_IMAGE: Final[str] = "image"
BLOCK_TYPE_THEMATIC_BREAK: Final[str] = "thematic_break"
BLOCK_TYPE_UNKNOWN: Final[str] = "unknown"

# 日志级别常量
LOG_LEVEL_DEBUG: Final[str] = "DEBUG"
LOG_LEVEL_INFO: Final[str] = "INFO"
LOG_LEVEL_WARNING: Final[str] = "WARNING"
LOG_LEVEL_ERROR: Final[str] = "ERROR"
LOG_LEVEL_CRITICAL: Final[str] = "CRITICAL"

# 文件扩展名常量
MARKDOWN_EXTENSIONS: Final[tuple] = (".md", ".markdown", ".mdown", ".mkdn", ".mkd")

# 进度显示常量
PROGRESS_BAR_WIDTH: Final[int] = 50
PROGRESS_BAR_CHAR: Final[str] = "="
PROGRESS_BAR_HEAD: Final[str] = ">"
PROGRESS_BAR_EMPTY: Final[str] = "-"

# 缓存常量
CACHE_DIR: Final[str] = ".cache"
VECTOR_CACHE_FILE: Final[str] = "vector_cache.pkl"
MODEL_CACHE_DIR: Final[str] = "models"

# 时间格式常量
DATETIME_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT: Final[str] = "%Y-%m-%d"
TIME_FORMAT: Final[str] = "%H:%M:%S"

# 语言常量
DEFAULT_LANGUAGE: Final[str] = "zh"
SUPPORTED_LANGUAGES: Final[tuple] = ("zh", "en")

# 版本常量
VERSION: Final[str] = "1.0.0"

