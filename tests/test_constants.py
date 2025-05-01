"""
测试 constants.py 中的常量定义。

这个模块包含对 constants.py 中所有常量的测试。
"""

import pytest
from knowledge_distiller_kd.core import constants
from typing import Final

def test_semantic_constants() -> None:
    """
    测试语义分析相关常量。
    """
    # 测试模型名称
    assert isinstance(constants.DEFAULT_SEMANTIC_MODEL, str)
    assert constants.DEFAULT_SEMANTIC_MODEL == "paraphrase-multilingual-MiniLM-L12-v2"
    
    # 测试相似度阈值
    assert isinstance(constants.DEFAULT_SIMILARITY_THRESHOLD, float)
    assert 0.0 <= constants.DEFAULT_SIMILARITY_THRESHOLD <= 1.0

def test_file_processing_constants() -> None:
    """
    测试文件处理相关常量。
    """
    # 测试决策键分隔符
    assert isinstance(constants.DECISION_KEY_SEPARATOR, str)
    assert constants.DECISION_KEY_SEPARATOR == "::"
    
    # 测试输出文件后缀
    assert isinstance(constants.DEFAULT_OUTPUT_SUFFIX, str)
    assert constants.DEFAULT_OUTPUT_SUFFIX == "_deduped"
    
    # 测试决策文件名
    assert isinstance(constants.DEFAULT_DECISION_FILE, str)
    assert constants.DEFAULT_DECISION_FILE == "decisions/decisions.json"

def test_error_code_constants() -> None:
    """
    测试错误代码常量。
    """
    # 测试所有错误代码常量
    error_codes = [
        'ERROR_CODE_INIT_FAILED',
        'ERROR_CODE_FILE_NOT_FOUND',
        'ERROR_CODE_PERMISSION_DENIED',
        'ERROR_CODE_INVALID_PATH',
        'ERROR_CODE_READ_FILE_FAILED',
        'ERROR_CODE_WRITE_FILE_FAILED',
        'ERROR_CODE_PARSE_FILE_FAILED',
        'ERROR_CODE_LOAD_MODEL_FAILED',
        'ERROR_CODE_ANALYZE_FAILED',
        'ERROR_CODE_SAVE_DECISIONS_FAILED',
        'ERROR_CODE_LOAD_DECISIONS_FAILED',
        'ERROR_CODE_APPLY_DECISIONS_FAILED'
    ]
    
    for code in error_codes:
        assert hasattr(constants, code)
        assert isinstance(getattr(constants, code), str)
        assert getattr(constants, code).isupper()

def test_block_type_constants() -> None:
    """
    测试块类型常量。
    """
    # 测试所有块类型常量
    block_types = [
        'BLOCK_TYPE_PARAGRAPH',
        'BLOCK_TYPE_HEADING',
        'BLOCK_TYPE_LIST',
        'BLOCK_TYPE_CODE',
        'BLOCK_TYPE_QUOTE',
        'BLOCK_TYPE_TABLE',
        'BLOCK_TYPE_HTML',
        'BLOCK_TYPE_LINK',
        'BLOCK_TYPE_IMAGE',
        'BLOCK_TYPE_THEMATIC_BREAK',
        'BLOCK_TYPE_UNKNOWN'
    ]
    
    for block_type in block_types:
        assert hasattr(constants, block_type)
        assert isinstance(getattr(constants, block_type), str)
        assert getattr(constants, block_type).islower()

def test_log_level_constants() -> None:
    """
    测试日志级别常量。
    """
    # 测试所有日志级别常量
    log_levels = [
        'LOG_LEVEL_DEBUG',
        'LOG_LEVEL_INFO',
        'LOG_LEVEL_WARNING',
        'LOG_LEVEL_ERROR',
        'LOG_LEVEL_CRITICAL'
    ]
    
    for level in log_levels:
        assert hasattr(constants, level)
        assert isinstance(getattr(constants, level), str)
        assert getattr(constants, level).isupper()

def test_file_extension_constants() -> None:
    """
    测试文件扩展名常量。
    """
    # 测试 Markdown 扩展名元组
    assert isinstance(constants.MARKDOWN_EXTENSIONS, tuple)
    assert all(isinstance(ext, str) for ext in constants.MARKDOWN_EXTENSIONS)
    assert all(ext.startswith('.') for ext in constants.MARKDOWN_EXTENSIONS)

def test_progress_bar_constants() -> None:
    """
    测试进度条相关常量。
    """
    # 测试进度条宽度
    assert isinstance(constants.PROGRESS_BAR_WIDTH, int)
    assert constants.PROGRESS_BAR_WIDTH > 0
    
    # 测试进度条字符
    assert isinstance(constants.PROGRESS_BAR_CHAR, str)
    assert len(constants.PROGRESS_BAR_CHAR) == 1
    
    # 测试进度条头部字符
    assert isinstance(constants.PROGRESS_BAR_HEAD, str)
    assert len(constants.PROGRESS_BAR_HEAD) == 1
    
    # 测试进度条空白字符
    assert isinstance(constants.PROGRESS_BAR_EMPTY, str)
    assert len(constants.PROGRESS_BAR_EMPTY) == 1

def test_cache_constants() -> None:
    """
    测试缓存相关常量。
    """
    # 测试缓存目录
    assert isinstance(constants.CACHE_DIR, str)
    assert constants.CACHE_DIR.startswith('.')
    
    # 测试向量缓存文件
    assert isinstance(constants.VECTOR_CACHE_FILE, str)
    assert constants.VECTOR_CACHE_FILE.endswith('.pkl')
    
    # 测试模型缓存目录
    assert isinstance(constants.MODEL_CACHE_DIR, str)
    assert constants.MODEL_CACHE_DIR == "models"

def test_datetime_format_constants() -> None:
    """
    测试日期时间格式常量。
    """
    # 测试日期时间格式
    assert isinstance(constants.DATETIME_FORMAT, str)
    assert '%Y' in constants.DATETIME_FORMAT
    assert '%m' in constants.DATETIME_FORMAT
    assert '%d' in constants.DATETIME_FORMAT
    assert '%H' in constants.DATETIME_FORMAT
    assert '%M' in constants.DATETIME_FORMAT
    assert '%S' in constants.DATETIME_FORMAT
    
    # 测试日期格式
    assert isinstance(constants.DATE_FORMAT, str)
    assert '%Y' in constants.DATE_FORMAT
    assert '%m' in constants.DATE_FORMAT
    assert '%d' in constants.DATE_FORMAT
    
    # 测试时间格式
    assert isinstance(constants.TIME_FORMAT, str)
    assert '%H' in constants.TIME_FORMAT
    assert '%M' in constants.TIME_FORMAT
    assert '%S' in constants.TIME_FORMAT

def test_language_constants() -> None:
    """
    测试语言相关常量。
    """
    # 测试默认语言
    assert isinstance(constants.DEFAULT_LANGUAGE, str)
    assert constants.DEFAULT_LANGUAGE in ('zh', 'en')
    
    # 测试支持的语言列表
    assert isinstance(constants.SUPPORTED_LANGUAGES, tuple)
    assert all(isinstance(lang, str) for lang in constants.SUPPORTED_LANGUAGES)
    assert constants.DEFAULT_LANGUAGE in constants.SUPPORTED_LANGUAGES 