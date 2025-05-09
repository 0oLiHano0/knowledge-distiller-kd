# tests/test_utils.py
"""
测试工具模块 (utils.py) 中的函数。
"""

import logging
from pathlib import Path
import pytest
import os
import tempfile
from typing import Union, Tuple, Any, List # 确保导入所需类型
from unittest.mock import MagicMock # 导入 MagicMock

# mistune 不再需要导入，因为相关函数和测试已移除

# 导入需要测试的函数和常量
from knowledge_distiller_kd.core import constants
from knowledge_distiller_kd.core.utils import (
    setup_logger,
    create_decision_key,
    parse_decision_key,
    # extract_text_from_children, # 已移除
    display_block_preview,
    # get_markdown_parser, # 已移除
    sort_blocks_key,
    find_markdown_files, # 如果需要测试这个函数
    calculate_md5,     # 如果需要测试这个函数
    # check_optional_dependency # 如果需要测试这个函数
)
from knowledge_distiller_kd.core.error_handler import KDError


# --- 测试 setup_logger ---
def test_setup_logger(caplog) -> None:
    """
    测试日志记录器的设置。
    """
    # 清理可能存在的旧 handlers，避免干扰
    logger_instance_before = logging.getLogger(constants.LOGGER_NAME)
    for handler in logger_instance_before.handlers[:]:
        logger_instance_before.removeHandler(handler)
        handler.close()

    # 设置为 DEBUG 级别进行测试
    setup_logger(logging.DEBUG) # 调用函数进行配置，不接收返回值

    # 获取配置好的 logger 实例
    logger_instance_after = logging.getLogger(constants.LOGGER_NAME)

    # 检查 logger 级别是否正确设置
    assert logger_instance_after.level == logging.DEBUG

    # 检查是否至少有一个 handler
    assert len(logger_instance_after.handlers) > 0
    has_stream = any(isinstance(h, logging.StreamHandler) for h in logger_instance_after.handlers)
    has_file = any(isinstance(h, logging.FileHandler) for h in logger_instance_after.handlers)
    assert has_stream or has_file # 至少配置了一种

    # 记录一条消息以验证 (可选)
    test_message = "这是一个调试信息"
    logger_instance_after.debug(test_message)
    # 如果需要，可以使用 caplog 检查消息是否被捕获
    # assert test_message in caplog.text


# --- 测试 create_decision_key 和 parse_decision_key ---
def test_create_decision_key() -> None:
    """测试创建决策键的功能。"""
    file_path = "/path/to/document.md"
    block_id = "block123"
    block_type = "NarrativeText"
    separator = constants.DECISION_KEY_SEPARATOR
    expected_key = f"{file_path}{separator}{block_id}{separator}{block_type}"
    assert create_decision_key(file_path, block_id, block_type) == expected_key

    file_path_obj = Path("/path/to/document.md")
    assert create_decision_key(str(file_path_obj), block_id, block_type) == expected_key

    file_path_sep = f"/path/{separator}/doc.md"
    expected_key_sep = f"{file_path_sep}{separator}{block_id}{separator}{block_type}"
    assert create_decision_key(file_path_sep, block_id, block_type) == expected_key_sep


def test_parse_decision_key() -> None:
    """测试解析决策键的功能（现在使用 rsplit 逻辑）。"""
    separator = constants.DECISION_KEY_SEPARATOR
    file_path_val = "/path/to/document.md"
    block_id_val = "block123"
    block_type_val = "NarrativeText"
    key = f"{file_path_val}{separator}{block_id_val}{separator}{block_type_val}"
    file_path, block_id, block_type = parse_decision_key(key)
    assert file_path == file_path_val
    assert block_id == block_id_val
    assert block_type == block_type_val

    # 测试包含分隔符的文件路径（rsplit应该能正确处理）
    file_path_sep_val = f"/path/{separator}/doc.md"
    key_sep = f"{file_path_sep_val}{separator}{block_id_val}{separator}{block_type_val}"
    file_path_sep, block_id_sep, block_type_sep = parse_decision_key(key_sep)
    assert file_path_sep == file_path_sep_val # rsplit 将文件路径部分正确保留
    assert block_id_sep == block_id_val
    assert block_type_sep == block_type_val

    

    # 测试无效键 (分隔符不足)
    invalid_key_less = f"/path/to/document.md{separator}block123"
    file_path_less, block_id_less, block_type_less = parse_decision_key(invalid_key_less)
    assert file_path_less is None
    assert block_id_less is None
    assert block_type_less is None

    # 测试无效键 (过多分隔符，rsplit 会按规则分割) # <--- 修改注释
    invalid_key_more = f"a{separator}b{separator}c{separator}d" # "a::b::c::d"
    file_path_more, block_id_more, block_type_more = parse_decision_key(invalid_key_more)
    # 断言 rsplit 的实际结果
    assert file_path_more == f"a{separator}b" # 即 "a::b"
    assert block_id_more == "c"
    assert block_type_more == "d"

    # 测试空字符串输入
    file_path_empty, block_id_empty, block_type_empty = parse_decision_key("")
    assert file_path_empty is None
    assert block_id_empty is None
    assert block_type_empty is None

    # 测试 None 输入
    file_path_none, block_id_none, block_type_none = parse_decision_key(None) # type: ignore
    assert file_path_none is None
    assert block_id_none is None
    assert block_type_none is None


# --- 测试 extract_text_from_children (已跳过) ---
# @pytest.mark.skip(reason="Skipping test_extract_text_from_children as function was removed/refactored")
# def test_extract_text_from_children() -> None:
#     """测试从子元素提取文本的功能。"""
#     pass # 函数已移除，测试跳过

# --- 测试 display_block_preview ---
def test_display_block_preview() -> None:
    """测试生成块内容预览的功能。"""
    # 使用正确的常量名 PREVIEW_LENGTH
    max_len = constants.PREVIEW_LENGTH
    long_text = "A" * (max_len + 20) # 确保足够长
    expected_long_preview = long_text[:max_len] + "..." # 截断并加省略号

    test_cases = [
        ("Short text", "Short text"),
        (long_text, expected_long_preview),
        ("Text with\nnewlines", "Text with newlines"), # 换行符被替换为空格
        ("", ""), # 空字符串
        (None, "[无内容]"), # None 输入
        ("A" * max_len, "A" * max_len), # 正好等于最大长度
        ("A" * (max_len - 1), "A" * (max_len - 1)), # 小于最大长度
        ("A" * (max_len + 1), "A" * max_len + "..."), # 超出1个字符
    ]

    for text, expected in test_cases:
        result = display_block_preview(text) # 默认使用 constants.PREVIEW_LENGTH
        assert result == expected

# --- 测试 get_markdown_parser (已移除) ---
# def test_get_markdown_parser() -> None: 函数已移除


# --- 测试 sort_blocks_key ---
def test_sort_blocks_key() -> None:
    """测试用于排序块的键函数（使用模拟对象）。"""
    # 使用 MagicMock 模拟具有 file_path 和 block_id 属性的对象
    # block_id 尝试使用数字或包含数字的字符串
    block1 = MagicMock(file_path="/path/a.md", block_id="id-10")
    block2 = MagicMock(file_path="/path/b.md", block_id="id-5")
    block3 = MagicMock(file_path="/path/a.md", block_id="id-2") # a.md, id 2
    block4 = MagicMock(file_path="/path/a.md", block_id="alpha") # a.md, 非数字ID
    block5 = MagicMock(file_path="/path/b.md", block_id="id-15")
    # 测试 block_id 为纯数字的情况
    block6 = MagicMock(file_path="/path/a.md", block_id=1) # a.md, id 1

    blocks = [block1, block2, block3, block4, block5, block6]

    # 预期排序：先按文件路径，再按 block_id (数字优先，然后字符串)
    # /path/a.md, 1
    # /path/a.md, id-2
    # /path/a.md, id-10
    # /path/a.md, alpha (字符串按字典序排在数字后)
    # /path/b.md, id-5
    # /path/b.md, id-15
    expected_order = [block6, block3, block1, block4, block2, block5]

    sorted_blocks = sorted(blocks, key=sort_blocks_key)

    assert sorted_blocks == expected_order