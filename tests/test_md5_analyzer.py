"""
MD5分析器测试模块。

此模块包含对MD5Analyzer类的单元测试。
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import List, Tuple, Dict

from knowledge_distiller_kd.core.md5_analyzer import MD5Analyzer
from test_data_generator import test_data_generator
from test_utils import (
    verify_file_content,
    verify_decision_file,
    cleanup_test_environment
)

@pytest.fixture
def mock_kd_tool() -> MagicMock:
    """
    创建一个模拟的 KDToolCLI 实例。
    """
    mock = MagicMock()
    mock.blocks_data = []
    return mock

@pytest.fixture
def md5_analyzer(mock_kd_tool: MagicMock) -> MD5Analyzer:
    """
    创建一个 MD5Analyzer 实例。
    """
    return MD5Analyzer(mock_kd_tool)

def test_initialization(md5_analyzer: MD5Analyzer) -> None:
    """
    测试 MD5Analyzer 的初始化。
    """
    assert md5_analyzer.tool is not None
    assert md5_analyzer.duplicate_blocks == {}
    assert md5_analyzer.md5_id_to_key == {}

def test_find_md5_duplicates_empty_blocks(md5_analyzer: MD5Analyzer) -> None:
    """
    测试在没有块数据的情况下查找 MD5 重复。
    """
    md5_analyzer.find_md5_duplicates()
    assert md5_analyzer.duplicate_blocks == {}
    assert md5_analyzer.md5_id_to_key == {}

def test_find_md5_duplicates_single_block(md5_analyzer: MD5Analyzer) -> None:
    """
    测试在只有一个块的情况下查找 MD5 重复。
    """
    test_file = Path("test.md")
    md5_analyzer.tool.blocks_data = [(test_file, 1, "paragraph", "Test content")]
    md5_analyzer.find_md5_duplicates()
    assert md5_analyzer.duplicate_blocks == {}
    assert len(md5_analyzer.md5_id_to_key) == 1

def test_find_md5_duplicates_identical_blocks(md5_analyzer: MD5Analyzer) -> None:
    """
    测试在存在相同内容块的情况下查找 MD5 重复。
    """
    test_file1 = Path("test1.md")
    test_file2 = Path("test2.md")
    content = "Test content"
    md5_analyzer.tool.blocks_data = [
        (test_file1, 1, "paragraph", content),
        (test_file2, 1, "paragraph", content)
    ]
    md5_analyzer.find_md5_duplicates()
    assert len(md5_analyzer.duplicate_blocks) == 1
    assert len(md5_analyzer.md5_id_to_key) == 2

def test_find_md5_duplicates_different_blocks(md5_analyzer: MD5Analyzer) -> None:
    """
    测试在不同内容块的情况下查找 MD5 重复。
    """
    test_file = Path("test.md")
    md5_analyzer.tool.blocks_data = [
        (test_file, 1, "paragraph", "Content 1"),
        (test_file, 2, "paragraph", "Content 2")
    ]
    md5_analyzer.find_md5_duplicates()
    assert md5_analyzer.duplicate_blocks == {}
    assert len(md5_analyzer.md5_id_to_key) == 2

def test_find_md5_duplicates_mixed_blocks(md5_analyzer: MD5Analyzer) -> None:
    """
    测试在混合内容块的情况下查找 MD5 重复。
    """
    test_file1 = Path("test1.md")
    test_file2 = Path("test2.md")
    test_file3 = Path("test3.md")
    content1 = "Content 1"
    content2 = "Content 2"
    md5_analyzer.tool.blocks_data = [
        (test_file1, 1, "paragraph", content1),
        (test_file2, 1, "paragraph", content1),
        (test_file3, 1, "paragraph", content2)
    ]
    md5_analyzer.find_md5_duplicates()
    assert len(md5_analyzer.duplicate_blocks) == 1
    assert len(md5_analyzer.md5_id_to_key) == 3

def test_find_md5_duplicates_different_types(md5_analyzer: MD5Analyzer) -> None:
    """
    测试在不同类型块的情况下查找 MD5 重复。
    """
    test_file = Path("test.md")
    content = "Test content"
    md5_analyzer.tool.blocks_data = [
        (test_file, 1, "paragraph", content),
        (test_file, 2, "list", content)
    ]
    md5_analyzer.find_md5_duplicates()
    assert md5_analyzer.duplicate_blocks == {}
    assert len(md5_analyzer.md5_id_to_key) == 2

def test_find_md5_duplicates_empty_content(md5_analyzer: MD5Analyzer) -> None:
    """
    测试在空内容块的情况下查找 MD5 重复。
    """
    test_file = Path("test.md")
    md5_analyzer.tool.blocks_data = [
        (test_file, 1, "paragraph", ""),
        (test_file, 2, "paragraph", "")
    ]
    md5_analyzer.find_md5_duplicates()
    assert len(md5_analyzer.duplicate_blocks) == 1
    assert len(md5_analyzer.md5_id_to_key) == 2

def test_find_md5_duplicates_whitespace(md5_analyzer: MD5Analyzer) -> None:
    """
    测试在包含空白字符的内容块的情况下查找 MD5 重复。
    由于我们现在的实现会标准化空白字符，所以这些块应该被视为重复。
    """
    test_file = Path("test.md")
    content1 = "Test content"
    content2 = "Test content  "
    md5_analyzer.tool.blocks_data = [
        (test_file, 1, "paragraph", content1),
        (test_file, 2, "paragraph", content2)
    ]
    md5_analyzer.find_md5_duplicates()
    # 由于标准化处理，这两个块应该被视为重复
    assert len(md5_analyzer.duplicate_blocks) == 1
    assert len(md5_analyzer.md5_id_to_key) == 2

def test_find_md5_duplicates_skip_headers(md5_analyzer: MD5Analyzer) -> None:
    """
    测试跳过标题的情况。
    """
    test_file = Path("test.md")
    content1 = "# 标题\n\n这是内容"
    content2 = "## 另一个标题\n\n这是内容"
    md5_analyzer.tool.blocks_data = [
        (test_file, 1, "content", content1),
        (test_file, 2, "content", content2)
    ]
    md5_analyzer.find_md5_duplicates()
    # 由于标题被跳过，这两个块应该被视为重复
    assert len(md5_analyzer.duplicate_blocks) == 1
    assert len(md5_analyzer.md5_id_to_key) == 2

def test_find_md5_duplicates_normalize_text(md5_analyzer: MD5Analyzer) -> None:
    """
    测试文本标准化功能。
    """
    test_file = Path("test.md")
    content1 = "这是内容，包含标点符号！"
    content2 = "这是内容，包含标点符号！"
    content3 = "这是内容，包含标点符号！"
    md5_analyzer.tool.blocks_data = [
        (test_file, 1, "content", content1),
        (test_file, 2, "content", content2),
        (test_file, 3, "content", content3)
    ]
    md5_analyzer.find_md5_duplicates()
    # 这三个块应该被视为重复
    assert len(md5_analyzer.duplicate_blocks) == 1
    assert len(md5_analyzer.md5_id_to_key) == 3

def test_find_md5_duplicates_different_content(md5_analyzer: MD5Analyzer) -> None:
    """
    测试不同内容的情况。
    """
    test_file = Path("test.md")
    content1 = "这是第一段内容"
    content2 = "这是第二段内容"
    md5_analyzer.tool.blocks_data = [
        (test_file, 1, "content", content1),
        (test_file, 2, "content", content2)
    ]
    md5_analyzer.find_md5_duplicates()
    # 这两个块不应该被视为重复
    assert len(md5_analyzer.duplicate_blocks) == 0
    assert len(md5_analyzer.md5_id_to_key) == 2 