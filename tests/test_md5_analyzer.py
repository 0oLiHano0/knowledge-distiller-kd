"""
MD5分析器测试模块。

此模块包含对MD5Analyzer类的单元测试。
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import List, Tuple, Dict
from unstructured.documents.elements import Title, NarrativeText, ListItem, CodeSnippet, Table, Text

from knowledge_distiller_kd.core.md5_analyzer import MD5Analyzer
from knowledge_distiller_kd.core.document_processor import ContentBlock

@pytest.fixture
def mock_kd_tool() -> MagicMock:
    """
    创建一个模拟的 KDToolCLI 实例。
    """
    mock = MagicMock()
    mock.blocks_data = []
    mock.block_decisions = {}
    return mock

@pytest.fixture
def md5_analyzer(mock_kd_tool: MagicMock) -> MD5Analyzer:
    """
    创建一个 MD5Analyzer 实例。
    """
    return MD5Analyzer(mock_kd_tool)

@pytest.fixture
def mock_element():
    """
    创建一个模拟元素的函数。
    """
    def _create_element(text: str, element_type: type, element_id: str = "test_id"):
        # 直接创建对应类型的元素
        if element_type == Title:
            return Title(text=text, element_id=element_id)
        elif element_type == NarrativeText:
            return NarrativeText(text=text, element_id=element_id)
        elif element_type == ListItem:
            return ListItem(text=text, element_id=element_id)
        elif element_type == CodeSnippet:
            return CodeSnippet(text=text, element_id=element_id)
        elif element_type == Table:
            return Table(text=text, element_id=element_id)
        else:
            return Text(text=text, element_id=element_id)
    
    return _create_element

def test_initialization(md5_analyzer: MD5Analyzer) -> None:
    """
    测试 MD5Analyzer 的初始化。
    """
    assert md5_analyzer.kd_tool is not None
    assert md5_analyzer.md5_duplicates == []

def test_find_md5_duplicates_empty_blocks(md5_analyzer: MD5Analyzer) -> None:
    """
    测试在没有块数据的情况下查找 MD5 重复。
    """
    md5_analyzer.find_md5_duplicates()
    assert md5_analyzer.md5_duplicates == []

def test_find_md5_duplicates_single_block(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """
    测试在只有一个块的情况下查找 MD5 重复。
    """
    element = mock_element("Test content", NarrativeText)
    block = ContentBlock(element, "test.md")
    md5_analyzer.kd_tool.blocks_data = [block]
    md5_analyzer.find_md5_duplicates()
    assert md5_analyzer.md5_duplicates == []

def test_find_md5_duplicates_identical_blocks(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """
    测试在存在相同内容块的情况下查找 MD5 重复。
    """
    element1 = mock_element("Test content", NarrativeText, "1")
    element2 = mock_element("Test content", NarrativeText, "2")
    block1 = ContentBlock(element1, "test1.md")
    block2 = ContentBlock(element2, "test2.md")
    md5_analyzer.kd_tool.blocks_data = [block1, block2]
    md5_analyzer.find_md5_duplicates()
    assert len(md5_analyzer.md5_duplicates) == 1
    assert len(md5_analyzer.md5_duplicates[0]) == 2

def test_find_md5_duplicates_different_blocks(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """
    测试在不同内容块的情况下查找 MD5 重复。
    """
    element1 = mock_element("Content 1", NarrativeText, "1")
    element2 = mock_element("Content 2", NarrativeText, "2")
    block1 = ContentBlock(element1, "test.md")
    block2 = ContentBlock(element2, "test.md")
    md5_analyzer.kd_tool.blocks_data = [block1, block2]
    md5_analyzer.find_md5_duplicates()
    assert md5_analyzer.md5_duplicates == []

def test_find_md5_duplicates_mixed_blocks(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """
    测试在混合内容块的情况下查找 MD5 重复。
    """
    element1 = mock_element("Content 1", NarrativeText, "1")
    element2 = mock_element("Content 1", NarrativeText, "2")
    element3 = mock_element("Content 2", NarrativeText, "3")
    block1 = ContentBlock(element1, "test1.md")
    block2 = ContentBlock(element2, "test2.md")
    block3 = ContentBlock(element3, "test3.md")
    md5_analyzer.kd_tool.blocks_data = [block1, block2, block3]
    md5_analyzer.find_md5_duplicates()
    assert len(md5_analyzer.md5_duplicates) == 1
    assert len(md5_analyzer.md5_duplicates[0]) == 2

def test_find_md5_duplicates_different_types(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """
    测试在不同类型块的情况下查找 MD5 重复。
    """
    element1 = mock_element("Test content", NarrativeText, "1")
    element2 = mock_element("Test content", ListItem, "2")
    block1 = ContentBlock(element1, "test.md")
    block2 = ContentBlock(element2, "test.md")
    md5_analyzer.kd_tool.blocks_data = [block1, block2]
    md5_analyzer.find_md5_duplicates()
    assert md5_analyzer.md5_duplicates == []

def test_find_md5_duplicates_empty_content(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """
    测试在空内容块的情况下查找 MD5 重复。
    """
    element1 = mock_element("", NarrativeText, "1")
    element2 = mock_element("", NarrativeText, "2")
    block1 = ContentBlock(element1, "test.md")
    block2 = ContentBlock(element2, "test.md")
    md5_analyzer.kd_tool.blocks_data = [block1, block2]
    md5_analyzer.find_md5_duplicates()
    assert len(md5_analyzer.md5_duplicates) == 1
    assert len(md5_analyzer.md5_duplicates[0]) == 2

def test_find_md5_duplicates_whitespace(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """
    测试在包含空白字符的内容块的情况下查找 MD5 重复。
    由于我们现在的实现会标准化空白字符，所以这些块应该被视为重复。
    """
    element1 = mock_element("Test content", NarrativeText, "1")
    element2 = mock_element("Test content  ", NarrativeText, "2")
    block1 = ContentBlock(element1, "test.md")
    block2 = ContentBlock(element2, "test.md")
    md5_analyzer.kd_tool.blocks_data = [block1, block2]
    md5_analyzer.find_md5_duplicates()
    assert len(md5_analyzer.md5_duplicates) == 1
    assert len(md5_analyzer.md5_duplicates[0]) == 2

def test_find_md5_duplicates_skip_headers(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """
    测试跳过标题的情况。
    """
    element1 = mock_element("# 标题\n\n这是内容", Title, "1")
    element2 = mock_element("## 另一个标题\n\n这是内容", Title, "2")
    block1 = ContentBlock(element1, "test.md")
    block2 = ContentBlock(element2, "test.md")
    md5_analyzer.kd_tool.blocks_data = [block1, block2]
    md5_analyzer.find_md5_duplicates()
    assert len(md5_analyzer.md5_duplicates) == 1
    assert len(md5_analyzer.md5_duplicates[0]) == 2

def test_find_md5_duplicates_normalize_text(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """
    测试文本标准化功能。
    """
    element1 = mock_element("这是内容，包含标点符号！", NarrativeText, "1")
    element2 = mock_element("这是内容，包含标点符号！", NarrativeText, "2")
    element3 = mock_element("这是内容，包含标点符号！", NarrativeText, "3")
    block1 = ContentBlock(element1, "test.md")
    block2 = ContentBlock(element2, "test.md")
    block3 = ContentBlock(element3, "test.md")
    md5_analyzer.kd_tool.blocks_data = [block1, block2, block3]
    md5_analyzer.find_md5_duplicates()
    assert len(md5_analyzer.md5_duplicates) == 1
    assert len(md5_analyzer.md5_duplicates[0]) == 3

def test_find_md5_duplicates_different_content(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """
    测试不同内容的情况。
    """
    element1 = mock_element("这是第一段内容", NarrativeText, "1")
    element2 = mock_element("这是第二段内容", NarrativeText, "2")
    block1 = ContentBlock(element1, "test.md")
    block2 = ContentBlock(element2, "test.md")
    md5_analyzer.kd_tool.blocks_data = [block1, block2]
    md5_analyzer.find_md5_duplicates()
    assert md5_analyzer.md5_duplicates == [] 