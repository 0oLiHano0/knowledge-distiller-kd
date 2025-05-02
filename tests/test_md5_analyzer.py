# tests/test_md5_analyzer.py
"""
测试 MD5 分析器模块。
"""

import pytest
from unittest.mock import Mock, MagicMock # 导入 MagicMock
from typing import List, Tuple, Any, Type # 添加 Type

# 导入需要测试的类和函数
from knowledge_distiller_kd.core.md5_analyzer import MD5Analyzer
from knowledge_distiller_kd.core.document_processor import ContentBlock
from knowledge_distiller_kd.core import constants
from knowledge_distiller_kd.core.utils import create_decision_key # 导入 create_decision_key

# 导入 unstructured 元素类型用于创建测试数据
from unstructured.documents.elements import Element, NarrativeText, Title, CodeSnippet

# --- Fixtures ---

@pytest.fixture
def mock_kd_tool() -> MagicMock:
    """创建一个模拟的 KDToolCLI 实例"""
    tool = MagicMock()
    tool.blocks_data = [] # 初始化为空列表
    tool.block_decisions = {} # 初始化为空字典
    return tool

@pytest.fixture
def md5_analyzer(mock_kd_tool: MagicMock) -> MD5Analyzer:
    """创建一个 MD5Analyzer 实例，注入模拟的 KDToolCLI"""
    return MD5Analyzer(mock_kd_tool)

# 辅助函数或 fixture 来创建 ContentBlock (使用真实 Element)
@pytest.fixture
def mock_element():
    """提供一个创建真实 Element 的辅助函数"""
    def _create_element(text: str, element_type: Type[Element] = NarrativeText, element_id: str = "test_id") -> Element:
        if issubclass(element_type, (NarrativeText, Title, CodeSnippet)):
            return element_type(text=text, element_id=element_id)
        else:
            from unstructured.documents.elements import Text
            return Text(text=text, element_id=element_id)
    return _create_element

# --- 测试用例 ---

def test_initialization(md5_analyzer: MD5Analyzer, mock_kd_tool: MagicMock) -> None:
    """测试 MD5Analyzer 初始化"""
    assert md5_analyzer.kd_tool == mock_kd_tool
    assert md5_analyzer.md5_duplicates == []

def test_find_md5_duplicates_empty_blocks(md5_analyzer: MD5Analyzer) -> None:
    """测试没有内容块的情况"""
    md5_analyzer.kd_tool.blocks_data = []
    result = md5_analyzer.find_md5_duplicates()
    assert result is True
    assert len(md5_analyzer.md5_duplicates) == 0

def test_find_md5_duplicates_single_block(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """测试只有一个内容块的情况"""
    element = mock_element("只有一个块")
    block = ContentBlock(element, "test.md")
    # 初始化决策
    key = create_decision_key(block.file_path, block.block_id, block.block_type)
    md5_analyzer.kd_tool.block_decisions[key] = constants.DECISION_UNDECIDED
    md5_analyzer.kd_tool.blocks_data = [block]
    result = md5_analyzer.find_md5_duplicates()
    assert result is True
    assert len(md5_analyzer.md5_duplicates) == 0

def test_find_md5_duplicates_identical_blocks(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """测试两个完全相同的内容块"""
    element1 = mock_element("相同内容", NarrativeText, "1")
    element2 = mock_element("相同内容", NarrativeText, "2")
    block1 = ContentBlock(element1, "file1.md")
    block2 = ContentBlock(element2, "file2.md")

    # ==================== 修改：预设决策状态 ====================
    key1 = create_decision_key(block1.file_path, block1.block_id, block1.block_type)
    key2 = create_decision_key(block2.file_path, block2.block_id, block2.block_type)
    md5_analyzer.kd_tool.block_decisions = {
        key1: constants.DECISION_UNDECIDED,
        key2: constants.DECISION_UNDECIDED
    }
    # ============================================================

    md5_analyzer.kd_tool.blocks_data = [block1, block2]
    md5_analyzer.find_md5_duplicates()
    assert len(md5_analyzer.md5_duplicates) == 1
    assert len(md5_analyzer.md5_duplicates[0]) == 2
    # 验证自动决策
    assert md5_analyzer.kd_tool.block_decisions.get(key1) == constants.DECISION_KEEP
    assert md5_analyzer.kd_tool.block_decisions.get(key2) == constants.DECISION_DELETE

def test_find_md5_duplicates_different_blocks(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """测试两个不同内容块"""
    element1 = mock_element("内容一", NarrativeText, "1")
    element2 = mock_element("内容二", NarrativeText, "2")
    block1 = ContentBlock(element1, "test.md")
    block2 = ContentBlock(element2, "test.md")
    # 初始化决策
    key1 = create_decision_key(block1.file_path, block1.block_id, block1.block_type)
    key2 = create_decision_key(block2.file_path, block2.block_id, block2.block_type)
    md5_analyzer.kd_tool.block_decisions = {
        key1: constants.DECISION_UNDECIDED,
        key2: constants.DECISION_UNDECIDED
    }
    md5_analyzer.kd_tool.blocks_data = [block1, block2]
    md5_analyzer.find_md5_duplicates()
    assert len(md5_analyzer.md5_duplicates) == 0

def test_find_md5_duplicates_mixed_blocks(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """测试混合内容块（部分重复）"""
    element1 = mock_element("重复内容", NarrativeText, "1")
    element2 = mock_element("不同内容", NarrativeText, "2")
    element3 = mock_element("重复内容", NarrativeText, "3")
    block1 = ContentBlock(element1, "a.md")
    block2 = ContentBlock(element2, "b.md")
    block3 = ContentBlock(element3, "c.md")

    # ==================== 修改：预设决策状态 ====================
    key1 = create_decision_key(block1.file_path, block1.block_id, block1.block_type)
    key2 = create_decision_key(block2.file_path, block2.block_id, block2.block_type)
    key3 = create_decision_key(block3.file_path, block3.block_id, block3.block_type)
    md5_analyzer.kd_tool.block_decisions = {
        key1: constants.DECISION_UNDECIDED,
        key2: constants.DECISION_UNDECIDED,
        key3: constants.DECISION_UNDECIDED
    }
    # ============================================================

    md5_analyzer.kd_tool.blocks_data = [block1, block2, block3]
    md5_analyzer.find_md5_duplicates()
    assert len(md5_analyzer.md5_duplicates) == 1
    assert len(md5_analyzer.md5_duplicates[0]) == 2
    duplicate_ids = {b.block_id for b in md5_analyzer.md5_duplicates[0]}
    assert duplicate_ids == {"1", "3"}
    # 验证决策
    assert md5_analyzer.kd_tool.block_decisions.get(key1) == constants.DECISION_KEEP
    assert md5_analyzer.kd_tool.block_decisions.get(key3) == constants.DECISION_DELETE
    assert md5_analyzer.kd_tool.block_decisions.get(key2) == constants.DECISION_UNDECIDED # 确认未被修改

def test_find_md5_duplicates_different_types(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """测试相同文本但不同类型的内容块"""
    element1 = mock_element("相同文本", NarrativeText, "1")
    element2 = mock_element("相同文本", Title, "2")
    block1 = ContentBlock(element1, "test.md")
    block2 = ContentBlock(element2, "test.md")
    key1 = create_decision_key(block1.file_path, block1.block_id, block1.block_type)
    key2 = create_decision_key(block2.file_path, block2.block_id, block2.block_type)
    md5_analyzer.kd_tool.block_decisions = {key1: 'undecided', key2: 'undecided'}
    md5_analyzer.kd_tool.blocks_data = [block1, block2]
    md5_analyzer.find_md5_duplicates()
    assert len(md5_analyzer.md5_duplicates) == 0

def test_find_md5_duplicates_empty_content(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """测试空内容块"""
    element1 = mock_element("", NarrativeText, "1")
    element2 = mock_element("", NarrativeText, "2")
    block1 = ContentBlock(element1, "a.md")
    block2 = ContentBlock(element2, "b.md")
    key1 = create_decision_key(block1.file_path, block1.block_id, block1.block_type)
    key2 = create_decision_key(block2.file_path, block2.block_id, block2.block_type)
    md5_analyzer.kd_tool.block_decisions = {key1: 'undecided', key2: 'undecided'}
    md5_analyzer.kd_tool.blocks_data = [block1, block2]
    md5_analyzer.find_md5_duplicates()
    assert len(md5_analyzer.md5_duplicates) == 1
    assert len(md5_analyzer.md5_duplicates[0]) == 2

def test_find_md5_duplicates_whitespace(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """测试仅包含空白字符的内容块"""
    element1 = mock_element("   \n\t ", NarrativeText, "1")
    element2 = mock_element(" \t\n  ", NarrativeText, "2")
    block1 = ContentBlock(element1, "a.md")
    block2 = ContentBlock(element2, "b.md")
    key1 = create_decision_key(block1.file_path, block1.block_id, block1.block_type)
    key2 = create_decision_key(block2.file_path, block2.block_id, block2.block_type)
    md5_analyzer.kd_tool.block_decisions = {key1: 'undecided', key2: 'undecided'}
    md5_analyzer.kd_tool.blocks_data = [block1, block2]
    md5_analyzer.find_md5_duplicates()
    assert len(md5_analyzer.md5_duplicates) == 1
    assert len(md5_analyzer.md5_duplicates[0]) == 2

def test_find_md5_duplicates_skip_headers(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """测试标题块的比较（应该基于移除 # 后的文本）。"""
    element1 = mock_element("# 标题\n\n这是内容", Title, "1")
    element2 = mock_element("## 另一个标题\n\n这是内容", Title, "2")
    block1 = ContentBlock(element1, "test.md")
    block2 = ContentBlock(element2, "test.md")
    key1 = create_decision_key(block1.file_path, block1.block_id, block1.block_type)
    key2 = create_decision_key(block2.file_path, block2.block_id, block2.block_type)
    md5_analyzer.kd_tool.block_decisions = {key1: 'undecided', key2: 'undecided'}
    md5_analyzer.kd_tool.blocks_data = [block1, block2]
    md5_analyzer.find_md5_duplicates()
    assert len(md5_analyzer.md5_duplicates) == 0

def test_find_md5_duplicates_normalize_text(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """测试标准化是否影响比较"""
    element1 = mock_element("文本  包含\n多余 空白", NarrativeText, "1")
    element2 = mock_element("文本 包含 多余 空白", NarrativeText, "2")
    block1 = ContentBlock(element1, "a.md")
    block2 = ContentBlock(element2, "b.md")

    # ==================== 修改：预设决策状态 ====================
    key1 = create_decision_key(block1.file_path, block1.block_id, block1.block_type)
    key2 = create_decision_key(block2.file_path, block2.block_id, block2.block_type)
    md5_analyzer.kd_tool.block_decisions = {
        key1: constants.DECISION_UNDECIDED,
        key2: constants.DECISION_UNDECIDED
    }
    # ============================================================

    md5_analyzer.kd_tool.blocks_data = [block1, block2]
    md5_analyzer.find_md5_duplicates()
    # ==================== 修改：断言应找到重复 ====================
    # 因为 document_processor.py 中的 normalize 已修改为替换换行符
    assert len(md5_analyzer.md5_duplicates) == 1
    # ============================================================
    assert len(md5_analyzer.md5_duplicates[0]) == 2
    assert md5_analyzer.kd_tool.block_decisions.get(key1) == constants.DECISION_KEEP
    assert md5_analyzer.kd_tool.block_decisions.get(key2) == constants.DECISION_DELETE


def test_find_md5_duplicates_different_content(md5_analyzer: MD5Analyzer, mock_element) -> None:
    """测试包含相同前缀但整体不同的内容块"""
    element1 = mock_element("这是第一部分", NarrativeText, "1")
    element2 = mock_element("这是第二部分", NarrativeText, "2")
    block1 = ContentBlock(element1, "test.md")
    block2 = ContentBlock(element2, "test.md")
    key1 = create_decision_key(block1.file_path, block1.block_id, block1.block_type)
    key2 = create_decision_key(block2.file_path, block2.block_id, block2.block_type)
    md5_analyzer.kd_tool.block_decisions = {key1: 'undecided', key2: 'undecided'}
    md5_analyzer.kd_tool.blocks_data = [block1, block2]
    md5_analyzer.find_md5_duplicates()
    assert len(md5_analyzer.md5_duplicates) == 0
