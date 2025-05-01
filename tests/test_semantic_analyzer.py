"""
语义分析器测试模块。

此模块包含对SemanticAnalyzer类的单元测试。
"""

import pytest
from pathlib import Path
from typing import List, Tuple, Dict
from unittest.mock import MagicMock, patch

from knowledge_distiller_kd.core.semantic_analyzer import SemanticAnalyzer
from tests.test_data_generator import test_data_generator
from tests.test_utils import (
    verify_file_content,
    verify_decision_file,
    cleanup_test_environment
)

# 移除旧的导入
# from semantic_analyzer import SemanticAnalyzer as OldSemanticAnalyzer, SENTENCE_TRANSFORMERS_AVAILABLE

# 添加新的常量
SENTENCE_TRANSFORMERS_AVAILABLE = True  # 临时设置为True，实际应该检查是否安装了sentence-transformers

@pytest.fixture
def mock_kd_tool() -> MagicMock:
    """
    创建一个模拟的 KDToolCLI 实例。
    """
    mock = MagicMock()
    mock.skip_semantic = False
    mock.similarity_threshold = 0.85
    mock.blocks_data = []
    mock.block_decisions = {}
    return mock

@pytest.fixture
def semantic_analyzer(mock_kd_tool) -> SemanticAnalyzer:
    """
    创建一个 SemanticAnalyzer 实例用于测试。
    """
    return SemanticAnalyzer(mock_kd_tool)

def test_initialization(semantic_analyzer: SemanticAnalyzer) -> None:
    """
    测试 SemanticAnalyzer 的初始化。
    """
    assert semantic_analyzer.model is None
    assert semantic_analyzer.semantic_duplicates == []
    assert semantic_analyzer.semantic_id_to_key == {}
    assert semantic_analyzer.vector_cache == {}
    assert semantic_analyzer._model_loaded is False
    assert semantic_analyzer.model_name == "paraphrase-multilingual-MiniLM-L12-v2"
    assert semantic_analyzer.similarity_threshold == 0.85

@patch('knowledge_distiller_kd.core.semantic_analyzer.SentenceTransformer')
def test_load_semantic_model(mock_sentence_transformer: MagicMock, semantic_analyzer: SemanticAnalyzer) -> None:
    """
    测试加载语义模型的功能。
    """
    # 模拟模型加载成功
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 768
    mock_sentence_transformer.return_value = mock_model
    
    semantic_analyzer.load_semantic_model()
    
    assert semantic_analyzer.model == mock_model
    assert semantic_analyzer._model_loaded is True
    assert semantic_analyzer.model_version == 768
    mock_sentence_transformer.assert_called_once_with("paraphrase-multilingual-MiniLM-L12-v2")

def test_load_semantic_model_skip(semantic_analyzer: SemanticAnalyzer) -> None:
    """
    测试跳过语义模型加载的情况。
    """
    semantic_analyzer.tool.skip_semantic = True
    semantic_analyzer.load_semantic_model()
    assert semantic_analyzer.model is None
    assert semantic_analyzer._model_loaded is False

@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
def test_find_semantic_duplicates(semantic_analyzer: SemanticAnalyzer) -> None:
    """
    测试查找语义重复的功能。
    """
    # 准备测试数据
    semantic_analyzer.tool.blocks_data = [
        ("file1.md", 1, "paragraph", "这是一个测试段落。"),
        ("file2.md", 1, "paragraph", "这是另一个测试段落。"),
        ("file3.md", 1, "paragraph", "这是一个完全不同的段落。")
    ]
    
    # 加载模型
    semantic_analyzer.load_semantic_model()
    
    # 执行查找
    semantic_analyzer.find_semantic_duplicates()
    
    # 验证结果
    assert isinstance(semantic_analyzer.semantic_duplicates, list)
    assert semantic_analyzer._model_loaded is True
    assert semantic_analyzer.vector_cache != {}

def test_find_semantic_duplicates_no_model(semantic_analyzer: SemanticAnalyzer) -> None:
    """
    测试在没有模型的情况下查找语义重复。
    """
    semantic_analyzer.model = None
    semantic_analyzer.find_semantic_duplicates()
    assert semantic_analyzer.semantic_duplicates == []
    assert semantic_analyzer._model_loaded is False

def test_find_semantic_duplicates_insufficient_blocks(semantic_analyzer: SemanticAnalyzer) -> None:
    """
    测试在块数量不足的情况下查找语义重复。
    """
    semantic_analyzer.tool.blocks_data = [("file1.md", 1, "paragraph", "这是一个测试段落。")]
    semantic_analyzer.find_semantic_duplicates()
    assert semantic_analyzer.semantic_duplicates == []

def test_display_semantic_duplicates_list(semantic_analyzer: SemanticAnalyzer) -> None:
    """
    测试显示语义重复列表的功能。
    """
    # 准备测试数据
    semantic_analyzer.semantic_duplicates = [
        (("file1.md", 1, "paragraph", "内容1"), ("file2.md", 1, "paragraph", "内容2"), 0.9)
    ]
    
    # 测试显示
    result = semantic_analyzer._display_semantic_duplicates_list()
    assert result is True
    assert len(semantic_analyzer.semantic_id_to_key) > 0

def test_display_semantic_duplicates_list_empty(semantic_analyzer: SemanticAnalyzer) -> None:
    """
    测试显示空语义重复列表的情况。
    """
    semantic_analyzer.semantic_duplicates = []
    result = semantic_analyzer._display_semantic_duplicates_list()
    assert result is False

def test_display_semantic_duplicates_list_skip(semantic_analyzer: SemanticAnalyzer) -> None:
    """
    测试跳过语义分析时显示列表的情况。
    """
    semantic_analyzer.tool.skip_semantic = True
    result = semantic_analyzer._display_semantic_duplicates_list()
    assert result is False 