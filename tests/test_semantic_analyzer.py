# tests/test_semantic_analyzer.py
"""
测试语义分析器模块。
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch # 确保导入 patch
from typing import List, Tuple, Any, Type

# 导入需要测试的类和函数
from knowledge_distiller_kd.core.semantic_analyzer import SemanticAnalyzer, SENTENCE_TRANSFORMERS_AVAILABLE
from knowledge_distiller_kd.core.document_processor import ContentBlock
from knowledge_distiller_kd.core import constants
from knowledge_distiller_kd.core.utils import create_decision_key # 导入

# 导入 unstructured 元素类型用于创建测试数据
from unstructured.documents.elements import Element, NarrativeText, Title, CodeSnippet

# --- Fixtures ---

@pytest.fixture
def mock_kd_tool() -> MagicMock:
    """创建一个模拟的 KDToolCLI 实例"""
    tool = MagicMock()
    tool.blocks_data = []
    tool.block_decisions = {}
    tool.skip_semantic = False # 默认不跳过
    return tool

# 模拟 SentenceTransformer 模型
@pytest.fixture
def mock_sentence_transformer() -> MagicMock:
    """创建一个模拟的 SentenceTransformer 模型实例"""
    model = MagicMock()
    dummy_vector_1 = np.array([0.1, 0.2, 0.3])
    dummy_vector_2 = np.array([0.4, 0.5, 0.6])
    dummy_vector_similar = np.array([0.11, 0.21, 0.31]) # 与1相似
    default_vector = np.array([0.0, 0.0, 0.0])

    # ==================== 修改：确保 mock_encode 返回正确形状的 NumPy 数组 ====================
    def mock_encode(texts, batch_size=32, show_progress_bar=False):
        results = []
        for text in texts:
            vec = default_vector
            if "相似1" in text: vec = dummy_vector_similar
            elif "文本1" in text: vec = dummy_vector_1
            elif "文本2" in text: vec = dummy_vector_2
            results.append(np.array(vec).reshape(3,)) # 确保是 (3,)
        # 返回一个 NumPy 数组，其第一维是文本数量
        return np.array(results) # Shape: (len(texts), 3)
    # ====================================================================================

    model.encode.side_effect = mock_encode
    # ==================== 修改：明确配置 get_sentence_embedding_dimension ====================
    # 让 mock 对象有一个可调用的 get_sentence_embedding_dimension 方法
    model.get_sentence_embedding_dimension = MagicMock(return_value=3)
    # ====================================================================================
    return model

@pytest.fixture
def semantic_analyzer(mock_kd_tool: MagicMock) -> SemanticAnalyzer:
    """创建一个 SemanticAnalyzer 实例，注入模拟的 KDToolCLI"""
    analyzer = SemanticAnalyzer(mock_kd_tool, similarity_threshold=0.9)
    return analyzer

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

def test_initialization(semantic_analyzer: SemanticAnalyzer, mock_kd_tool: MagicMock) -> None:
    """测试 SemanticAnalyzer 初始化"""
    assert semantic_analyzer.tool == mock_kd_tool
    assert semantic_analyzer.similarity_threshold == 0.9
    assert semantic_analyzer.model is None
    assert semantic_analyzer.semantic_duplicates == []
    assert semantic_analyzer.semantic_id_to_key == {}
    assert semantic_analyzer.vector_cache == {}

# 使用 patch 模拟 SentenceTransformer 的导入和实例化
@patch("knowledge_distiller_kd.core.semantic_analyzer.SentenceTransformer") # 直接 patch 类
def test_load_semantic_model_success(mock_st_class: MagicMock, # 参数名来自 patch
                                     semantic_analyzer: SemanticAnalyzer,
                                     mock_sentence_transformer: MagicMock) -> None:
    """测试成功加载语义模型"""
    # 配置被 patch 的类，使其在实例化时返回我们的 mock 实例
    mock_st_class.return_value = mock_sentence_transformer

    semantic_analyzer.load_semantic_model()

    # 验证 SentenceTransformer 类被调用（实例化）了一次
    mock_st_class.assert_called_once_with(semantic_analyzer.model_name)
    # 验证 model 属性被设置为返回的 mock 实例
    assert semantic_analyzer.model == mock_sentence_transformer
    assert semantic_analyzer._model_loaded is True
    # 验证 model_version 被正确设置 (通过调用 mock 实例的方法)
    assert semantic_analyzer.model_version == 3
    # 验证 mock 实例的方法被调用
    mock_sentence_transformer.get_sentence_embedding_dimension.assert_called_once()


def test_load_semantic_model_skipped(semantic_analyzer: SemanticAnalyzer) -> None:
    """测试跳过加载语义模型"""
    semantic_analyzer.tool.skip_semantic = True
    semantic_analyzer.load_semantic_model()
    assert semantic_analyzer.model is None
    assert semantic_analyzer._model_loaded is False

# 测试加载失败的情况
@patch("knowledge_distiller_kd.core.semantic_analyzer.SentenceTransformer")
def test_load_semantic_model_failure(mock_st_class: MagicMock, # 来自 patch
                                     semantic_analyzer: SemanticAnalyzer) -> None:
    """测试加载模型失败"""
    # ==================== 修改：让 patch 的类在实例化时抛出异常 ====================
    mock_st_class.side_effect = Exception("模型加载错误")
    # ==========================================================================

    semantic_analyzer.load_semantic_model()

    # 验证 SentenceTransformer 类被尝试调用（实例化）
    mock_st_class.assert_called_once()
    # 验证模型加载失败后，skip_semantic 标志被设置
    assert semantic_analyzer.tool.skip_semantic is True
    # 验证 model 属性最终为 None
    assert semantic_analyzer.model is None
    assert semantic_analyzer._model_loaded is False


# --- 测试向量计算 ---
@patch("knowledge_distiller_kd.core.semantic_analyzer.SentenceTransformer")
def test_compute_vectors(mock_st_class: MagicMock, # 来自 patch
                         semantic_analyzer: SemanticAnalyzer,
                         mock_sentence_transformer: MagicMock) -> None:
    """测试向量计算和缓存"""
    mock_st_class.return_value = mock_sentence_transformer
    semantic_analyzer.load_semantic_model()

    texts = ["文本1", "文本2", "文本1"]
    vectors = semantic_analyzer._compute_vectors(texts)

    assert len(vectors) == 3
    assert all(isinstance(v, np.ndarray) for v in vectors)
    # 验证维度
    assert all(v.shape == (3,) for v in vectors) # 修正 mock_encode 后应通过
    # 验证值
    np.testing.assert_array_equal(vectors[0], np.array([0.1, 0.2, 0.3]))
    np.testing.assert_array_equal(vectors[1], np.array([0.4, 0.5, 0.6]))
    np.testing.assert_array_equal(vectors[2], np.array([0.1, 0.2, 0.3]))
    # 验证不同文本的向量是否不同
    assert not np.array_equal(vectors[0], vectors[1])
    # 验证缓存
    mock_sentence_transformer.encode.assert_called_once()
    semantic_analyzer.vector_cache.clear()
    mock_sentence_transformer.encode.reset_mock()
    vectors_again = semantic_analyzer._compute_vectors(texts)
    assert mock_sentence_transformer.encode.call_count == 1
    assert len(semantic_analyzer.vector_cache) == 2

# --- 测试查找语义重复 ---
@patch("knowledge_distiller_kd.core.semantic_analyzer.SentenceTransformer")
@patch("knowledge_distiller_kd.core.semantic_analyzer.st_util") # Patch st_util
def test_find_semantic_duplicates(mock_st_util_patched: MagicMock, # 来自 patch
                                  mock_st_class: MagicMock, # 来自 patch
                                  semantic_analyzer: SemanticAnalyzer,
                                  mock_sentence_transformer: MagicMock,
                                  mock_element) -> None:
    """测试查找语义相似对"""
    mock_st_class.return_value = mock_sentence_transformer
    semantic_analyzer.load_semantic_model()

    element1 = mock_element("文本1", NarrativeText, "id1")
    element2 = mock_element("文本2", NarrativeText, "id2")
    element3 = mock_element("文本1 相似1", NarrativeText, "id3")
    element4 = mock_element("文本4 已删除", NarrativeText, "id4")
    block1 = ContentBlock(element1, "file_a.md")
    block2 = ContentBlock(element2, "file_b.md")
    block3 = ContentBlock(element3, "file_c.md")
    block4 = ContentBlock(element4, "file_d.md")

    semantic_analyzer.tool.blocks_data = [block1, block2, block3, block4]

    key1 = create_decision_key(block1.file_path, block1.block_id, block1.block_type)
    key2 = create_decision_key(block2.file_path, block2.block_id, block2.block_type)
    key3 = create_decision_key(block3.file_path, block3.block_id, block3.block_type)
    key4 = create_decision_key(block4.file_path, block4.block_id, block4.block_type)

    semantic_analyzer.tool.block_decisions = {
        key1: constants.DECISION_UNDECIDED, key2: constants.DECISION_UNDECIDED,
        key3: constants.DECISION_UNDECIDED, key4: constants.DECISION_DELETE
    }

    mock_similarity_matrix = np.array([
        [1.0, 0.2, 0.95], [0.2, 1.0, 0.3], [0.95, 0.3, 1.0]
    ])
    mock_st_util_patched.cos_sim.return_value = mock_similarity_matrix

    semantic_analyzer.find_semantic_duplicates()

    # 验证 encode 被调用 (在 _compute_vectors 内部)
    mock_sentence_transformer.encode.assert_called_once()
    call_args, _ = mock_sentence_transformer.encode.call_args
    assert call_args[0] == ["文本1", "文本2", "文本1 相似1"]

    # 验证 cos_sim 被调用
    mock_st_util_patched.cos_sim.assert_called_once()

    # 验证结果
    assert len(semantic_analyzer.semantic_duplicates) == 1
    found_pair = semantic_analyzer.semantic_duplicates[0]
    assert (found_pair[0] == block1 and found_pair[1] == block3) or \
           (found_pair[0] == block3 and found_pair[1] == block1)
    assert found_pair[2] == pytest.approx(0.95)

# --- 测试交互式处理 (跳过) ---
# @patch('builtins.input', side_effect=['k s1-1', 'q'])
# def test_review_semantic_duplicates_interactive(mock_input, semantic_analyzer, mock_element):
#     pass

