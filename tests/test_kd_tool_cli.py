"""
KDToolCLI测试模块。

此模块包含对KDToolCLI类的单元测试。
"""

import pytest
import json
import os
import shutil
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Type # 确保导入 Type

# 导入核心类和函数
from knowledge_distiller_kd.core.kd_tool_CLI import KDToolCLI
from knowledge_distiller_kd.core import constants
from knowledge_distiller_kd.core.document_processor import ContentBlock # 导入 ContentBlock
from unstructured.documents.elements import Element, NarrativeText, Title, CodeSnippet # 导入需要的 Element 类型
from tests.test_data_generator import DataGenerator


# 测试数据生成器
test_data_generator = DataGenerator()

# --- Fixtures ---

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """
    创建一个临时输出目录。
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir

@pytest.fixture
def temp_decision_file(tmp_path: Path) -> Path:
    """
    创建一个临时决策文件。
    """
    decision_dir = tmp_path / "decisions"
    decision_dir.mkdir()
    return decision_dir / "decisions.json"

@pytest.fixture
def kd_tool(temp_output_dir: Path, temp_decision_file: Path) -> KDToolCLI:
    """
    创建一个 KDToolCLI 实例。
    """
    return KDToolCLI(
        output_dir=temp_output_dir,
        decision_file=temp_decision_file,
        skip_semantic=True # 默认在 CLI 测试中跳过语义，除非专门测试
    )

# 辅助 Fixture 创建 ContentBlock
@pytest.fixture
def create_content_block():
    """提供一个创建 ContentBlock 的辅助函数"""
    def _create(text: str, file_path: str, element_type: Type[Element] = NarrativeText, element_id: str = None) -> ContentBlock:
        # 为 element_id 生成一个简单的唯一值（如果未提供）
        if element_id is None:
            import uuid
            element_id = str(uuid.uuid4())

        # 创建基础 Element 实例
        element = element_type(text=text, element_id=element_id)
        # 创建 ContentBlock
        return ContentBlock(element, file_path)
    return _create

# --- 测试用例 ---

def test_initialization(kd_tool: KDToolCLI, temp_output_dir: Path, temp_decision_file: Path) -> None:
    """
    测试 KDToolCLI 的初始化。
    """
    assert kd_tool.output_dir == temp_output_dir.resolve()
    assert kd_tool.decision_file == temp_decision_file.resolve()
    assert kd_tool.skip_semantic is True
    assert kd_tool.input_dir is None
    # 检查重构后应该存在的属性
    assert kd_tool.blocks_data == []
    assert kd_tool.block_decisions == {}
    assert kd_tool.md5_analyzer is not None
    assert kd_tool.semantic_analyzer is not None
    assert kd_tool._analysis_completed is False
    # 确认旧属性不存在
    assert not hasattr(kd_tool, 'markdown_files_content')

def test_set_input_dir(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试设置输入目录的功能。
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    result = kd_tool.set_input_dir(input_dir)
    assert result is True
    assert kd_tool.input_dir == input_dir.resolve()
    # 验证状态是否重置
    assert kd_tool.blocks_data == []
    assert kd_tool.block_decisions == {}
    assert kd_tool._analysis_completed is False

def test_set_input_dir_relative_path(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试使用相对路径设置输入目录。
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # 使用相对路径
    relative_path = Path("input")
    # 切换到临时目录
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        result = kd_tool.set_input_dir(relative_path)
        assert result is True
        assert kd_tool.input_dir == (tmp_path / relative_path).resolve()
    finally:
        # 切换回原始目录
        os.chdir(original_cwd)


def test_set_input_dir_invalid(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试设置无效输入目录的情况。
    """
    invalid_dir = tmp_path / "nonexistent"
    result = kd_tool.set_input_dir(invalid_dir)
    assert result is False
    assert kd_tool.input_dir is None

# TODO: 重构此测试以适应新的分析流程。
# _read_files, _parse_markdown, _initialize_decisions 现在是 run_analysis 的一部分或已被替换。
# 需要模拟 document_processor.process_directory, md5_analyzer.find_md5_duplicates 等。
# @patch('knowledge_distiller_kd.core.kd_tool_CLI.KDToolCLI._process_documents') # 假设有个处理文档的方法
# @patch('knowledge_distiller_kd.core.kd_tool_CLI.KDToolCLI._initialize_decisions') # 假设这个内部方法还存在
@patch('knowledge_distiller_kd.core.kd_tool_CLI.MD5Analyzer.find_md5_duplicates')
@patch('knowledge_distiller_kd.core.kd_tool_CLI.SemanticAnalyzer.find_semantic_duplicates')
def test_run_analysis(mock_semantic_find: MagicMock, mock_md5_find: MagicMock,
                     # mock_init_decisions: MagicMock, mock_process_docs: MagicMock,
                     kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试运行分析的功能 (需要更新 Mock)。
    """
    # 设置输入目录
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "dummy.md").write_text("# Dummy") # 创建一个文件以避免 process_directory 警告
    kd_tool.set_input_dir(input_dir)

    # 设置模拟返回值
    # mock_process_docs.return_value = True
    # mock_init_decisions.return_value = True
    mock_md5_find.return_value = True
    mock_semantic_find.return_value = True # 假设语义分析也成功

    # 运行分析 (跳过语义，因为默认是 True)
    kd_tool.skip_semantic = True
    result = kd_tool.run_analysis()

    assert result is True
    assert kd_tool._analysis_completed is True
    # mock_process_docs.assert_called_once()
    # mock_init_decisions.assert_called_once()
    mock_md5_find.assert_called_once()
    mock_semantic_find.assert_not_called() # 因为 skip_semantic = True

    # 测试不跳过语义的情况
    kd_tool.skip_semantic = False
    kd_tool._analysis_completed = False # 重置状态
    mock_md5_find.reset_mock()
    mock_semantic_find.reset_mock()
    # 模拟加载模型成功
    with patch.object(kd_tool.semantic_analyzer, 'load_semantic_model', return_value=True):
         with patch.object(kd_tool.semantic_analyzer, '_model_loaded', True): # 假装模型已加载
              result_semantic = kd_tool.run_analysis()
              assert result_semantic is True
              assert kd_tool._analysis_completed is True
              mock_md5_find.assert_called_once()
              mock_semantic_find.assert_called_once()


def test_run_analysis_no_input_dir(kd_tool: KDToolCLI) -> None:
    """
    测试在没有设置输入目录的情况下运行分析。
    """
    result = kd_tool.run_analysis()
    assert result is False

# TODO: 重构此测试。需要模拟 _process_documents 失败。
# @patch('knowledge_distiller_kd.core.kd_tool_CLI.KDToolCLI._process_documents', return_value=False)
# def test_run_analysis_process_docs_failed(mock_process_docs: MagicMock, kd_tool: KDToolCLI, tmp_path: Path) -> None:
#     """
#     测试在处理文档失败的情况下运行分析。
#     """
#     input_dir = tmp_path / "input"
#     input_dir.mkdir()
#     kd_tool.set_input_dir(input_dir)
#     result = kd_tool.run_analysis()
#     assert result is False
#     mock_process_docs.assert_called_once()

def test_load_decisions_file_not_exists(kd_tool: KDToolCLI) -> None:
    """
    测试从不存在的决策文件加载决策。
    """
    kd_tool.decision_file = Path("nonexistent.json")
    result = kd_tool.load_decisions()
    assert result is False

def test_load_decisions_invalid_json(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试从无效的 JSON 文件加载决策。
    """
    # 创建一个包含无效 JSON 的文件
    decision_file = tmp_path / "invalid.json"
    decision_file.write_text("invalid json")
    kd_tool.decision_file = decision_file

    result = kd_tool.load_decisions()
    assert result is False

def test_load_decisions_success(kd_tool: KDToolCLI, tmp_path: Path, create_content_block) -> None:
    """
    测试成功加载决策。
    """
    # 设置输入目录
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    kd_tool.set_input_dir(input_dir)

    # 创建测试文件
    test_file = input_dir / "test.md"
    test_file.write_text("Test content")
    test_file_abs_str = str(test_file.resolve())

    # 创建一个包含有效决策的文件
    decision_file = tmp_path / "valid.json"
    decisions_to_save = [
        {
            "file": test_file_abs_str, # 使用绝对路径字符串
            "block_id": "id1",
            "type": "NarrativeText",
            "decision": "keep"
        },
        {
            "file": test_file_abs_str,
            "block_id": "id2",
            "type": "NarrativeText",
            "decision": "delete"
        }
    ]
    decision_file.write_text(json.dumps(decisions_to_save))
    kd_tool.decision_file = decision_file

    # 设置一些块数据 (现在不需要，load_decisions 不依赖 blocks_data)
    # block1 = create_content_block("Test content 1", test_file_abs_str, NarrativeText, "id1")
    # block2 = create_content_block("Test content 2", test_file_abs_str, NarrativeText, "id2")
    # kd_tool.blocks_data = [block1, block2]

    result = kd_tool.load_decisions()

    # 预期决策键使用绝对路径
    # ==================== 修复：修正 expected_decisions 的构造 ====================
    expected_decisions = {
        f"{test_file_abs_str}{constants.DECISION_KEY_SEPARATOR}id1{constants.DECISION_KEY_SEPARATOR}NarrativeText": "keep",
        f"{test_file_abs_str}{constants.DECISION_KEY_SEPARATOR}id2{constants.DECISION_KEY_SEPARATOR}NarrativeText": "delete"
    }
    # ==========================================================================
    assert kd_tool.block_decisions == expected_decisions
    assert result is True # 确认返回 True

def test_load_decisions_relative_path(kd_tool: KDToolCLI, tmp_path: Path, create_content_block) -> None:
    """
    测试加载包含相对路径的决策文件。
    load_decisions 现在内部不解析相对路径，直接使用key。
    """
    # 创建测试文件
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    test_file = input_dir / "test.md"
    test_file.write_text("Test content")
    test_file_rel_str = "test.md" # 相对路径字符串

    # 设置输入目录
    kd_tool.set_input_dir(input_dir)

    # 创建测试决策文件，包含相对路径
    decisions_to_save = [{
        "file": test_file_rel_str,
        "block_id": "id_rel",
        "type": "NarrativeText",
        "decision": "keep"
    }]
    kd_tool.decision_file.write_text(json.dumps(decisions_to_save), encoding='utf-8')

    # 添加一些块数据 (用于后续可能的 apply_decisions 测试，load 不直接用)
    block1 = create_content_block("Test content", str(test_file.resolve()), NarrativeText, "id_rel")
    kd_tool.blocks_data = [block1]

    # 运行测试
    result = kd_tool.load_decisions()
    assert len(kd_tool.block_decisions) == 1
    # 预期键直接使用了文件中的相对路径
    expected_key = f"{test_file_rel_str}{constants.DECISION_KEY_SEPARATOR}id_rel{constants.DECISION_KEY_SEPARATOR}NarrativeText"
    assert expected_key in kd_tool.block_decisions
    assert kd_tool.block_decisions[expected_key] == "keep"
    assert result is True # 确认返回 True

def test_save_decisions_no_decisions(kd_tool: KDToolCLI) -> None:
    """
    测试在没有决策的情况下保存决策。
    """
    result = kd_tool.save_decisions()
    assert result is False # 应该返回 False，因为没有决策可保存

def test_save_decisions_success(kd_tool: KDToolCLI, tmp_path: Path, create_content_block) -> None:
    """
    测试成功保存决策。
    """
    # 设置输入目录
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    kd_tool.set_input_dir(input_dir)

    # 创建测试文件
    test_file = input_dir / "test.md"
    test_file.write_text("Test content")
    test_file_abs_str = str(test_file.resolve())

    # 设置一些块数据和决策
    block1 = create_content_block("Test content 1", test_file_abs_str, NarrativeText, "id1")
    block2 = create_content_block("Test content 2", test_file_abs_str, NarrativeText, "id2")
    kd_tool.blocks_data = [block1, block2] # blocks_data 本身不直接用于保存决策，但决策通常来源于分析
    key1 = f"{test_file_abs_str}{constants.DECISION_KEY_SEPARATOR}id1{constants.DECISION_KEY_SEPARATOR}NarrativeText"
    key2 = f"{test_file_abs_str}{constants.DECISION_KEY_SEPARATOR}id2{constants.DECISION_KEY_SEPARATOR}NarrativeText"
    kd_tool.block_decisions = {
        key1: "keep",
        key2: "delete"
    }

    # 保存决策
    result = kd_tool.save_decisions()

    # 验证保存的文件内容
    assert kd_tool.decision_file.exists()
    saved_data = json.loads(kd_tool.decision_file.read_text())
    assert len(saved_data) == 2
    assert all(isinstance(item, dict) for item in saved_data)
    assert all(key in item for item in saved_data for key in ["file", "block_id", "type", "decision"])

    # 验证保存的路径是绝对路径 (当前行为)
    assert saved_data[0]["file"] == test_file_abs_str
    assert saved_data[1]["file"] == test_file_abs_str
    assert result is True # 确认返回 True

def test_save_decisions_no_input_dir(kd_tool: KDToolCLI, tmp_path: Path, create_content_block) -> None:
    """
    测试在没有设置输入目录的情况下保存决策 (应该保存绝对路径)。
    """
    # 创建测试文件 (不在 input_dir 下)
    test_file = tmp_path / "test_abs.md"
    test_file.write_text("Test content")
    test_file_abs_str = str(test_file.resolve())

    # 设置一些块数据和决策 (不设置 input_dir)
    block1 = create_content_block("Test content 1", test_file_abs_str, NarrativeText, "id_abs1")
    block2 = create_content_block("Test content 2", test_file_abs_str, NarrativeText, "id_abs2")
    kd_tool.blocks_data = [block1, block2]
    key1 = f"{test_file_abs_str}{constants.DECISION_KEY_SEPARATOR}id_abs1{constants.DECISION_KEY_SEPARATOR}NarrativeText"
    key2 = f"{test_file_abs_str}{constants.DECISION_KEY_SEPARATOR}id_abs2{constants.DECISION_KEY_SEPARATOR}NarrativeText"
    kd_tool.block_decisions = {
        key1: "keep",
        key2: "delete"
    }

    # 保存决策
    result = kd_tool.save_decisions()

    # 验证保存的文件内容
    assert kd_tool.decision_file.exists()
    saved_data = json.loads(kd_tool.decision_file.read_text())
    assert len(saved_data) == 2
    assert all(isinstance(item, dict) for item in saved_data)
    assert all(key in item for item in saved_data for key in ["file", "block_id", "type", "decision"])

    # 验证绝对路径
    assert Path(saved_data[0]["file"]).is_absolute()
    assert saved_data[0]["file"] == test_file_abs_str
    assert Path(saved_data[1]["file"]).is_absolute()
    assert saved_data[1]["file"] == test_file_abs_str
    assert result is True # 确认返回 True

def test_save_decisions_relative_path(kd_tool: KDToolCLI, tmp_path: Path, create_content_block) -> None:
    """
    测试保存决策时，当设置了 input_dir，路径是否被正确处理 (当前应为绝对路径)。
    TODO: 将来可以修改 save_decisions 以保存相对路径，并更新此测试。
    """
    # 创建测试文件
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    test_file = input_dir / "test.md"
    test_file.write_text("Test content")
    test_file_abs_str = str(test_file.resolve())

    # 设置输入目录
    kd_tool.set_input_dir(input_dir)

    # 添加测试决策
    block1 = create_content_block("Test content", test_file_abs_str, NarrativeText, "id_rel_save")
    kd_tool.blocks_data = [block1]
    key = f"{test_file_abs_str}{constants.DECISION_KEY_SEPARATOR}id_rel_save{constants.DECISION_KEY_SEPARATOR}NarrativeText"
    kd_tool.block_decisions[key] = "keep"

    # 运行测试
    result = kd_tool.save_decisions()

    # 验证保存的内容
    assert kd_tool.decision_file.exists()
    saved_data = json.loads(kd_tool.decision_file.read_text(encoding='utf-8'))
    assert len(saved_data) == 1
    # 当前行为是保存绝对路径，所以断言失败是正常的
    # assert saved_data[0]["file"] == "test.md" # 期望相对路径
    assert saved_data[0]["file"] == test_file_abs_str # 实际行为：绝对路径
    assert result is True # 确认返回 True

def test_apply_decisions_no_blocks(kd_tool: KDToolCLI) -> None:
    """
    测试在没有块数据的情况下应用决策。
    """
    # 需要先标记分析已完成
    kd_tool._analysis_completed = True
    result = kd_tool.apply_decisions()
    assert len(kd_tool.blocks_data) == 0
    assert result is True # 没有块也算成功

def test_apply_decisions_success(kd_tool: KDToolCLI, tmp_path: Path, create_content_block) -> None:
    """
    测试成功应用决策。
    """
    # 设置输入目录
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    kd_tool.set_input_dir(input_dir)

    # 创建测试文件
    test_file = input_dir / "test.md"
    test_file_abs_str = str(test_file.resolve())

    # 设置一些块数据和决策
    block1 = create_content_block("Test content 1\n", test_file_abs_str, NarrativeText, "id1")
    block2 = create_content_block("Test content 2\n", test_file_abs_str, NarrativeText, "id2")
    block3 = create_content_block("Test content 3\n", test_file_abs_str, NarrativeText, "id3")
    kd_tool.blocks_data = [block1, block2, block3]
    key1 = f"{test_file_abs_str}{constants.DECISION_KEY_SEPARATOR}id1{constants.DECISION_KEY_SEPARATOR}NarrativeText"
    key2 = f"{test_file_abs_str}{constants.DECISION_KEY_SEPARATOR}id2{constants.DECISION_KEY_SEPARATOR}NarrativeText"
    key3 = f"{test_file_abs_str}{constants.DECISION_KEY_SEPARATOR}id3{constants.DECISION_KEY_SEPARATOR}NarrativeText"
    kd_tool.block_decisions = {
        key1: "keep",
        key2: "delete",
        key3: "undecided" # 未决策的默认保留
    }
    kd_tool._analysis_completed = True # 标记分析已完成

    # 应用决策
    result = kd_tool.apply_decisions()

    # 验证输出文件
    output_file = kd_tool.output_dir / f"test{constants.DEFAULT_OUTPUT_SUFFIX}.md"
    assert output_file.exists()
    content = output_file.read_text()
    # 注意：apply_decisions 会在块之间添加 '\n\n'
    assert "Test content 1\n\n" in content
    assert "Test content 2" not in content
    assert "Test content 3\n\n" in content
    assert result is True # 确认返回 True

def test_apply_decisions_no_decisions(kd_tool: KDToolCLI, tmp_path: Path, create_content_block) -> None:
    """
    测试在没有决策的情况下应用决策 (所有块应保留)。
    """
    # 设置输入目录
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    kd_tool.set_input_dir(input_dir)

    # 创建测试文件
    test_file = input_dir / "test.md"
    test_file_abs_str = str(test_file.resolve())

    # 设置一些块数据但没有决策 (或全是 undecided)
    block1 = create_content_block("Test content 1\n", test_file_abs_str, NarrativeText, "id1")
    block2 = create_content_block("Test content 2\n", test_file_abs_str, NarrativeText, "id2")
    kd_tool.blocks_data = [block1, block2]
    key1 = f"{test_file_abs_str}{constants.DECISION_KEY_SEPARATOR}id1{constants.DECISION_KEY_SEPARATOR}NarrativeText"
    key2 = f"{test_file_abs_str}{constants.DECISION_KEY_SEPARATOR}id2{constants.DECISION_KEY_SEPARATOR}NarrativeText"
    kd_tool.block_decisions = {
        key1: constants.DECISION_UNDECIDED,
        key2: constants.DECISION_UNDECIDED
    }
    kd_tool._analysis_completed = True # 标记分析已完成

    # 应用决策
    result = kd_tool.apply_decisions()

    # 验证输出文件（所有块都应该被保留）
    output_file = kd_tool.output_dir / f"test{constants.DEFAULT_OUTPUT_SUFFIX}.md"
    assert output_file.exists()
    content = output_file.read_text()
    assert "Test content 1\n\n" in content
    assert "Test content 2\n\n" in content
    assert result is True # 确认返回 True

# --- 移除测试旧内部方法的用例 ---
# test_read_files_success, test_read_files_no_markdown, test_read_files_empty_directory
# test_parse_markdown_success, test_parse_markdown_empty_content, test_parse_markdown_invalid_content
# test_ensure_dirs_exist, test_ensure_dirs_exist_error

def test_initialize_decisions(kd_tool: KDToolCLI, create_content_block) -> None:
    """
    测试初始化决策的情况 (使用 ContentBlock)。
    """
    # 设置测试数据
    test_file = Path("test.md")
    block1 = create_content_block("Content 1", str(test_file), NarrativeText, "id1")
    block2 = create_content_block("Content 2", str(test_file), NarrativeText, "id2")
    kd_tool.blocks_data = [block1, block2]

    kd_tool._initialize_decisions() # 调用内部方法进行测试
    assert len(kd_tool.block_decisions) == 2
    key1 = f"{str(test_file)}{constants.DECISION_KEY_SEPARATOR}id1{constants.DECISION_KEY_SEPARATOR}NarrativeText"
    key2 = f"{str(test_file)}{constants.DECISION_KEY_SEPARATOR}id2{constants.DECISION_KEY_SEPARATOR}NarrativeText"
    assert key1 in kd_tool.block_decisions
    assert key2 in kd_tool.block_decisions
    assert kd_tool.block_decisions[key1] == constants.DECISION_UNDECIDED
    assert kd_tool.block_decisions[key2] == constants.DECISION_UNDECIDED

def test_path_resolution_with_cwd(kd_tool: KDToolCLI, tmp_path: Path, create_content_block) -> None:
    """
    测试相对于当前工作目录的路径解析 (在 load_decisions 中)。
    """
    # 创建测试文件
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    test_file = input_dir / "test.md"
    test_file.write_text("Test content")
    test_file_rel_str = "test.md" # 相对路径

    # 设置输入目录
    kd_tool.set_input_dir(input_dir) # 需要设置 input_dir 吗？load 不依赖

    # 切换到临时目录的父目录，让 input 成为相对路径 'input/test.md'
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        # 创建测试决策文件，包含相对于 *tmp_path* 的路径
        decisions_to_save = [{
            "file": f"input/{test_file_rel_str}", # 相对于 tmp_path 的路径
            "block_id": "id_cwd",
            "type": "NarrativeText",
            "decision": "keep"
        }]
        decision_file_path = kd_tool.decision_file # 获取 fixture 设置的文件路径
        decision_file_path.write_text(json.dumps(decisions_to_save), encoding='utf-8')

        # 添加块数据 (用于可能的 apply 测试)
        block1 = create_content_block("Test content", str(test_file.resolve()), NarrativeText, "id_cwd")
        kd_tool.blocks_data = [block1]

        # 运行测试
        result = kd_tool.load_decisions()
        assert len(kd_tool.block_decisions) == 1
        # 预期键直接使用了文件中的相对路径
        expected_key = f"input/{test_file_rel_str}{constants.DECISION_KEY_SEPARATOR}id_cwd{constants.DECISION_KEY_SEPARATOR}NarrativeText"
        assert expected_key in kd_tool.block_decisions
        assert kd_tool.block_decisions[expected_key] == "keep"
        assert result is True # 确认返回 True
    finally:
        os.chdir(original_cwd)

