# tests/ui/test_cli_interface.py
"""
Unit tests for the CliInterface class.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call, mock_open, DEFAULT
import sys
from io import StringIO
import subprocess

# Modules to test
from knowledge_distiller_kd.ui.cli_interface import CliInterface
from knowledge_distiller_kd.core.engine import KnowledgeDistillerEngine
from knowledge_distiller_kd.core import constants
from knowledge_distiller_kd.core.error_handler import ConfigurationError, FileOperationError, AnalysisError
from knowledge_distiller_kd.core.utils import create_decision_key
from knowledge_distiller_kd.processing.document_processor import ContentBlock
from unstructured.documents.elements import NarrativeText, Title, CodeSnippet
from knowledge_distiller_kd.core import utils as core_utils

# --- Fixtures ---

@pytest.fixture
def mock_engine() -> MagicMock:
    """Creates a mock KnowledgeDistillerEngine object."""
    engine = MagicMock(spec=KnowledgeDistillerEngine)
    engine.get_status_summary.return_value = {
        "input_dir": "/fake/input", "decision_file": "/fake/decisions.json", "output_dir": "/fake/output",
        "skip_semantic": False, "similarity_threshold": 0.8, "analysis_completed": False,
        "decisions_loaded": False, "total_blocks": 0, "md5_duplicates_groups": 0,
        "semantic_duplicates_pairs": 0, "decided_blocks": 0,
    }
    engine.set_input_dir.return_value = True
    engine.run_analysis.return_value = True
    engine.get_md5_duplicates.return_value = []
    engine.get_semantic_duplicates.return_value = []
    engine.update_decision.return_value = True
    # Initialize block_decisions as a dict for the mock
    engine.block_decisions = {}
    return engine

@pytest.fixture
def cli_instance(mock_engine: MagicMock) -> CliInterface:
    """Creates a CliInterface instance with a mocked engine."""
    return CliInterface(engine=mock_engine)

# Helper to create ContentBlock for tests
@pytest.fixture
def create_content_block():
    """Factory fixture to create ContentBlock instances for testing."""
    def _create(text: str, file_path: str, block_id: str, block_type: str = "NarrativeText") -> ContentBlock:
        element_map = {"NarrativeText": NarrativeText, "Title": Title, "CodeSnippet": CodeSnippet}
        element_cls = element_map.get(block_type, NarrativeText)
        element = element_cls(text=text, element_id=block_id)
        # Ensure file_path is absolute for consistent key generation in tests
        abs_file_path = str(Path(file_path).resolve())
        cb = ContentBlock(element, abs_file_path)
        cb.element.__class__ = element_cls
        cb.original_text = text
        return cb
    return _create

# --- Test Cases ---

@patch('builtins.input', side_effect=['q'])
@patch('builtins.print')
def test_run_quit_immediately(mock_print: MagicMock, mock_input: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock):
    cli_instance.run()
    mock_input.assert_called_once_with("请输入选项: ")
    mock_print.assert_any_call("正在退出...")
    mock_print.assert_any_call("感谢使用 KD Tool！")
    mock_engine.get_status_summary.assert_called_once()

@patch('builtins.print')
def test_display_main_menu_basic(mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock):
    status = mock_engine.get_status_summary.return_value
    cli_instance._display_main_menu()
    mock_print.assert_any_call("\n--- KD Tool 主菜单 ---")
    mock_print.assert_any_call(f"当前输入目录: {status['input_dir']}")
    mock_print.assert_any_call("1. 设置输入目录")
    mock_print.assert_any_call("q. 退出")

@patch('builtins.print')
def test_display_main_menu_analysis_complete(mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock):
    status = {
        "input_dir": "/test/dir", "decision_file": "dec.json", "output_dir": "out",
        "skip_semantic": True, "similarity_threshold": 0.7, "analysis_completed": True,
        "decisions_loaded": True, "total_blocks": 150, "md5_duplicates_groups": 5,
        "semantic_duplicates_pairs": 0, "decided_blocks": 20,
    }
    mock_engine.get_status_summary.return_value = status
    cli_instance._display_main_menu()
    mock_print.assert_any_call("\n--- KD Tool 主菜单 ---")
    mock_print.assert_any_call("分析状态: 已完成")
    mock_print.assert_any_call("2. 运行分析")
    mock_print.assert_any_call("3. 查看/处理重复项")
    mock_print.assert_any_call("6. 应用决策 (生成去重文件)")

# --- Test Handlers ---
@patch('builtins.print')
@patch('builtins.input', side_effect=['1', '/valid/path', 'q'])
def test_run_handle_set_input_dir_success(mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock, tmp_path: Path):
    input_path_str = "/valid/path"; resolved_path = Path(input_path_str).resolve()
    mock_engine.set_input_dir.return_value = True
    initial_status = mock_engine.get_status_summary.return_value
    updated_status = initial_status.copy(); updated_status["input_dir"] = str(resolved_path)
    mock_engine.get_status_summary.side_effect = [initial_status, updated_status, updated_status]
    cli_instance.run()
    assert mock_input.call_count == 3
    mock_input.assert_has_calls([call("请输入选项: "), call("请输入输入目录的路径: "), call("请输入选项: ")])
    mock_engine.set_input_dir.assert_called_once_with(Path(input_path_str))
    mock_print.assert_any_call(f"[*] 输入目录已成功设置为: {resolved_path}")

@patch('builtins.print')
@patch('builtins.input', side_effect=['1', '/invalid/path', 'q'])
def test_run_handle_set_input_dir_fail(mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock):
    input_path_str = "/invalid/path"; mock_engine.set_input_dir.return_value = False
    cli_instance.run()
    assert mock_input.call_count == 3
    mock_input.assert_has_calls([call("请输入选项: "), call("请输入输入目录的路径: "), call("请输入选项: ")])
    mock_engine.set_input_dir.assert_called_once_with(Path(input_path_str))
    mock_print.assert_any_call("[错误] 设置输入目录失败。请检查日志或路径是否有效且为目录。")

@patch('builtins.print')
@patch('builtins.input', side_effect=['2', 'q'])
def test_run_handle_run_analysis_success(mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock):
    mock_engine.get_status_summary.return_value['input_dir'] = "/some/path" # Make option visible
    mock_engine.run_analysis.return_value = True
    cli_instance.run()
    assert mock_input.call_count == 2
    mock_input.assert_has_calls([call("请输入选项: "), call("请输入选项: ")])
    mock_engine.run_analysis.assert_called_once()
    mock_print.assert_any_call("\n[*] 正在运行分析...")
    mock_print.assert_any_call("[*] 分析完成。")

@patch('builtins.print')
@patch('builtins.input', side_effect=['2', 'q'])
def test_run_handle_run_analysis_fail(mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock):
    mock_engine.get_status_summary.return_value['input_dir'] = "/some/path"
    mock_engine.run_analysis.return_value = False
    cli_instance.run()
    assert mock_input.call_count == 2
    mock_input.assert_has_calls([call("请输入选项: "), call("请输入选项: ")])
    mock_engine.run_analysis.assert_called_once()
    mock_print.assert_any_call("\n[*] 正在运行分析...")
    mock_print.assert_any_call("[错误] 分析过程中发生错误。请检查日志获取详细信息。")

# --- Test Review Handlers ---
@patch('builtins.print')
@patch('builtins.input', side_effect=['m', 'q'])
@patch('knowledge_distiller_kd.ui.cli_interface.CliInterface.review_md5_duplicates')
def test_handle_view_process_duplicates_selects_md5(mock_review_md5: MagicMock, mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface):
    cli_instance._handle_view_process_duplicates()
    mock_print.assert_any_call("\n--- 查看/处理重复项 ---")
    mock_print.assert_any_call("m. 处理 MD5 重复项")
    mock_input.assert_any_call("请选择处理类型 (m/s, q 返回): ")
    mock_review_md5.assert_called_once()

@patch('builtins.print')
@patch('builtins.input', side_effect=['s', 'q'])
@patch('knowledge_distiller_kd.ui.cli_interface.CliInterface.review_semantic_duplicates')
def test_handle_view_process_duplicates_selects_semantic(mock_review_semantic: MagicMock, mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface):
    cli_instance._handle_view_process_duplicates()
    mock_print.assert_any_call("s. 处理语义相似项")
    mock_input.assert_any_call("请选择处理类型 (m/s, q 返回): ")
    mock_review_semantic.assert_called_once()

@patch('builtins.print')
@patch('builtins.input', side_effect=['q'])
def test_handle_view_process_duplicates_quit(mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface):
    with patch.object(cli_instance, 'review_md5_duplicates') as md5_rev, \
         patch.object(cli_instance, 'review_semantic_duplicates') as sem_rev:
        cli_instance._handle_view_process_duplicates()
        mock_input.assert_called_once_with("请选择处理类型 (m/s, q 返回): ")
        md5_rev.assert_not_called()
        sem_rev.assert_not_called()

# --- Test MD5 Review Logic ---
@patch('builtins.print')
@patch('builtins.input', side_effect=['q'])
@patch('knowledge_distiller_kd.core.utils.display_block_preview')
def test_review_md5_duplicates_display_and_quit(mock_display_preview: MagicMock, mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock, create_content_block):
    block1 = create_content_block("Content A", "/path/file1.md", "id1")
    block2 = create_content_block("Content A", "/path/file2.md", "id2")
    mock_duplicates = [[block1, block2]]
    mock_engine.get_md5_duplicates.return_value = mock_duplicates
    key1 = create_decision_key(str(Path(block1.file_path).resolve()), block1.block_id, "NarrativeText")
    key2 = create_decision_key(str(Path(block2.file_path).resolve()), block2.block_id, "NarrativeText")
    mock_engine.block_decisions = {key1: constants.DECISION_UNDECIDED, key2: constants.DECISION_UNDECIDED}
    cli_instance.review_md5_duplicates()
    mock_engine.get_md5_duplicates.assert_called_once()
    mock_print.assert_any_call("\n--- MD5 重复项审查 (共 1 组) ---")
    mock_print.assert_any_call(f"组 1 / 1 (共 2 项):")
    mock_print.assert_any_call(f"  1. [ ] {Path(block1.file_path).resolve()} # {block1.block_id} ({block1.block_type})")
    mock_print.assert_any_call(f"      预览: Content A")
    mock_print.assert_any_call(f"  2. [ ] {Path(block2.file_path).resolve()} # {block2.block_id} ({block2.block_type})")
    mock_print.assert_any_call(f"      预览: Content A")
    mock_input.assert_called_once_with("操作 (k[索引] 保留, d[索引] 删除, a 全删, n 下一组, p 上一组, q 退出): ")
    mock_display_preview.assert_not_called()
    mock_engine.update_decision.assert_not_called()

@patch('builtins.print')
@patch('builtins.input', side_effect=['k1', 'q'])
@patch('knowledge_distiller_kd.core.utils.display_block_preview')
def test_review_md5_keep_action(mock_display_preview: MagicMock, mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock, create_content_block):
    block1 = create_content_block("Content A", "/path/file1.md", "id1")
    block2 = create_content_block("Content A", "/path/file2.md", "id2")
    mock_duplicates = [[block1, block2]]
    mock_engine.get_md5_duplicates.return_value = mock_duplicates
    key1 = create_decision_key(str(Path(block1.file_path).resolve()), block1.block_id, "NarrativeText")
    mock_engine.block_decisions = {key1: constants.DECISION_UNDECIDED}
    cli_instance.review_md5_duplicates()
    mock_engine.update_decision.assert_called_once_with(key1, constants.DECISION_KEEP)
    mock_print.assert_any_call("[*] 已将 1 标记为 [KEEP]")

@patch('builtins.print')
@patch('builtins.input', side_effect=['d2', 'q'])
@patch('knowledge_distiller_kd.core.utils.display_block_preview')
def test_review_md5_delete_action(mock_display_preview: MagicMock, mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock, create_content_block):
    block1 = create_content_block("Content A", "/path/file1.md", "id1")
    block2 = create_content_block("Content A", "/path/file2.md", "id2")
    mock_duplicates = [[block1, block2]]
    mock_engine.get_md5_duplicates.return_value = mock_duplicates
    key2 = create_decision_key(str(Path(block2.file_path).resolve()), block2.block_id, "NarrativeText")
    mock_engine.block_decisions = {key2: constants.DECISION_UNDECIDED}
    cli_instance.review_md5_duplicates()
    mock_engine.update_decision.assert_called_once_with(key2, constants.DECISION_DELETE)
    mock_print.assert_any_call("[*] 已将 2 标记为 [DELETE]")

@patch('builtins.print')
@patch('builtins.input', side_effect=['a', 'q'])
@patch('knowledge_distiller_kd.core.utils.display_block_preview')
def test_review_md5_delete_all_action(mock_display_preview: MagicMock, mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock, create_content_block):
    block1 = create_content_block("Content A", "/path/file1.md", "id1")
    block2 = create_content_block("Content A", "/path/file2.md", "id2")
    block3 = create_content_block("Content A", "/path/file3.md", "id3")
    mock_duplicates = [[block1, block2, block3]]
    mock_engine.get_md5_duplicates.return_value = mock_duplicates
    key1 = create_decision_key(str(Path(block1.file_path).resolve()), block1.block_id, "NarrativeText")
    key2 = create_decision_key(str(Path(block2.file_path).resolve()), block2.block_id, "NarrativeText")
    key3 = create_decision_key(str(Path(block3.file_path).resolve()), block3.block_id, "NarrativeText")
    mock_engine.block_decisions = {key1: constants.DECISION_UNDECIDED, key2: constants.DECISION_UNDECIDED, key3: constants.DECISION_UNDECIDED}
    cli_instance.review_md5_duplicates()
    expected_calls = [call(key2, constants.DECISION_DELETE), call(key3, constants.DECISION_DELETE)]
    mock_engine.update_decision.assert_has_calls(expected_calls, any_order=True)
    assert mock_engine.update_decision.call_count == 2
    mock_print.assert_any_call("[*] 操作完成: 2 个已标记, 0 个失败。")

@patch('builtins.print')
@patch('builtins.input', side_effect=['k99', 'q'])
@patch('knowledge_distiller_kd.core.utils.display_block_preview')
def test_review_md5_invalid_index(mock_display_preview: MagicMock, mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock, create_content_block):
    block1 = create_content_block("Content A", "/path/file1.md", "id1")
    block2 = create_content_block("Content A", "/path/file2.md", "id2")
    mock_duplicates = [[block1, block2]]
    mock_engine.get_md5_duplicates.return_value = mock_duplicates
    cli_instance.review_md5_duplicates()
    mock_print.assert_any_call("[错误] 无效的索引: 99 (应在 1 到 2 之间)")
    mock_engine.update_decision.assert_not_called()

@patch('builtins.print')
@patch('builtins.input', side_effect=['keep 1', 'q'])
@patch('knowledge_distiller_kd.core.utils.display_block_preview')
def test_review_md5_invalid_format(mock_display_preview: MagicMock, mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock, create_content_block):
    block1 = create_content_block("Content A", "/path/file1.md", "id1")
    mock_duplicates = [[block1]]
    mock_engine.get_md5_duplicates.return_value = mock_duplicates
    cli_instance.review_md5_duplicates()
    mock_print.assert_any_call("无效操作格式: 'keep 1'. 使用 k[索引] 或 d[索引].")
    mock_engine.update_decision.assert_not_called()

# --- Test Semantic Review Logic ---
@patch('builtins.print')
def test_review_semantic_duplicates_no_pairs(mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock):
    mock_engine.get_semantic_duplicates.return_value = []
    cli_instance.review_semantic_duplicates()
    mock_engine.get_semantic_duplicates.assert_called_once()
    mock_print.assert_any_call("\n[*] 未找到语义相似对。")

@patch('builtins.print')
@patch('builtins.input', side_effect=['q'])
@patch('knowledge_distiller_kd.core.utils.display_block_preview')
def test_review_semantic_duplicates_display_and_quit(mock_display_preview: MagicMock, mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock, create_content_block):
    block1 = create_content_block("This is the first sentence.", "/path/fileA.md", "ida1")
    block2 = create_content_block("This is a very similar first sentence.", "/path/fileB.md", "idb1")
    similarity = 0.95
    mock_pairs = [(block1, block2, similarity)]
    mock_engine.get_semantic_duplicates.return_value = mock_pairs
    key1 = create_decision_key(str(Path(block1.file_path).resolve()), block1.block_id, "NarrativeText")
    key2 = create_decision_key(str(Path(block2.file_path).resolve()), block2.block_id, "NarrativeText")
    mock_engine.block_decisions = {key1: constants.DECISION_UNDECIDED, key2: constants.DECISION_UNDECIDED}
    cli_instance.review_semantic_duplicates()
    mock_engine.get_semantic_duplicates.assert_called_once()
    mock_print.assert_any_call("\n--- 语义相似项审查 (共 1 对) ---")
    mock_print.assert_any_call(f"对 1 / 1 (相似度: {similarity:.4f}):")
    mock_print.assert_any_call(f"  1. [ ] {Path(block1.file_path).resolve()} # {block1.block_id} ({block1.block_type})")
    mock_print.assert_any_call(f"      预览: This is the first sentence.")
    mock_print.assert_any_call(f"  2. [ ] {Path(block2.file_path).resolve()} # {block2.block_id} ({block2.block_type})")
    mock_print.assert_any_call(f"      预览: This is a very similar first sentence.")
    mock_input.assert_called_once_with("操作 (k1/k2 保留, d1/d2 删除, skip 跳过, n 下一对, p 上一对, q 退出): ")
    mock_display_preview.assert_not_called()
    mock_engine.update_decision.assert_not_called()

# ==================== Add Semantic Action Tests ====================
@patch('builtins.print')
@patch('builtins.input', side_effect=['k1', 'q'])
def test_review_semantic_keep1_action(mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock, create_content_block):
    """Test the 'k1' action to keep the first block."""
    block1 = create_content_block("Sentence one.", "fA.md", "idA1")
    block2 = create_content_block("Sentence two.", "fB.md", "idB1")
    mock_pairs = [(block1, block2, 0.85)]
    mock_engine.get_semantic_duplicates.return_value = mock_pairs
    key1 = create_decision_key(block1.file_path, block1.block_id, "NarrativeText")
    mock_engine.block_decisions = {key1: constants.DECISION_UNDECIDED}
    cli_instance.review_semantic_duplicates()
    mock_engine.update_decision.assert_called_once_with(key1, constants.DECISION_KEEP)
    mock_print.assert_any_call("[*] 已将块 1 标记为 [KEEP]")

@patch('builtins.print')
@patch('builtins.input', side_effect=['d2', 'q'])
def test_review_semantic_delete2_action(mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock, create_content_block):
    """Test the 'd2' action to delete the second block."""
    block1 = create_content_block("Sentence one.", "fA.md", "idA1")
    block2 = create_content_block("Sentence two.", "fB.md", "idB1")
    mock_pairs = [(block1, block2, 0.85)]
    mock_engine.get_semantic_duplicates.return_value = mock_pairs
    key2 = create_decision_key(block2.file_path, block2.block_id, "NarrativeText")
    mock_engine.block_decisions = {key2: constants.DECISION_UNDECIDED}
    cli_instance.review_semantic_duplicates()
    mock_engine.update_decision.assert_called_once_with(key2, constants.DECISION_DELETE)
    mock_print.assert_any_call("[*] 已将块 2 标记为 [DELETE]")

@patch('builtins.print')
@patch('builtins.input', side_effect=['k3', 'q']) # Invalid index
def test_review_semantic_invalid_index(mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock, create_content_block):
    """Test entering an invalid index like k3."""
    block1 = create_content_block("Sentence one.", "fA.md", "idA1")
    block2 = create_content_block("Sentence two.", "fB.md", "idB1")
    mock_pairs = [(block1, block2, 0.85)]
    mock_engine.get_semantic_duplicates.return_value = mock_pairs
    cli_instance.review_semantic_duplicates()
    mock_print.assert_any_call("无效操作格式: 'k3'. 使用 k1, k2, d1, d2.")
    mock_engine.update_decision.assert_not_called()

@patch('builtins.print')
@patch('builtins.input', side_effect=['keep 1', 'q']) # Invalid format
def test_review_semantic_invalid_format(mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock, create_content_block):
    """Test entering an invalid command format."""
    block1 = create_content_block("Sentence one.", "fA.md", "idA1")
    block2 = create_content_block("Sentence two.", "fB.md", "idB1")
    mock_pairs = [(block1, block2, 0.85)]
    mock_engine.get_semantic_duplicates.return_value = mock_pairs
    cli_instance.review_semantic_duplicates()
    mock_print.assert_any_call("无效操作格式: 'keep 1'. 使用 k1, k2, d1, d2.")
    mock_engine.update_decision.assert_not_called()
# =================================================================

# ==================== Add Final Handler Tests ====================
@patch('builtins.print')
@patch('builtins.input', side_effect=['4', 'q'])
def test_run_handle_load_decisions(mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock):
    """Test selecting '4' to load decisions."""
    mock_engine.get_status_summary.return_value['analysis_completed'] = True
    mock_engine.load_decisions.return_value = True
    cli_instance.run()
    mock_engine.load_decisions.assert_called_once()
    mock_print.assert_any_call("\n[*] 正在加载决策文件...") # Check UI message

@patch('builtins.print')
@patch('builtins.input', side_effect=['5', 'q'])
def test_run_handle_save_decisions(mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock):
    """Test selecting '5' to save decisions."""
    mock_engine.get_status_summary.return_value['analysis_completed'] = True
    mock_engine.save_decisions.return_value = True
    cli_instance.run()
    mock_engine.save_decisions.assert_called_once()
    mock_print.assert_any_call("\n[*] 正在保存当前决策...") # Check UI message

@patch('builtins.print')
@patch('builtins.input', side_effect=['6', 'q'])
def test_run_handle_apply_decisions(mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface, mock_engine: MagicMock):
    """Test selecting '6' to apply decisions."""
    mock_engine.get_status_summary.return_value['analysis_completed'] = True
    mock_engine.apply_decisions.return_value = True
    cli_instance.run()
    mock_engine.apply_decisions.assert_called_once()
    mock_print.assert_any_call("\n[*] 正在应用决策生成输出文件...") # Check UI message

@patch('builtins.print')
@patch('builtins.input', side_effect=['c', 'q', 'q']) # Select config, then quit config menu, then quit main menu
@patch('knowledge_distiller_kd.ui.cli_interface.CliInterface._handle_config') # Mock the handler
def test_run_handle_config_menu(mock_handle_config: MagicMock, mock_input: MagicMock, mock_print: MagicMock, cli_instance: CliInterface):
    """Test selecting 'c' to enter the config menu."""
    cli_instance.run()
    # Check that the config handler was called
    mock_handle_config.assert_called_once()
# ==================================================================

@patch('sys.exit')
@patch('sys.stdout', new_callable=StringIO)
def test_cli_pre_filter_only(mock_stdout, mock_exit, mock_engine: MagicMock):
    """Test --pre-filter flag works correctly."""
    # 模拟命令行参数
    test_args = ['cli.py', '-i', '/test/dir', '--pre-filter']
    
    # 在每次测试前重置mock以避免之前的调用影响当前测试
    mock_exit.reset_mock()
    mock_stdout.truncate(0)
    mock_stdout.seek(0)
    
    with patch('sys.argv', test_args):
        with patch('knowledge_distiller_kd.cli.parse_args') as mock_parse:
            # 创建模拟的参数对象
            mock_args = MagicMock()
            mock_args.input_dir = '/test/dir'
            mock_args.pre_filter = True
            mock_args.skip_prefilter = False
            mock_args.skip_semantic = False
            mock_args.threshold = 0.8
            mock_args.log_level = 'INFO'
            mock_args.output_dir = '/test/output'
            mock_args.decision_file = '/test/decisions.json'
            mock_parse.return_value = mock_args
            
            # 直接替换cli.py中导入的KnowledgeDistillerEngine类
            with patch('knowledge_distiller_kd.cli.KnowledgeDistillerEngine') as mock_engine_class:
                # 创建我们的模拟引擎实例
                engine_instance = MagicMock()
                engine_instance.input_dir = Path('/test/dir')
                # 确保run_prefilter_only被调用时不会设置run_prefilter_only_called属性
                engine_instance.run_prefilter_only.side_effect = lambda: (4, 
                                                       [Path('/test/file1.md'), Path('/test/file3.md')],
                                                       [[Path('/test/file2.md'), Path('/test/file2_dup.md')]])
                
                # 设置其他必要的mock
                engine_instance.set_input_dir.return_value = True
                
                # 设置模拟类返回我们的模拟实例
                mock_engine_class.return_value = engine_instance
                
                # 直接替换CliInterface类以避免run()被调用
                with patch('knowledge_distiller_kd.cli.CliInterface') as mock_cli_class:
                    cli_instance = MagicMock()
                    mock_cli_class.return_value = cli_instance
                    
                    # 模拟 ORMStorage
                    with patch('knowledge_distiller_kd.cli.ORMStorage') as mock_storage:
                        mock_storage_instance = MagicMock()
                        mock_storage.return_value = mock_storage_instance
                        
                        # 运行CLI主函数
                        from knowledge_distiller_kd.cli import main
                        main()
                        
                        # 验证参数解析和预过滤是否被调用
                        mock_parse.assert_called_once()
                        engine_instance.run_prefilter_only.assert_called_once()
                        
                        # 验证输出信息和退出代码
                        output = mock_stdout.getvalue()
                        assert "[Prefilter] Scanned 4 files, filtered 1 duplicates → 2 files remain." in output
                        
                        # 验证sys.exit被调用了，且最后一次调用是以0作为参数
                        assert mock_exit.called, "sys.exit应被调用"
                        args, _ = mock_exit.call_args_list[-1]  # 获取最后一次调用的参数
                        assert args[0] == 0, f"系统应以退出码0退出，实际为{args[0]}"

@patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine.run_analysis')
def test_cli_skip_prefilter(mock_run_analysis, mock_engine: MagicMock):
    """Test --skip-prefilter flag skips prefilter step."""
    # 模拟命令行参数
    test_args = ['cli.py', '-i', '/test/dir', '--skip-prefilter']
    
    with patch('sys.argv', test_args):
        with patch('knowledge_distiller_kd.cli.parse_args') as mock_parse:
            # 创建模拟的参数对象
            mock_args = MagicMock()
            mock_args.input_dir = '/test/dir'
            mock_args.pre_filter = False
            mock_args.skip_prefilter = True
            mock_args.skip_semantic = False
            mock_args.threshold = 0.8
            mock_args.log_level = 'INFO'
            mock_args.output_dir = '/test/output'
            mock_args.decision_file = '/test/decisions.json'
            mock_parse.return_value = mock_args
            
            # 直接替换cli.py中导入的KnowledgeDistillerEngine类
            with patch('knowledge_distiller_kd.cli.KnowledgeDistillerEngine') as mock_engine_class:
                # 创建我们的模拟引擎实例并明确设置run_prefilter_only方法
                engine_instance = MagicMock()
                engine_instance.input_dir = Path('/test/dir')
                engine_instance.skip_prefilter = True
                engine_instance.run_prefilter_only = MagicMock()  # 显式创建mock以跟踪调用
                mock_run_analysis.return_value = True
                
                # 设置模拟类返回我们的模拟实例
                mock_engine_class.return_value = engine_instance
                
                # 直接替换 cli.py 中的 ORMStorage 实例化
                with patch('knowledge_distiller_kd.cli.ORMStorage') as mock_storage:
                    mock_storage_instance = MagicMock()
                    mock_storage.return_value = mock_storage_instance
                    
                    # 模拟 CliInterface
                    with patch('knowledge_distiller_kd.cli.CliInterface') as mock_cli_class:
                        cli_instance = MagicMock()
                        mock_cli_class.return_value = cli_instance
                        
                        # 拦截sys.exit
                        with patch('sys.exit') as mock_exit:
                            # 运行CLI主函数
                            from knowledge_distiller_kd.cli import main
                            main()
                            
                            # 验证引擎初始化参数包含skip_prefilter=True
                            mock_engine_class.assert_called_once()
                            _, kwargs = mock_engine_class.call_args
                            assert kwargs.get('skip_prefilter', False) is True, "skip_prefilter参数未正确传递"
                            
                            # 确认run_prefilter_only不会被调用
                            engine_instance.run_prefilter_only.assert_not_called()
                            
                            # 验证UI运行代码被调用
                            cli_instance.run.assert_called_once(), "应该调用UI运行主循环"

@patch('argparse.ArgumentParser.error')
def test_cli_conflicting_pre_filter_args(mock_error):
    """Test that using both --pre-filter and --skip-prefilter flags raises an error."""
    # 模拟命令行参数
    test_args = ['cli.py', '-i', '/test/dir', '--pre-filter', '--skip-prefilter']
    
    with patch('sys.argv', test_args):
        # 使用实际的 parse_args 函数
        from knowledge_distiller_kd.cli import parse_args
        
        # 调用 parse_args 应该引发参数冲突错误
        parse_args()
        
        # 验证是否调用了 parser.error
        mock_error.assert_called_once_with("参数冲突：不能同时使用 --pre-filter 和 --skip-prefilter。")

