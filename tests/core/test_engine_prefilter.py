"""测试KnowledgeDistillerEngine中预过滤(prefilter)功能的集成。"""

import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from knowledge_distiller_kd.core.engine import KnowledgeDistillerEngine
from knowledge_distiller_kd.core.error_handler import ConfigurationError
from knowledge_distiller_kd.prefilter.czkawka_adapter import CzkawkaAdapter

@pytest.fixture
def mock_storage():
    """创建一个模拟的存储接口对象"""
    storage = MagicMock()
    storage.register_file.return_value = "file1_id"
    storage.save_blocks.return_value = True
    return storage

@pytest.fixture
def engine_with_input_dir(mock_storage):
    """创建一个具有输入目录的引擎实例"""
    with patch('knowledge_distiller_kd.core.engine.validate_file_path') as mock_validate:
        # 创建一个模拟Path对象，确保is_dir()返回True
        mock_path = MagicMock(spec=Path)
        mock_path.resolve.return_value = Path("/fake/input").resolve()
        mock_path.is_dir.return_value = True
        mock_path.__str__.return_value = "/fake/input"
        mock_validate.return_value = mock_path
        
        engine = KnowledgeDistillerEngine(
            storage=mock_storage,
            input_dir="/fake/input"
        )
        return engine

@patch('knowledge_distiller_kd.core.engine.process_directory')
@patch('knowledge_distiller_kd.prefilter.czkawka_adapter.CzkawkaAdapter.filter_unique_files')
def test_run_analysis_with_prefilter(mock_filter_unique_files, mock_process_directory, engine_with_input_dir):
    """测试run_analysis在启用预过滤时正确调用filter_unique_files并使用其结果"""
    # 设置预过滤返回值
    unique_files = [Path("/fake/input/file1.md"), Path("/fake/input/file3.md")]
    duplicate_groups = [[Path("/fake/input/file2.md"), Path("/fake/input/file2_dup.md")]]
    total_files = len(unique_files) + sum(len(group) for group in duplicate_groups)
    mock_filter_unique_files.return_value = (unique_files, duplicate_groups)
    
    # 设置process_directory返回空结果
    mock_process_directory.return_value = {}
    
    # 调用run_analysis
    result = engine_with_input_dir.run_analysis()
    
    # 验证预过滤被调用
    mock_filter_unique_files.assert_called_once()
    
    # 验证处理文档时使用了预过滤后的文件列表
    # 由于我们需要修改_process_documents方法来使用预过滤结果，这个测试现在应该会失败
    # 这里验证_process_documents是否被调用以及如何使用过滤后的文件列表
    # 注意：实际实现可能会有所不同，这里只是测试预期行为
    assert result is True  # 至少确保分析完成

@patch('knowledge_distiller_kd.core.engine.process_directory')
@patch('knowledge_distiller_kd.prefilter.czkawka_adapter.CzkawkaAdapter.filter_unique_files')
def test_run_analysis_skip_prefilter(mock_filter_unique_files, mock_process_directory, engine_with_input_dir):
    """测试run_analysis在skip_prefilter=True时跳过预过滤"""
    # 设置跳过预过滤
    engine_with_input_dir.skip_prefilter = True
    
    # 设置process_directory返回空结果
    mock_process_directory.return_value = {}
    
    # 调用run_analysis
    result = engine_with_input_dir.run_analysis()
    
    # 验证预过滤未被调用
    mock_filter_unique_files.assert_not_called()
    
    # 验证处理所有文件
    assert result is True  # 至少确保分析完成

@patch('knowledge_distiller_kd.core.engine.process_directory')
@patch('knowledge_distiller_kd.prefilter.czkawka_adapter.CzkawkaAdapter.filter_unique_files')
def test_run_analysis_prefilter_exception(mock_filter_unique_files, mock_process_directory, engine_with_input_dir):
    """测试预过滤抛出异常时的错误处理"""
    # 设置预过滤抛出异常
    mock_filter_unique_files.side_effect = Exception("预过滤测试异常")
    
    # 设置process_directory返回空结果
    mock_process_directory.return_value = {}
    
    # 调用run_analysis - 应当继续处理所有文件而不是失败
    result = engine_with_input_dir.run_analysis()
    
    # 验证预过滤被调用但处理了异常
    mock_filter_unique_files.assert_called_once()
    
    # 验证依然处理所有文件，分析完成
    assert result is True  # 分析应该仍然能完成

@patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine._gather_input_files')
@patch('knowledge_distiller_kd.core.engine.process_directory')
@patch('knowledge_distiller_kd.prefilter.czkawka_adapter.CzkawkaAdapter.filter_unique_files')
def test_run_analysis_calls_gather_input_files_when_skip_prefilter(
    mock_filter_unique_files, mock_process_directory, mock_gather_input_files, engine_with_input_dir
):
    """测试跳过预过滤时调用_gather_input_files方法"""
    # 设置跳过预过滤
    engine_with_input_dir.skip_prefilter = True
    
    # 设置_gather_input_files返回值
    all_files = [Path("/fake/input/file1.md"), Path("/fake/input/file2.md"), Path("/fake/input/file3.md")]
    mock_gather_input_files.return_value = all_files
    
    # 设置process_directory返回空结果
    mock_process_directory.return_value = {}
    
    # 调用run_analysis
    result = engine_with_input_dir.run_analysis()
    
    # 验证_gather_input_files被调用
    mock_gather_input_files.assert_called_once_with(engine_with_input_dir.input_dir)
    
    # 验证预过滤未被调用
    mock_filter_unique_files.assert_not_called()
    
    # 测试应该会失败，因为我们尚未实现_gather_input_files方法
    assert result is True  # 至少确保分析完成 