"""测试KnowledgeDistillerEngine中预过滤(prefilter)日志和统计功能。"""

import pytest
import time
from unittest.mock import MagicMock, patch, call, ANY
from pathlib import Path
import logging

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

@patch('time.monotonic')
@patch('knowledge_distiller_kd.prefilter.czkawka_adapter.CzkawkaAdapter.filter_unique_files')
@patch('knowledge_distiller_kd.core.engine.logger')  # 直接模拟engine.py中使用的logger
def test_run_prefilter_only_logs_statistics(mock_logger, mock_filter_unique_files, mock_monotonic, engine_with_input_dir):
    """测试run_prefilter_only方法正确记录统计信息日志"""
    # 设置时间模拟，返回两个不同的值，计算耗时为1秒
    mock_monotonic.side_effect = [100.0, 101.0]
    
    # 设置预过滤返回值
    unique_files = [Path("/fake/input/file1.md"), Path("/fake/input/file3.md")]
    duplicate_groups = [
        [Path("/fake/input/file2.md"), Path("/fake/input/file2_dup.md")],
        [Path("/fake/input/file4.md"), Path("/fake/input/file4_dup1.md"), Path("/fake/input/file4_dup2.md")]
    ]
    mock_filter_unique_files.return_value = (unique_files, duplicate_groups)
    
    # 为了处理bind方法，我们需要设置一个mock对象来响应bind调用
    mock_bind = MagicMock()
    mock_logger.bind.return_value = mock_bind
    
    # 调用run_prefilter_only
    total, uniques, dupes = engine_with_input_dir.run_prefilter_only()
    
    # 验证总文件数、唯一文件数和重复文件组数
    assert total == 7  # 2个唯一文件 + 2个文件在第一组 + 3个文件在第二组
    assert len(uniques) == 2
    assert len(dupes) == 2
    
    # 验证日志记录
    # 1. 验证常规日志记录
    mock_logger.info.assert_any_call(f"[Prefilter] Scanned {total} files, filtered {3} duplicates → {2} remain. (耗时: {1000}ms)")
    
    # 2. 如果使用loguru，验证结构化日志记录
    mock_logger.bind.assert_called_with(
        total_files=7,
        filtered_count=3,
        unique_count=2,
        elapsed_ms=1000
    )
    mock_bind.info.assert_called_with("prefilter_summary")

@patch('time.monotonic')
@patch('knowledge_distiller_kd.prefilter.czkawka_adapter.CzkawkaAdapter.filter_unique_files')
@patch('knowledge_distiller_kd.core.engine.logger')  # 直接模拟engine.py中使用的logger
@patch('knowledge_distiller_kd.core.engine.process_directory')
def test_run_analysis_prefilter_logs_statistics(mock_process_directory, mock_logger, mock_filter_unique_files, mock_monotonic, engine_with_input_dir):
    """测试run_analysis方法中的预过滤步骤正确记录统计信息日志"""
    # 设置时间模拟，返回两个不同的值，计算耗时为1秒
    mock_monotonic.side_effect = [100.0, 101.0]
    
    # 设置预过滤返回值
    unique_files = [Path("/fake/input/file1.md"), Path("/fake/input/file3.md")]
    duplicate_groups = [
        [Path("/fake/input/file2.md"), Path("/fake/input/file2_dup.md")],
        [Path("/fake/input/file4.md"), Path("/fake/input/file4_dup1.md"), Path("/fake/input/file4_dup2.md")]
    ]
    mock_filter_unique_files.return_value = (unique_files, duplicate_groups)
    
    # 设置process_directory返回空结果
    mock_process_directory.return_value = {}
    
    # 为了处理bind方法，我们需要设置一个mock对象来响应bind调用
    mock_bind = MagicMock()
    mock_logger.bind.return_value = mock_bind
    
    # 调用run_analysis
    result = engine_with_input_dir.run_analysis()
    
    # 验证日志记录
    # 1. 验证常规日志记录
    mock_logger.info.assert_any_call(f"[Prefilter] Scanned {7} files, filtered {3} duplicates → {2} remain. (耗时: {1000}ms)")
    
    # 2. 验证结构化日志记录
    mock_logger.bind.assert_any_call(
        total_files=7,
        filtered_count=3,  # 2组中共3个重复文件(第一组1个重复，第二组2个重复)
        unique_count=2,
        elapsed_ms=1000
    )
    mock_bind.info.assert_any_call("prefilter_summary")
    
    # 验证分析完成
    assert result is True

@patch('knowledge_distiller_kd.prefilter.czkawka_adapter.CzkawkaAdapter.filter_unique_files')
def test_run_prefilter_only_no_input_dir(mock_filter_unique_files, mock_storage):
    """测试run_prefilter_only在未设置输入目录时抛出异常"""
    # 创建一个没有输入目录的引擎
    engine = KnowledgeDistillerEngine(storage=mock_storage)
    
    # 验证抛出ConfigurationError异常
    with pytest.raises(ConfigurationError, match="Input directory not set"):
        engine.run_prefilter_only()
    
    # 验证预过滤未被调用
    mock_filter_unique_files.assert_not_called()

@patch('knowledge_distiller_kd.prefilter.czkawka_adapter.CzkawkaAdapter.filter_unique_files')
def test_run_prefilter_only_exception_handling(mock_filter_unique_files, engine_with_input_dir):
    """测试run_prefilter_only在预过滤抛出异常时的错误处理"""
    # 设置预过滤抛出异常
    test_exception = Exception("预过滤测试异常")
    mock_filter_unique_files.side_effect = test_exception
    
    # 验证抛出AnalysisError异常
    with pytest.raises(Exception, match="预过滤失败"):
        engine_with_input_dir.run_prefilter_only()

@patch('time.monotonic')
@patch('knowledge_distiller_kd.prefilter.czkawka_adapter.CzkawkaAdapter.filter_unique_files')
@patch('knowledge_distiller_kd.core.engine.logger')  # 直接模拟engine.py中使用的logger
@patch('knowledge_distiller_kd.core.engine.hasattr', return_value=False)  # 使用patch装饰器替代with语句
def test_run_prefilter_only_fallback_to_standard_logging(mock_hasattr, mock_logger, mock_filter_unique_files, mock_monotonic, engine_with_input_dir):
    """测试run_prefilter_only在没有bind方法时使用标准日志格式"""
    # 设置时间模拟，返回两个不同的值，计算耗时为1秒
    mock_monotonic.side_effect = [100.0, 101.0]
    
    # 设置预过滤返回值
    unique_files = [Path("/fake/input/file1.md"), Path("/fake/input/file3.md")]
    duplicate_groups = [
        [Path("/fake/input/file2.md"), Path("/fake/input/file2_dup.md")],
        [Path("/fake/input/file4.md"), Path("/fake/input/file4_dup1.md"), Path("/fake/input/file4_dup2.md")]
    ]
    mock_filter_unique_files.return_value = (unique_files, duplicate_groups)
    
    # 调用run_prefilter_only
    total, uniques, dupes = engine_with_input_dir.run_prefilter_only()
    
    # 验证总文件数、唯一文件数和重复文件组数
    assert total == 7
    assert len(uniques) == 2
    assert len(dupes) == 2
    
    # 验证使用了标准日志格式
    mock_logger.info.assert_any_call(f"prefilter_summary: total_files={total}, filtered_count=3, unique_count=2, elapsed_ms=1000") 