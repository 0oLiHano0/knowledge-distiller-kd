"""
测试存储生命周期管理和错误处理。

验证:
1. 存储实例的生命周期管理 (initialize, finalize)
2. 引擎中对storage方法调用的错误处理
"""

import pytest
import atexit
from unittest.mock import MagicMock, patch, call
from sqlalchemy.exc import SQLAlchemyError

from knowledge_distiller_kd.core.factories import create_storage, _cleanup_storage
from knowledge_distiller_kd.core.error_handler import KDStorageError
from knowledge_distiller_kd.core.engine import KnowledgeDistillerEngine
from knowledge_distiller_kd.storage.storage_interface import StorageInterface
from knowledge_distiller_kd.core.models import (
    ContentBlock as ContentBlockDTO,
    AnalysisType,
    BlockType
)


@pytest.fixture
def mock_config() -> MagicMock:
    """创建配置的模拟对象"""
    config = MagicMock()
    # 设置必要的配置属性
    config.engine.similarity_threshold = 0.85
    config.engine.semantic_model = "test-model"
    config.engine.batch_size = 32
    config.engine.cache_base_dir = ".kd_cache"
    return config


@pytest.fixture
def mock_storage() -> MagicMock:
    """创建存储接口的模拟对象"""
    storage = MagicMock(spec=StorageInterface)
    return storage


@pytest.fixture
def mock_logger() -> MagicMock:
    """创建日志器的模拟对象"""
    logger = MagicMock()
    return logger


@pytest.fixture
def engine_with_mock_storage(mock_storage, mock_config, mock_logger) -> KnowledgeDistillerEngine:
    """创建使用模拟存储的引擎实例"""
    with patch('knowledge_distiller_kd.core.engine.MD5Analyzer'), \
         patch('knowledge_distiller_kd.core.engine.SemanticAnalyzer'):
        engine = KnowledgeDistillerEngine(
            storage=mock_storage,
            config=mock_config,
            logger=mock_logger
        )
        return engine


# 测试工厂生命周期管理

def test_create_storage_initializes_storage():
    """测试create_storage函数调用storage.initialize()"""
    # 设置模拟对象
    mock_config = MagicMock()
    mock_storage = MagicMock(spec=StorageInterface)
    
    # 模拟ORMStorage创建过程
    with patch('knowledge_distiller_kd.core.factories.ORMStorage', return_value=mock_storage), \
         patch('knowledge_distiller_kd.core.factories.atexit.register') as mock_register:
        
        # 调用工厂函数
        storage = create_storage(mock_config)
        
        # 验证initialize被调用
        mock_storage.initialize.assert_called_once()
        
        # 验证atexit.register被调用，用于注册清理函数
        mock_register.assert_called_once_with(_cleanup_storage)
        
        # 验证返回了正确的对象
        assert storage == mock_storage


def test_create_storage_handles_initialization_errors():
    """测试create_storage函数处理初始化错误"""
    # 设置模拟对象
    mock_config = MagicMock()
    mock_storage = MagicMock(spec=StorageInterface)
    mock_storage.initialize.side_effect = SQLAlchemyError("Database error")
    
    # 模拟ORMStorage创建过程
    with patch('knowledge_distiller_kd.core.factories.ORMStorage', return_value=mock_storage), \
         patch('knowledge_distiller_kd.core.factories.logger'):
        
        # 调用工厂函数，应该抛出KDStorageError
        with pytest.raises(KDStorageError) as excinfo:
            create_storage(mock_config)
        
        # 验证错误消息包含原始异常信息
        assert "数据库初始化失败" in str(excinfo.value)
        assert "Database error" in str(excinfo.value)


def test_cleanup_storage_calls_finalize():
    """测试_cleanup_storage函数调用storage.finalize()"""
    # 设置模拟对象
    mock_storage = MagicMock(spec=StorageInterface)
    
    # 模拟全局存储实例
    with patch('knowledge_distiller_kd.core.factories._storage_instance', mock_storage), \
         patch('knowledge_distiller_kd.core.factories.logger'):
        
        # 调用清理函数
        _cleanup_storage()
        
        # 验证finalize被调用
        mock_storage.finalize.assert_called_once()


def test_cleanup_storage_handles_finalize_errors():
    """测试_cleanup_storage函数处理finalize错误"""
    # 设置模拟对象
    mock_storage = MagicMock(spec=StorageInterface)
    mock_storage.finalize.side_effect = Exception("Finalize error")
    
    # 模拟全局存储实例
    with patch('knowledge_distiller_kd.core.factories._storage_instance', mock_storage), \
         patch('knowledge_distiller_kd.core.factories.logger') as mock_logger:
        
        # 调用清理函数，不应该抛出异常
        _cleanup_storage()
        
        # 验证错误被记录
        mock_logger.exception.assert_called_once()
        
        # 验证finalize被调用
        mock_storage.finalize.assert_called_once()


# 测试引擎中的存储错误处理

def test_engine_handles_register_file_error(engine_with_mock_storage, mock_storage, mock_logger):
    """测试引擎处理register_file错误"""
    # 配置mock抛出异常
    mock_storage.register_file.side_effect = SQLAlchemyError("Database error")
    
    # 调用使用register_file的方法
    result = engine_with_mock_storage._process_files([MagicMock()])
    
    # 验证错误被记录
    mock_logger.exception.assert_called()
    
    # 验证方法返回False表示失败
    assert result is False


def test_engine_handles_save_blocks_error(engine_with_mock_storage, mock_storage, mock_logger):
    """测试引擎处理save_blocks错误"""
    # 设置成功的register_file调用
    mock_storage.register_file.return_value = "test_file_id"
    
    # 但save_blocks抛出异常
    mock_storage.save_blocks.side_effect = SQLAlchemyError("Database error")
    
    # 创建测试块
    test_block = ContentBlockDTO(
        block_id="test_block_id",
        file_id="test_file_id",
        block_type=BlockType.TEXT,
        text="Test content",
        metadata={}
    )
    
    # 调用更新块的方法
    result = engine_with_mock_storage.update_block(test_block)
    
    # 验证错误被记录
    mock_logger.exception.assert_called()
    
    # 验证方法返回False表示失败
    assert result is False


def test_engine_handles_get_block_error(engine_with_mock_storage, mock_storage, mock_logger):
    """测试引擎处理get_block错误"""
    # 配置mock抛出异常
    mock_storage.get_block.side_effect = SQLAlchemyError("Database error")
    
    # 调用获取块的方法
    result = engine_with_mock_storage.get_block("test_block_id")
    
    # 验证错误被记录
    mock_logger.exception.assert_called()
    
    # 验证方法返回None表示失败
    assert result is None


def test_engine_handles_get_analysis_results_error(engine_with_mock_storage, mock_storage, mock_logger):
    """测试引擎处理get_analysis_results错误"""
    # 配置mock抛出异常
    mock_storage.get_analysis_results.side_effect = SQLAlchemyError("Database error")
    
    # 调用获取分析结果的方法
    result = engine_with_mock_storage.get_analysis_results(AnalysisType.MD5_DUPLICATE)
    
    # 验证错误被记录
    mock_logger.exception.assert_called()
    
    # 验证方法返回空列表表示失败
    assert result == []


def test_engine_handles_update_decision_db_error(engine_with_mock_storage, mock_storage, mock_logger):
    """测试引擎处理update_decision中的数据库错误"""
    # 设置成功的get_block调用
    mock_block = MagicMock()
    mock_block.file_id = "test_file_id"
    mock_block.metadata = {}
    mock_storage.get_block.return_value = mock_block
    
    # 但save_blocks抛出异常
    mock_storage.save_blocks.side_effect = SQLAlchemyError("Database error")
    
    # 调用更新决策的方法
    result = engine_with_mock_storage.update_decision("block_id:other_info", "keep")
    
    # 验证错误被记录
    mock_logger.exception.assert_called()
    
    # 验证方法返回False表示失败
    assert result is False 