"""
引擎依赖注入测试模块。

测试 KnowledgeDistillerEngine 的依赖注入机制，确保引擎能够正确接收和使用注入的依赖项。
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any

from knowledge_distiller_kd.core.config import AppConfig
from knowledge_distiller_kd.core.engine import KnowledgeDistillerEngine
from knowledge_distiller_kd.storage.storage_interface import StorageInterface


@pytest.fixture
def mock_logger() -> MagicMock:
    """
    创建 logger 的模拟对象
    """
    logger = MagicMock()
    return logger


@pytest.fixture
def mock_config() -> MagicMock:
    """
    创建配置的模拟对象，模拟 AppConfig 的行为
    """
    config = MagicMock(spec=AppConfig)
    
    # 配置 engine 属性
    engine_config = MagicMock()
    engine_config.similarity_threshold = 0.85
    engine_config.semantic_model = "test-model"
    engine_config.batch_size = 32
    engine_config.cache_dir = "cache"
    engine_config.cache_base_dir = ".kd_cache"
    config.engine = engine_config
    
    return config


@pytest.fixture
def mock_storage() -> MagicMock:
    """
    创建存储接口的模拟对象
    """
    storage = MagicMock(spec=StorageInterface)
    return storage


def test_engine_init_with_di(mock_storage: MagicMock, mock_config: MagicMock, mock_logger: MagicMock, tmp_path: Path):
    """
    测试引擎初始化时能正确接收并存储依赖注入的对象
    """
    # 创建输入目录
    input_dir = tmp_path / "input_dir"
    input_dir.mkdir()
    
    # 使用 patch 模拟 MD5Analyzer 和 SemanticAnalyzer
    with patch('knowledge_distiller_kd.core.engine.MD5Analyzer') as MockMD5, \
         patch('knowledge_distiller_kd.core.engine.SemanticAnalyzer') as MockSemantic:
        
        # 初始化引擎
        engine = KnowledgeDistillerEngine(
            storage=mock_storage,
            config=mock_config,
            logger=mock_logger,
            input_dir=input_dir
        )
        
        # 验证依赖项被正确存储
        assert engine.storage is mock_storage
        assert engine.config is mock_config
        assert engine.logger is mock_logger
        
        # 验证引擎从配置中读取了相似度阈值
        assert engine.similarity_threshold == mock_config.engine.similarity_threshold
        
        # 验证分析器初始化时使用了正确的配置
        MockSemantic.assert_called_once_with(
            similarity_threshold=mock_config.engine.similarity_threshold,
            model_name=mock_config.engine.semantic_model,
            batch_size=mock_config.engine.batch_size,
            cache_dir=mock_config.engine.cache_base_dir
        )


def test_engine_uses_injected_logger(mock_storage: MagicMock, mock_config: MagicMock, mock_logger: MagicMock):
    """
    测试引擎使用注入的 logger 进行日志记录
    """
    with patch('knowledge_distiller_kd.core.engine.MD5Analyzer'), \
         patch('knowledge_distiller_kd.core.engine.SemanticAnalyzer'):
        
        # 初始化引擎
        engine = KnowledgeDistillerEngine(
            storage=mock_storage,
            config=mock_config,
            logger=mock_logger
        )
        
        # 调用会产生日志的方法
        engine.set_similarity_threshold(0.9)
        
        # 验证注入的 logger 被调用
        mock_logger.info.assert_called()


def test_engine_uses_injected_storage(mock_storage: MagicMock, mock_config: MagicMock, mock_logger: MagicMock, tmp_path: Path):
    """
    测试引擎使用注入的 storage 对象
    """
    # 创建输入目录
    input_dir = tmp_path / "input_dir"
    input_dir.mkdir()
    
    with patch('knowledge_distiller_kd.core.engine.MD5Analyzer'), \
         patch('knowledge_distiller_kd.core.engine.SemanticAnalyzer'):
        
        # 初始化引擎
        engine = KnowledgeDistillerEngine(
            storage=mock_storage,
            config=mock_config,
            logger=mock_logger,
            input_dir=input_dir
        )
        
        # 创建测试文件
        test_file = input_dir / "test.md"
        test_file.touch()
        
        # 直接调用 register_file 方法
        mock_storage.register_file.return_value = "test_file_id"
        file_id = engine.storage.register_file(str(test_file))
        
        # 验证 storage 的方法被调用
        mock_storage.register_file.assert_called_once_with(str(test_file))
        assert file_id == "test_file_id" 