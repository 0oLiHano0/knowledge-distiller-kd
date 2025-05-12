"""
依赖管理工厂测试模块。

测试 core/factories.py 中的工厂函数，确保能正确创建和配置依赖项。
"""

import pytest
from unittest.mock import patch, MagicMock

from knowledge_distiller_kd.core.factories import (
    create_app_config,
    create_storage,
    create_logger,
    create_engine
)
from knowledge_distiller_kd.core.config import AppConfig
from knowledge_distiller_kd.storage.storage_interface import StorageInterface
from knowledge_distiller_kd.storage.orm_storage import ORMStorage
from loguru import logger


def test_create_app_config():
    """测试 create_app_config 函数返回有效的 AppConfig 实例。"""
    # 调用工厂函数获取配置
    config = create_app_config()
    
    # 验证返回类型和基本属性
    assert isinstance(config, AppConfig)
    assert hasattr(config, 'storage')
    assert hasattr(config, 'logging')
    assert hasattr(config, 'engine')


def test_create_storage():
    """测试 create_storage 函数使用配置创建并初始化 ORMStorage。"""
    # 创建一个配置实例
    config = MagicMock(spec=AppConfig)
    config.storage.database_url = "sqlite+aiosqlite:///./test.db"
    
    # 使用patch模拟ORMStorage的行为
    with patch('knowledge_distiller_kd.core.factories.ORMStorage') as mock_orm_storage:
        # 设置mock实例的行为
        mock_instance = MagicMock()
        mock_orm_storage.return_value = mock_instance
        
        # 调用工厂函数
        storage = create_storage(config)
        
        # 验证创建了ORMStorage实例
        mock_orm_storage.assert_called_once()
        
        # 验证调用了initialize方法
        mock_instance.initialize.assert_called_once()
        
        # 验证返回了正确的实例
        assert storage == mock_instance


def test_create_logger():
    """测试 create_logger 函数创建并配置Loguru日志器实例。"""
    # 创建一个配置实例
    config = MagicMock(spec=AppConfig)
    
    # 设置日志配置
    logging_config = MagicMock()
    logging_config.log_file_path = "logs/test.log"
    logging_config.log_level = "INFO"
    logging_config.log_rotation = "10 MB"
    logging_config.log_retention = "7 days"
    logging_config.log_serialize_json = True
    config.logging = logging_config
    
    # 模拟Loguru logger和Path
    with patch('knowledge_distiller_kd.core.factories.logger') as mock_logger, \
         patch('knowledge_distiller_kd.core.factories.Path') as mock_path:
        
        # 设置Path对象的行为
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.parent = MagicMock()
        mock_path_instance.parent.exists.return_value = True
        
        # 配置mock logger
        mock_logger.remove = MagicMock()
        mock_logger.add = MagicMock()
        
        # 调用工厂函数
        result = create_logger(config)
        
        # 验证移除了默认处理器
        mock_logger.remove.assert_called_once()
        
        # 验证添加了文件处理器和控制台处理器
        assert mock_logger.add.call_count == 2
        
        # 验证返回了正确的实例
        assert result == mock_logger


def test_create_engine():
    """测试 create_engine 函数返回引擎实例（已更新）。"""
    # 创建模拟对象
    storage = MagicMock(spec=StorageInterface)
    config = MagicMock(spec=AppConfig)
    
    # 配置 engine 属性
    engine_config = MagicMock()
    engine_config.similarity_threshold = 0.85
    engine_config.semantic_model = "test-model"
    engine_config.batch_size = 32
    engine_config.cache_dir = "cache"
    engine_config.cache_base_dir = ".kd_cache"
    config.engine = engine_config
    
    logger_mock = MagicMock()
    
    # 模拟 KnowledgeDistillerEngine 类
    with patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine') as MockEngine:
        # 设置返回值
        engine_instance = MagicMock()
        MockEngine.return_value = engine_instance
        
        # 调用工厂函数
        result = create_engine(storage, config, logger_mock)
        
        # 验证引擎创建时使用了正确的参数
        MockEngine.assert_called_once_with(
            storage=storage,
            config=config,
            logger=logger_mock,
            similarity_threshold=config.engine.similarity_threshold
        )
        
        # 验证返回了正确的实例
        assert result is engine_instance 