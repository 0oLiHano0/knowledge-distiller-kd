import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sqlalchemy.exc

from knowledge_distiller_kd.core.factories import (
    create_app_config,
    create_storage,
    create_logger,
    create_engine
)
from knowledge_distiller_kd.core.config import AppConfig, LoggingConfig
from knowledge_distiller_kd.storage.storage_interface import StorageInterface
from knowledge_distiller_kd.storage.orm_storage import ORMStorage
from knowledge_distiller_kd.core.error_handler import KDStorageError, ConfigurationError


def test_create_storage_handles_initialization_error():
    """测试 create_storage 处理存储初始化错误"""
    # 创建配置
    config = MagicMock(spec=AppConfig)
    config.storage.database_url = "sqlite+aiosqlite:///./test.db"
    
    # 模拟 ORMStorage 实例
    mock_orm_storage = MagicMock()
    mock_orm_storage.initialize.side_effect = sqlalchemy.exc.SQLAlchemyError("模拟数据库连接失败")
    
    # 模拟 ORMStorage 类
    with patch('knowledge_distiller_kd.core.factories.ORMStorage', return_value=mock_orm_storage) as mock_storage_class:
        # 确认抛出适当的异常
        with pytest.raises(KDStorageError):
            create_storage(config)
        
        # 验证调用
        mock_storage_class.assert_called_once()
        mock_orm_storage.initialize.assert_called_once()
        
        # 验证资源清理被尝试
        mock_orm_storage.finalize.assert_called_once()


def test_create_storage_validates_config():
    """测试 create_storage 验证配置"""
    # 测试空配置
    with pytest.raises(ConfigurationError) as excinfo:
        create_storage(None)
    assert "配置不能为空" in str(excinfo.value)
    
    # 测试空数据库URL
    config = MagicMock(spec=AppConfig)
    config.storage.database_url = ""
    
    with pytest.raises(ConfigurationError) as excinfo:
        create_storage(config)
    assert "数据库URL不能为空" in str(excinfo.value)


def test_create_logger_ensures_log_directory_exists():
    """测试 create_logger 确保日志目录存在"""
    # 创建配置
    config = MagicMock(spec=AppConfig)
    logging_config = MagicMock(spec=LoggingConfig)
    logging_config.log_file_path = "/non_existent_dir/test.log"
    logging_config.log_level = "INFO"
    logging_config.log_rotation = "10 MB"
    logging_config.log_retention = "7 days"
    logging_config.log_serialize_json = True
    config.logging = logging_config
    
    # 模拟 Path 对象及其方法
    mock_path_instance = MagicMock()
    mock_path_instance.parent = MagicMock()
    mock_path_instance.parent.exists.return_value = False
    
    # 模拟 logger 对象
    mock_logger = MagicMock()
    
    with patch('knowledge_distiller_kd.core.factories.Path', return_value=mock_path_instance) as mock_path, \
         patch('knowledge_distiller_kd.core.factories.logger', mock_logger):
        
        # 调用工厂函数
        result = create_logger(config)
        
        # 验证目录检查和创建
        mock_path_instance.parent.exists.assert_called_once()
        mock_path_instance.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        # 验证 logger 配置
        assert mock_logger.remove.call_count == 1
        assert mock_logger.add.call_count == 2
        assert result == mock_logger


def test_create_logger_validates_config():
    """测试 create_logger 验证配置"""
    # 测试空配置
    with pytest.raises(ConfigurationError) as excinfo:
        create_logger(None)
    assert "配置不能为空" in str(excinfo.value)
    
    # 测试配置异常
    config = MagicMock(spec=AppConfig)
    config.logging.side_effect = AttributeError("模拟配置访问错误")
    
    with pytest.raises(ConfigurationError) as excinfo:
        create_logger(config)
    assert "日志系统初始化失败" in str(excinfo.value)


def test_create_logger_with_real_temp_dir():
    """使用真实临时目录测试 create_logger"""
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = Path(temp_dir) / "logs" / "test.log"
        
        # 创建配置
        config = MagicMock(spec=AppConfig)
        logging_config = MagicMock(spec=LoggingConfig)
        logging_config.log_file_path = str(log_path)
        logging_config.log_level = "INFO"
        logging_config.log_rotation = "10 MB"
        logging_config.log_retention = "7 days"
        logging_config.log_serialize_json = True
        config.logging = logging_config
        
        # 模拟 logger
        mock_logger = MagicMock()
        
        with patch('knowledge_distiller_kd.core.factories.logger', mock_logger):
            # 调用工厂函数
            result = create_logger(config)
            
            # 验证日志目录已创建
            assert log_path.parent.exists()
            
            # 验证 logger 配置
            assert mock_logger.remove.call_count == 1
            assert mock_logger.add.call_count == 2


def test_create_engine_validates_parameters():
    """测试 create_engine 验证参数"""
    # 创建依赖
    mock_storage = MagicMock(spec=StorageInterface)
    mock_config = MagicMock(spec=AppConfig)
    mock_logger = MagicMock()
    
    # 测试空存储
    with pytest.raises(ConfigurationError) as excinfo:
        create_engine(None, mock_config, mock_logger)
    assert "存储接口不能为空" in str(excinfo.value)
    
    # 测试空配置
    with pytest.raises(ConfigurationError) as excinfo:
        create_engine(mock_storage, None, mock_logger)
    assert "配置不能为空" in str(excinfo.value)
    
    # 测试空日志器
    with pytest.raises(ConfigurationError) as excinfo:
        create_engine(mock_storage, mock_config, None)
    assert "日志器不能为空" in str(excinfo.value)


def test_create_engine_handles_exceptions():
    """测试 create_engine 处理异常"""
    # 创建依赖
    mock_storage = MagicMock(spec=StorageInterface)
    mock_config = MagicMock(spec=AppConfig)
    mock_logger = MagicMock()
    
    # 创建引擎配置
    engine_config = MagicMock()
    mock_config.engine = engine_config
    
    # 模拟引擎导入异常
    with patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine', side_effect=ImportError("模拟导入错误")):
        with pytest.raises(ConfigurationError) as excinfo:
            create_engine(mock_storage, mock_config, mock_logger)
        assert "引擎创建失败" in str(excinfo.value)


def test_create_engine_with_valid_params():
    """测试使用有效参数创建引擎"""
    # 创建依赖
    storage = MagicMock(spec=StorageInterface)
    config = MagicMock(spec=AppConfig)
    logger_mock = MagicMock()
    
    # 配置引擎配置
    engine_config = MagicMock()
    engine_config.similarity_threshold = 0.87
    config.engine = engine_config
    
    # 模拟引擎类
    with patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine') as MockEngine:
        engine_instance = MagicMock()
        MockEngine.return_value = engine_instance
        
        # 调用工厂函数
        result = create_engine(storage, config, logger_mock)
        
        # 验证引擎创建正确
        MockEngine.assert_called_once_with(
            storage=storage,
            config=config,
            logger=logger_mock,
            similarity_threshold=config.engine.similarity_threshold
        )
        assert result == engine_instance


def test_create_engine_with_complete_config():
    """测试使用完整配置创建引擎，包括可选参数"""
    # 创建依赖
    storage = MagicMock(spec=StorageInterface)
    config = MagicMock(spec=AppConfig)
    logger_mock = MagicMock()
    
    # 配置完整的引擎配置
    engine_config = MagicMock()
    engine_config.similarity_threshold = 0.87
    engine_config.semantic_model = "custom-model"
    engine_config.batch_size = 64
    engine_config.cache_dir = "custom_cache"
    engine_config.cache_base_dir = "custom_base"
    config.engine = engine_config
    
    # 模拟引擎类
    with patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine') as MockEngine:
        engine_instance = MagicMock()
        MockEngine.return_value = engine_instance
        
        # 调用工厂函数
        result = create_engine(storage, config, logger_mock)
        
        # 验证引擎创建正确
        MockEngine.assert_called_once_with(
            storage=storage,
            config=config,
            logger=logger_mock,
            similarity_threshold=config.engine.similarity_threshold
        )
        assert result == engine_instance


def test_create_app_config_handles_exceptions():
    """测试 create_app_config 处理异常"""
    # 模拟配置加载异常
    with patch('knowledge_distiller_kd.core.factories.get_config', side_effect=ValueError("模拟配置加载错误")):
        with pytest.raises(ConfigurationError) as excinfo:
            create_app_config()
        assert "配置加载失败" in str(excinfo.value) 