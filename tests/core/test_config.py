import os
import pytest
from pydantic import ValidationError
from knowledge_distiller_kd.core.config import (
    StorageConfig,
    LoggingConfig,
    EngineConfig,
    AppConfig,
    get_config
)

def test_storage_config_default_values():
    """测试StorageConfig的默认值"""
    config = StorageConfig()
    assert config.database_url == "sqlite+aiosqlite:///./instance/kd_default.sqlite"
    assert config.db_dir == "data"
    assert config.db_name == "kd_tool.db"

def test_storage_config_custom_values():
    """测试StorageConfig的自定义值"""
    test_db_url = "sqlite+aiosqlite:///./test.db"
    config = StorageConfig(database_url=test_db_url)
    assert config.database_url == test_db_url

def test_logging_config_default_values():
    """测试LoggingConfig的默认值"""
    config = LoggingConfig()
    assert config.log_file_path == "logs/kd_tool.log"
    assert config.log_level == "INFO"
    assert config.log_rotation == "10 MB"
    assert config.log_retention == "7 days"
    assert config.log_serialize_json is True
    assert config.log_dir == "logs"
    assert config.log_name == "kd_tool.log"

def test_logging_config_invalid_log_level():
    """测试无效的日志级别会引发验证错误"""
    with pytest.raises(ValidationError):
        LoggingConfig(log_level="INVALID")

def test_engine_config_default_values():
    """测试EngineConfig的默认值"""
    config = EngineConfig()
    assert config.similarity_threshold == 0.85
    assert config.czkawka_path is None
    assert config.semantic_model == "paraphrase-multilingual-MiniLM-L12-v2"
    assert config.batch_size == 32
    assert config.cache_dir == "cache"
    assert config.cache_base_dir == ".kd_cache"

def test_app_config_default_values():
    """测试AppConfig的默认值和嵌套配置"""
    config = AppConfig()
    assert isinstance(config.storage, StorageConfig)
    assert isinstance(config.logging, LoggingConfig)
    assert isinstance(config.engine, EngineConfig)
    
    # 测试从constants.py迁移的默认值
    assert config.db_dir == "data"
    assert config.db_name == "kd_tool.db"
    assert config.log_dir == "logs"
    assert config.log_name == "kd_tool.log"
    assert config.semantic_model == "paraphrase-multilingual-MiniLM-L12-v2"
    assert config.batch_size == 32
    assert config.cache_dir == "cache"
    assert config.cache_base_dir == ".kd_cache"

def test_app_config_from_env():
    """测试从环境变量加载嵌套配置"""
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["SIMILARITY_THRESHOLD"] = "0.90"
    os.environ["BATCH_SIZE"] = "64"
    os.environ["CACHE_DIR"] = "custom_cache"
    
    config = AppConfig()
    assert config.storage.database_url == "sqlite+aiosqlite:///./test.db"
    assert config.logging.log_level == "DEBUG"
    assert config.engine.similarity_threshold == 0.90
    assert config.engine.batch_size == 64
    assert config.engine.cache_dir == "custom_cache"
    
    # 清理环境变量
    del os.environ["DATABASE_URL"]
    del os.environ["LOG_LEVEL"]
    del os.environ["SIMILARITY_THRESHOLD"]
    del os.environ["BATCH_SIZE"]
    del os.environ["CACHE_DIR"]

def test_get_config_singleton():
    """测试get_config()返回单例实例"""
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2  # 验证是同一个实例 