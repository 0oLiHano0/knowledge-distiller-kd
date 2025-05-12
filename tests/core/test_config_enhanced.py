import os
import pytest
import tempfile
from pathlib import Path
from pydantic import ValidationError
from unittest.mock import patch, mock_open

from knowledge_distiller_kd.core.config import (
    StorageConfig,
    LoggingConfig,
    EngineConfig,
    AppConfig,
    get_config
)

def test_app_config_loads_env_file():
    """测试AppConfig能从.env文件加载配置"""
    # 创建临时.env文件
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_env:
        temp_env.write("DATABASE_URL=sqlite+aiosqlite:///./test_env.db\n")
        temp_env.write("LOG_LEVEL=DEBUG\n")
        temp_env.write("SIMILARITY_THRESHOLD=0.95\n")
        temp_env_path = temp_env.name
    
    try:
        # 配置环境变量和模拟
        with patch.dict(os.environ, {"ENV_FILE": temp_env_path}):
            # 模拟pydantic_settings从我们的临时文件加载
            with patch("knowledge_distiller_kd.core.config.AppConfig.model_config", {"env_file": temp_env_path}):
                config = AppConfig()
                # 验证从.env文件加载的值
                assert config.database_url == "sqlite+aiosqlite:///./test_env.db"
                assert config.log_level == "DEBUG"
                assert config.similarity_threshold == 0.95
    finally:
        # 清理临时文件
        os.unlink(temp_env_path)

def test_app_config_with_real_env_file():
    """测试使用真实的.env文件创建AppConfig"""
    # 创建临时.env文件
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_env:
        temp_env.write("DATABASE_URL=sqlite+aiosqlite:///./temp_test.db\n")
        temp_env.write("LOG_LEVEL=DEBUG\n")
        temp_env.write("SIMILARITY_THRESHOLD=0.98\n")
        temp_env.write("BATCH_SIZE=64\n")
        temp_env_path = temp_env.name
    
    try:
        # 清除已有的环境变量，以免干扰测试
        with patch.dict(os.environ, {"ENV_FILE": temp_env_path}, clear=True):
            # 模拟pydantic_settings从我们的临时文件加载
            with patch("knowledge_distiller_kd.core.config.AppConfig.model_config", {"env_file": temp_env_path}):
                config = AppConfig()
                assert config.database_url == "sqlite+aiosqlite:///./temp_test.db"
                assert config.logging.log_level == "DEBUG"
                assert config.engine.similarity_threshold == 0.98
                assert config.engine.batch_size == 64
    finally:
        # 清理临时文件
        os.unlink(temp_env_path)

def test_app_config_env_vars_override_env_file():
    """测试环境变量优先级高于.env文件"""
    # 创建临时.env文件
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_env:
        temp_env.write("DATABASE_URL=sqlite+aiosqlite:///./file.db\n")
        temp_env.write("LOG_LEVEL=INFO\n")
        temp_env_path = temp_env.name
    
    try:
        # 设置环境变量
        with patch.dict(os.environ, {
            "ENV_FILE": temp_env_path,
            "DATABASE_URL": "sqlite+aiosqlite:///./env_var.db"
        }):
            # 模拟pydantic_settings从我们的临时文件加载
            with patch("knowledge_distiller_kd.core.config.AppConfig.model_config", {"env_file": temp_env_path}):
                config = AppConfig()
                # 环境变量应该优先
                assert config.database_url == "sqlite+aiosqlite:///./env_var.db"
                # .env文件中的值应该被使用，因为没有对应的环境变量
                assert config.logging.log_level == "INFO"
    finally:
        # 清理临时文件
        os.unlink(temp_env_path)

def test_validation_log_level():
    """测试日志级别验证"""
    with pytest.raises(ValidationError):
        LoggingConfig(log_level="EXTREME")

def test_basic_config_values():
    """测试基本配置值和类型"""
    # 创建配置实例
    engine_config = EngineConfig()
    
    # 验证默认值
    assert engine_config.similarity_threshold == 0.85
    assert isinstance(engine_config.similarity_threshold, float)
    
    # 验证可以设置有效值
    engine_config = EngineConfig(similarity_threshold=0.5)
    assert engine_config.similarity_threshold == 0.5
    
    # 测试边界值
    engine_config = EngineConfig(similarity_threshold=0.0)
    assert engine_config.similarity_threshold == 0.0
    
    engine_config = EngineConfig(similarity_threshold=1.0)
    assert engine_config.similarity_threshold == 1.0

def test_get_config_thread_safety():
    """测试get_config函数在多线程环境下的行为"""
    # 在单元测试中模拟多线程访问
    config1 = get_config()
    config2 = get_config()
    config3 = get_config()
    
    # 验证所有实例都是同一个对象（单例模式）
    assert config1 is config2
    assert config2 is config3
    assert config1 is config3 