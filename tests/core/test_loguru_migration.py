import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from importlib import import_module, reload

# 导入需要测试的模块，以验证日志迁移
from knowledge_distiller_kd.core.factories import create_logger, create_app_config
from knowledge_distiller_kd.core.config import AppConfig


def test_no_logging_module_usage():
    """
    验证项目中不再直接使用logging模块
    """
    # 检查几个关键文件不再导入logging模块
    key_modules = [
        'knowledge_distiller_kd.core.engine',
        'knowledge_distiller_kd.prefilter.czkawka_adapter',
        'knowledge_distiller_kd.processing.document_processor',
        'knowledge_distiller_kd.storage.orm_storage'
    ]
    
    for module_name in key_modules:
        # 加载模块
        module = import_module(module_name)
        
        # 检查模块是否还直接导入logging
        assert 'logging' not in sys.modules[module_name].__dict__, f"{module_name} 仍在直接使用logging模块"


def test_loguru_imports():
    """
    验证关键模块已正确导入并使用loguru
    """
    key_modules = [
        'knowledge_distiller_kd.core.engine',
        'knowledge_distiller_kd.prefilter.czkawka_adapter',
        'knowledge_distiller_kd.processing.document_processor',
        'knowledge_distiller_kd.storage.orm_storage'
    ]
    
    for module_name in key_modules:
        # 加载模块
        module = import_module(module_name)
        
        # 检查模块是否导入了loguru
        module_dict = sys.modules[module_name].__dict__
        uses_loguru = 'logger' in module_dict and str(type(module_dict['logger'])).find('loguru') != -1
        assert uses_loguru, f"{module_name} 未正确导入或使用loguru.logger"


def test_logger_in_engine():
    """
    验证Engine正确使用了注入的logger
    """
    # 导入Engine类
    from knowledge_distiller_kd.core.engine import KnowledgeDistillerEngine
    
    # 创建mock对象
    mock_storage = MagicMock()
    mock_config = MagicMock()
    mock_logger = MagicMock()
    
    # 创建Engine实例，传入mock logger
    engine = KnowledgeDistillerEngine(
        storage=mock_storage,
        config=mock_config,
        logger=mock_logger,
        similarity_threshold=0.85
    )
    
    # 调用可能使用logger的方法，set_input_dir方法会记录日志
    engine.set_input_dir("dummy_path")
    
    # 验证logger被调用
    assert mock_logger.info.called or mock_logger.debug.called, "Engine未使用注入的logger"


def test_create_logger_factory():
    """
    测试工厂创建的logger是否正确配置
    """
    # 创建临时目录用于日志文件
    tmp_log_path = Path("./tmp_test_logs")
    tmp_log_path.mkdir(exist_ok=True)
    
    # 创建测试配置
    mock_config = MagicMock(spec=AppConfig)
    mock_logging_config = MagicMock()
    mock_logging_config.log_file_path = str(tmp_log_path / "test.log")
    mock_logging_config.log_level = "DEBUG"
    mock_logging_config.log_rotation = "1 MB"
    mock_logging_config.log_retention = "1 day"
    mock_logging_config.log_serialize_json = True
    mock_config.logging = mock_logging_config
    
    # 使用删除和patch来避免影响系统logger
    with patch('loguru.logger.add'), patch('loguru.logger.remove'):
        # 调用工厂方法
        test_logger = create_logger(mock_config)
        
        # 验证logger配置是否生效
        # 由于loguru.logger是一个单例，我们只能间接验证调用了正确的方法
        from loguru import logger
        assert test_logger is logger, "工厂方法未返回正确的loguru.logger实例"
    
    # 清理
    try:
        import shutil
        shutil.rmtree(tmp_log_path)
    except:
        pass 