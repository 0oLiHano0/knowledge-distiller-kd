"""
依赖管理工厂模块。

提供一组工厂函数，用于创建和获取应用程序依赖项，包括：
- 应用配置 (AppConfig)
- 存储实例 (StorageInterface)
- 日志器 (logger)
- 引擎实例 (KnowledgeDistillerEngine)

该模块采用集中化的依赖创建和提供，便于依赖注入和测试。
"""

from typing import Optional, Any

from knowledge_distiller_kd.core.config import AppConfig, get_config
from knowledge_distiller_kd.storage.storage_interface import StorageInterface
from knowledge_distiller_kd.storage.orm_storage import ORMStorage
from loguru import logger


def create_app_config() -> AppConfig:
    """
    创建并返回应用配置实例。
    
    Returns:
        AppConfig: 配置实例
    """
    return get_config()


def create_storage(config: AppConfig) -> StorageInterface:
    """
    创建并返回存储实例。
    
    Args:
        config (AppConfig): 应用配置
        
    Returns:
        StorageInterface: 存储接口实现
    """
    # 使用配置中的数据库URL创建ORMStorage
    storage = ORMStorage()
    # 初始化存储实例
    storage.initialize()
    return storage


def create_logger(config: AppConfig) -> Any:
    """
    创建并返回日志器实例。
    
    Args:
        config (AppConfig): 应用配置
        
    Returns:
        Any: 日志器实例（占位，将在 KD-LOGGING-001 中实现）
    """
    # 占位实现，将在 KD-LOGGING-001 中完成
    return logger


def create_engine(storage: StorageInterface, config: AppConfig, logger: Any) -> Any:
    """
    创建并返回引擎实例。
    
    Args:
        storage (StorageInterface): 存储接口实现
        config (AppConfig): 应用配置
        logger: 日志器实例
        
    Returns:
        KnowledgeDistillerEngine: 引擎实例
    """
    from knowledge_distiller_kd.core.engine import KnowledgeDistillerEngine
    
    # 从配置中获取引擎参数
    engine_config = config.engine
    
    # 创建引擎实例
    engine = KnowledgeDistillerEngine(
        storage=storage,
        config=config,
        logger=logger,
        similarity_threshold=engine_config.similarity_threshold
    )
    
    return engine 