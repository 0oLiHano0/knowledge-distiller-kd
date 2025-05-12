"""
依赖管理工厂模块。

提供一组工厂函数，用于创建和获取应用程序依赖项，包括：
- 应用配置 (AppConfig)
- 存储实例 (StorageInterface)
- 日志器 (logger)
- 引擎实例 (KnowledgeDistillerEngine)

该模块采用集中化的依赖创建和提供，便于依赖注入和测试。
"""

import atexit
import sys
from typing import Optional, Any, Dict
from pathlib import Path
from sqlalchemy.exc import SQLAlchemyError

from knowledge_distiller_kd.core.config import AppConfig, get_config
from knowledge_distiller_kd.storage.storage_interface import StorageInterface
from knowledge_distiller_kd.storage.orm_storage import ORMStorage
from knowledge_distiller_kd.core.error_handler import KDStorageError, ConfigurationError
from loguru import logger

# 存储全局存储实例引用，用于资源清理
_storage_instance: Optional[StorageInterface] = None

def create_app_config() -> AppConfig:
    """
    创建并返回应用配置实例。
    
    Returns:
        AppConfig: 配置实例
    """
    try:
        return get_config()
    except Exception as e:
        error_msg = f"配置加载失败: {str(e)}"
        logger.exception(error_msg)
        raise ConfigurationError(error_msg, "CONFIG_LOAD_ERROR", {"original": str(e)}) from e


def create_storage(config: AppConfig) -> StorageInterface:
    """
    创建并返回存储实例。
    
    Args:
        config (AppConfig): 应用配置
        
    Returns:
        StorageInterface: 存储接口实现
        
    Raises:
        ConfigurationError: 当配置无效时
        KDStorageError: 当存储初始化失败时
    """
    global _storage_instance
    
    # 参数验证
    if config is None:
        error_msg = "配置不能为空"
        logger.error(error_msg)
        raise ConfigurationError(error_msg, "CONFIG_EMPTY_ERROR")
    
    # 验证数据库URL
    if not config.storage.database_url:
        error_msg = "数据库URL不能为空"
        logger.error(error_msg)
        raise ConfigurationError(error_msg, "CONFIG_DB_URL_EMPTY")
    
    try:
        # 使用配置中的数据库URL创建ORMStorage
        storage = ORMStorage()
        
        # 初始化存储实例
        try:
            storage.initialize()
            logger.info("Storage initialized successfully")
            
            # 保存全局引用用于清理
            _storage_instance = storage
            
            # 注册应用退出时的清理函数
            atexit.register(_cleanup_storage)
            
            return storage
        except SQLAlchemyError as e:
            error_msg = f"数据库初始化失败: {str(e)}"
            logger.exception(error_msg)
            
            # 确保资源被释放
            if hasattr(storage, 'finalize'):
                try:
                    storage.finalize()
                except Exception as finalize_error:
                    logger.exception(f"清理存储资源时出错: {finalize_error}")
            
            raise KDStorageError(error_msg, "DB_INIT_ERROR", {"original": str(e)}) from e
        except Exception as e:
            error_msg = f"存储初始化失败: {str(e)}"
            logger.exception(error_msg)
            
            # 确保资源被释放
            if hasattr(storage, 'finalize'):
                try:
                    storage.finalize()
                except Exception as finalize_error:
                    logger.exception(f"清理存储资源时出错: {finalize_error}")
            
            raise KDStorageError(error_msg, "STORAGE_INIT_ERROR", {"original": str(e)}) from e
    except Exception as e:
        # 处理创建ORMStorage实例时的错误
        if not isinstance(e, (ConfigurationError, KDStorageError)):
            error_msg = f"创建存储实例失败: {str(e)}"
            logger.exception(error_msg)
            raise KDStorageError(error_msg, "STORAGE_CREATE_ERROR", {"original": str(e)}) from e
        raise


def _cleanup_storage() -> None:
    """
    清理存储资源的函数，在应用退出时自动调用。
    """
    global _storage_instance
    
    if _storage_instance is not None:
        try:
            logger.info("Finalizing storage...")
            _storage_instance.finalize()
            logger.info("Storage finalized successfully")
        except Exception as e:
            logger.exception(f"Error finalizing storage: {e}")
        finally:
            _storage_instance = None


def create_logger(config: AppConfig) -> Any:
    """
    创建并返回日志器实例。
    
    Args:
        config (AppConfig): 应用配置
        
    Returns:
        Any: 配置好的loguru日志器实例
        
    Raises:
        ConfigurationError: 当配置无效时
    """
    # 参数验证
    if config is None:
        error_msg = "配置不能为空"
        # 这里直接使用Python的异常，因为logger可能未初始化
        raise ConfigurationError(error_msg, "CONFIG_EMPTY_ERROR")
    
    try:
        # 从配置中获取日志配置
        logging_config = config.logging
        
        # 移除默认处理器，避免重复输出
        logger.remove()
        
        # 确保日志目录存在
        log_file_path = Path(logging_config.log_file_path)
        if not log_file_path.parent.exists():
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 添加文件日志处理器
        logger.add(
            sink=logging_config.log_file_path,
            level=logging_config.log_level.upper(),
            rotation=logging_config.log_rotation,
            retention=logging_config.log_retention,
            serialize=logging_config.log_serialize_json,
            encoding='utf-8',
            enqueue=True
        )
        
        # 添加控制台日志处理器
        logger.add(
            sink=sys.stderr,
            level=logging_config.log_level.upper(),
            serialize=False,
            colorize=True
        )
        
        logger.info(f"日志系统已初始化，级别：{logging_config.log_level.upper()}，文件路径：{logging_config.log_file_path}")
        
        return logger
    except Exception as e:
        if not isinstance(e, ConfigurationError):
            error_msg = f"日志系统初始化失败: {str(e)}"
            print(error_msg)  # 直接打印，因为logger可能未初始化
            raise ConfigurationError(error_msg, "LOGGER_INIT_ERROR", {"original": str(e)}) from e
        raise


def create_engine(storage: StorageInterface, config: AppConfig, logger: Any) -> Any:
    """
    创建并返回引擎实例。
    
    Args:
        storage (StorageInterface): 存储接口实现
        config (AppConfig): 应用配置
        logger: 日志器实例
        
    Returns:
        KnowledgeDistillerEngine: 引擎实例
        
    Raises:
        ConfigurationError: 当参数无效时
    """
    # 参数验证
    if storage is None:
        error_msg = "存储接口不能为空"
        if logger:
            logger.error(error_msg)
        raise ConfigurationError(error_msg, "STORAGE_EMPTY_ERROR")
    
    if config is None:
        error_msg = "配置不能为空"
        if logger:
            logger.error(error_msg)
        raise ConfigurationError(error_msg, "CONFIG_EMPTY_ERROR")
    
    if logger is None:
        error_msg = "日志器不能为空"
        print(error_msg)  # 直接打印，因为logger为空
        raise ConfigurationError(error_msg, "LOGGER_EMPTY_ERROR")
    
    try:
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
    except Exception as e:
        if not isinstance(e, ConfigurationError):
            error_msg = f"引擎创建失败: {str(e)}"
            logger.exception(error_msg)
            raise ConfigurationError(error_msg, "ENGINE_CREATE_ERROR", {"original": str(e)}) from e
        raise 