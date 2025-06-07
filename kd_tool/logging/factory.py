"""
日志工厂模块
-----------
提供统一的日志配置接口。
"""

from __future__ import annotations
import sys

from loguru import logger

from kd_tool.logging.settings import LoggingConfigDTO


def configure_logging(use_env: bool = False) -> None:
    """配置日志系统。"""
    try:
        # 加载配置
        config = LoggingConfigDTO.from_env() if use_env else LoggingConfigDTO.default()
        
        # 移除所有现有处理器
        logger.remove()
        
        # 配置控制台输出
        if config.console.enabled:
            logger.add(
                sys.stdout,
                format=config.format,
                level=config.level.value,
                colorize=config.console.colorize,
                backtrace=config.console.backtrace,
                diagnose=config.console.diagnose,
                enqueue=config.console.enqueue,
                catch=config.console.catch,
                serialize=config.serialize
            )
        
        # 配置文件输出
        if config.file.enabled:
            logger.add(
                str(config.file.path),  # 确保路径是字符串
                format=config.file.format,
                level=config.level.value,
                rotation=config.file.rotation,
                retention=config.file.retention,
                compression=config.file.compression,
                serialize=config.file.serialize,
                enqueue=True,
                catch=True
            )
            
    except Exception as exc:
        raise RuntimeError(f"日志配置失败: {str(exc)}") from exc
