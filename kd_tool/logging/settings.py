"""
日志配置模块
-----------
提供统一的日志配置接口。
"""

from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
import os

from pydantic import BaseModel, Field

__all__ = ["LogLevel", "LoggingConfigDTO"]


class LogLevel(str, Enum):
    """日志级别枚举。"""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ConsoleConfigDTO(BaseModel):
    """控制台输出配置。"""
    enabled: bool = Field(default=True, description="是否启用控制台输出")
    colorize: bool = Field(default=True, description="是否启用颜色")
    backtrace: bool = Field(default=True, description="是否显示堆栈跟踪")
    diagnose: bool = Field(default=True, description="是否显示诊断信息")
    enqueue: bool = Field(default=False, description="是否启用异步队列")
    catch: bool = Field(default=True, description="是否捕获异常")


class FileConfigDTO(BaseModel):
    """文件输出配置。"""
    enabled: bool = Field(default=False, description="是否启用文件输出")
    path: Path = Field(default=Path("logs/app.log"), description="日志文件路径")
    rotation: str = Field(default="00:00", description="轮转时间，例如：00:00 表示每天午夜")
    retention: str = Field(default="10 days", description="保留时间")
    compression: str = Field(default="zip", description="压缩格式")
    format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        description="文件日志格式"
    )
    serialize: bool = Field(default=True, description="是否序列化为 JSON")


class LoggingConfigDTO(BaseModel):
    """日志配置。"""
    level: LogLevel = Field(default=LogLevel.INFO, description="日志级别")
    format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        description="日志格式"
    )
    serialize: bool = Field(default=False, description="是否序列化日志")
    console: ConsoleConfigDTO = Field(default_factory=ConsoleConfigDTO, description="控制台输出配置")
    file: FileConfigDTO = Field(default_factory=FileConfigDTO, description="文件输出配置")

    @classmethod
    def from_env(cls) -> "LoggingConfigDTO":
        """从环境变量加载配置。"""
        config = cls()
        
        # 从环境变量覆盖配置
        if "KD_LOGGING_LEVEL" in os.environ:
            config.level = LogLevel(os.environ["KD_LOGGING_LEVEL"])
        
        if "KD_LOGGING_FORMAT" in os.environ:
            config.format = os.environ["KD_LOGGING_FORMAT"]
        
        if "KD_LOGGING_SERIALIZE" in os.environ:
            config.serialize = os.environ["KD_LOGGING_SERIALIZE"].lower() == "true"
        
        # 控制台配置
        if "KD_LOGGING_CONSOLE_ENABLED" in os.environ:
            config.console.enabled = os.environ["KD_LOGGING_CONSOLE_ENABLED"].lower() == "true"
        if "KD_LOGGING_CONSOLE_COLORIZE" in os.environ:
            config.console.colorize = os.environ["KD_LOGGING_CONSOLE_COLORIZE"].lower() == "true"
        if "KD_LOGGING_CONSOLE_BACKTRACE" in os.environ:
            config.console.backtrace = os.environ["KD_LOGGING_CONSOLE_BACKTRACE"].lower() == "true"
        if "KD_LOGGING_CONSOLE_DIAGNOSE" in os.environ:
            config.console.diagnose = os.environ["KD_LOGGING_CONSOLE_DIAGNOSE"].lower() == "true"
        if "KD_LOGGING_CONSOLE_ENQUEUE" in os.environ:
            config.console.enqueue = os.environ["KD_LOGGING_CONSOLE_ENQUEUE"].lower() == "true"
        if "KD_LOGGING_CONSOLE_CATCH" in os.environ:
            config.console.catch = os.environ["KD_LOGGING_CONSOLE_CATCH"].lower() == "true"
        
        # 文件配置
        if "KD_LOGGING_FILE_ENABLED" in os.environ:
            config.file.enabled = os.environ["KD_LOGGING_FILE_ENABLED"].lower() == "true"
        if "KD_LOGGING_FILE_PATH" in os.environ:
            config.file.path = Path(os.environ["KD_LOGGING_FILE_PATH"])
        if "KD_LOGGING_FILE_ROTATION" in os.environ:
            config.file.rotation = os.environ["KD_LOGGING_FILE_ROTATION"]
        if "KD_LOGGING_FILE_RETENTION" in os.environ:
            config.file.retention = os.environ["KD_LOGGING_FILE_RETENTION"]
        if "KD_LOGGING_FILE_COMPRESSION" in os.environ:
            config.file.compression = os.environ["KD_LOGGING_FILE_COMPRESSION"]
        if "KD_LOGGING_FILE_FORMAT" in os.environ:
            config.file.format = os.environ["KD_LOGGING_FILE_FORMAT"]
        if "KD_LOGGING_FILE_SERIALIZE" in os.environ:
            config.file.serialize = os.environ["KD_LOGGING_FILE_SERIALIZE"].lower() == "true"
        
        return config

    @classmethod
    def default(cls) -> "LoggingConfigDTO":
        """获取默认配置。"""
        return cls()
