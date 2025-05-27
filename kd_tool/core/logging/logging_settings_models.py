"""
=================================================
logging_settings_models.py - 日志配置模型 (v4.6)
=================================================

**模块功能**:

- **核心职责**: 定义日志系统 (`kd_tool.core.logging_setup`) 所需的配置设置模型。
- **v4.6 核心变更**:
    - **[架构指令]** `LoggingSettings` 从原 `schemas` 目录迁移至此 (`kd_tool/core/logging/`)。
    - **[架构指令]** `LoggingSettings` 是本模块当前唯一的配置模型。

---
"""
from typing import Optional, Literal, Any
from pathlib import Path
from pydantic import BaseModel, Field, model_validator


class LoggingSettings(BaseModel):
    """
    日志系统的配置。
    **规范**: 定义 Loguru 日志系统的所有行为参数。
    """
    log_level: Literal['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING',
        'ERROR', 'CRITICAL'] = Field(default='INFO', description='全局日志级别。')
    log_to_console: bool = Field(default=True, description='是否将日志输出到控制台。')
    log_to_file: bool = Field(default=False, description='是否将日志输出到文件。')
    log_file_path: Optional[Path] = Field(default=None, description=
        '日志文件路径。**规范**: 如果 log_to_file 为 True，此项必填。')
    log_rotation: Optional[str] = Field(default='10 MB', description=
        "日志文件轮换策略 (Loguru 格式, e.g., '10 MB', '1 week', '00:00')。")
    log_format: str = Field(default=
        '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
        , description='Loguru 日志格式字符串。')
    enqueue: bool = Field(default=True, description='是否启用异步日志记录，以提高性能。')
    serialize: bool = Field(default=False, description=
        '是否将日志消息序列化为 JSON 格式，便于机器处理。')

    @model_validator(mode='after')
    @classmethod
    def check_log_file_path_if_logging_to_file(cls, data: Any) ->Any:
        """
        **验证器**: 确保 'log_to_file' 为 True 时提供了 'log_file_path'。
        """
        if isinstance(data, cls):
            if data.log_to_file and not data.log_file_path:
                raise ValueError(
                    "如果 'log_to_file' 为 True, 'log_file_path' 必须提供。")
        return data


    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True
        validate_assignment = True
