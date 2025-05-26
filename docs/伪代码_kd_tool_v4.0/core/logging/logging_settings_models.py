# kd_tool/core/logging/logging_settings_models.py (v4.6 - LoggingSettings 迁移版)
# -*- coding: utf-8 -*-

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

# --- Python 标准库及第三方库导入 ---
from typing import Optional, Literal, Any # [指令] 必须导入所需类型
from pathlib import Path # [指令] 必须导入 Path

# --- Pydantic 导入 ---
from pydantic import BaseModel, Field, model_validator # [指令] 必须导入 Pydantic 核心

# ==============================================================================
# 日志系统配置 (LoggingSettings)
# ==============================================================================
# [架构师说明]: LoggingSettings 定义了 Loguru 日志系统的所有行为参数。
#               它之前位于 schemas/settings_models.py。

class LoggingSettings(BaseModel):
    """
    日志系统的配置。
    **规范**: 定义 Loguru 日志系统的所有行为参数。
    """
    log_level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="全局日志级别。"
    )
    log_to_console: bool = Field(
        default=True,
        description="是否将日志输出到控制台。"
    )
    log_to_file: bool = Field(
        default=False,
        description="是否将日志输出到文件。"
    )
    log_file_path: Optional[Path] = Field(
        default=None,
        description="日志文件路径。**规范**: 如果 log_to_file 为 True，此项必填。"
    )
    log_rotation: Optional[str] = Field(
        default="10 MB",
        description="日志文件轮换策略 (Loguru 格式, e.g., '10 MB', '1 week', '00:00')。"
    )
    log_format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Loguru 日志格式字符串。"
    )
    enqueue: bool = Field(
        default=True,
        description="是否启用异步日志记录，以提高性能。"
    )
    serialize: bool = Field(
        default=False,
        description="是否将日志消息序列化为 JSON 格式，便于机器处理。"
    )

    @model_validator(mode='after')
    @classmethod
    def check_log_file_path_if_logging_to_file(cls, data: Any) -> Any:
        """
        **验证器**: 确保 'log_to_file' 为 True 时提供了 'log_file_path'。
        """
        # 在 Pydantic v2 中，data 是模型实例本身
        if isinstance(data, cls): # 确保 data 是 LoggingSettings 的实例
            if data.log_to_file and not data.log_file_path:
                raise ValueError("如果 'log_to_file' 为 True, 'log_file_path' 必须提供。")
        return data

    class Config:
        extra = 'forbid' # **规范**: 禁止未知字段。
        arbitrary_types_allowed = True # 允许 Path 等非 Pydantic 基本类型。
        validate_assignment = True # 允许在运行时修改并验证