"""
=================================================
kd_tool/logging/settings.py - 日志设置数据传输对象 (v4.1)
=================================================

**模块功能**:

- **核心职责**: 定义 `LoggingSettingsDTO`，作为日志配置的载体。
- LoggingSettingsDTO (Pydantic)
---
"""

# kd_tool/logging/settings.py
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path


class LoggingSettingsDTO(BaseModel):
    """
    WHY : 统一管理日志配置
    WHAT: 提供级别、格式、文件等字段
    HOW : 继承 Pydantic BaseModel 保证类型安全
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    level: str = Field(
        "INFO", pattern="^(TRACE|DEBUG|INFO|SUCCESS|WARNING|ERROR|CRITICAL)$"
    )
    log_serialize_json: bool = False
    log_file: Path | None = None
    rotation: str = "00:00"  # 每天切
    retention: str = "10 days"  # 保留10天
