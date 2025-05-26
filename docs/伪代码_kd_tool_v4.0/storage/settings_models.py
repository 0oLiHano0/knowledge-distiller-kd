# kd_tool/storage/settings_models.py (v4.6 - StorageSettingsDTO 迁移版)
# -*- coding: utf-8 -*-

"""
=================================================
settings_models.py - 存储层配置模型 (v4.6)
=================================================

**模块功能**:

- **核心职责**: 定义存储服务 (`kd_tool.storage`) 所需的配置设置模型。
- **v4.6 核心变更**:
    - **[架构指令]** `StorageSettingsDTO` 从原 `schemas` 目录迁移至此。
    - **[架构指令]** `StorageSettingsDTO` 是本模块当前唯一的配置模型。

---
"""

# --- Python 标准库及第三方库导入 ---
from typing import Optional, Any # [指令] 必须导入所需类型
from pathlib import Path # [指令] 必须导入 Path

# --- Pydantic 导入 ---
from pydantic import BaseModel, Field, model_validator # [指令] 必须导入 Pydantic 核心


# ==============================================================================
# 存储服务配置 (StorageSettingsDTO)
# ==============================================================================
# [架构师说明]: StorageSettingsDTO 定义所有与数据持久化相关的配置。
#               它之前位于 schemas/settings_models.py。

class StorageSettingsDTO(BaseModel):
    """
    存储服务的配置设置 DTO。
    **规范**: 定义所有与数据持久化相关的配置。
    """
    backend_type: str = Field(
        default="sqlite",
        description="""
        存储后端类型。
        **规范**: 目前主要支持 'sqlite'。未来可扩展至 'memory_debug' 等。
        **编码要求**: StorageFactory 将根据此类型选择具体的 StorageInterface 实现。
        """
    )
    connection_string: Optional[str] = Field(
        default=None,
        description="""
        数据库连接字符串。
        **规范**: 对于 'sqlite'，格式为 'sqlite:///path/to/your/db/file.db'。
        **编码要求**: 如果 backend_type 为 'sqlite'，此字段 **必须** 提供。
        """
    )
    base_directory: Optional[Path] = Field(
        default=None,
        description="""
        文件系统存储的基础目录路径 (如果后端使用文件系统)。
        **规范**: 用于存储可能需要持久化的非数据库文件（如原始文件备份、缓存等）。
        **编码要求**: 具体使用方式由具体的 StorageInterface 实现决定。
        """
    )

    @model_validator(mode='after')
    @classmethod
    def check_consistency(cls, data: Any) -> Any:
        """
        **验证器**: 确保 'sqlite' 后端提供了 'connection_string'。
        **规范**: 必须确保配置的内部一致性。
        """
        # 在 Pydantic v2 中，data 是模型实例本身
        if isinstance(data, cls): # 确保 data 是 StorageSettingsDTO 的实例
            if data.backend_type == "sqlite" and not data.connection_string:
                raise ValueError("对于 'sqlite' 后端类型, 'connection_string' 必须提供。")
        return data

    class Config:
        extra = 'forbid' # **规范**: 禁止 AppConfig 中出现未在此定义的额外字段。
        arbitrary_types_allowed = True # 允许 Path 等非 Pydantic 基本类型。