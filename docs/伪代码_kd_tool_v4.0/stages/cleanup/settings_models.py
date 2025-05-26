# kd_tool/stages/cleanup/settings_models.py (v4.6 - Cleanup Settings 迁移版)
# -*- coding: utf-8 -*-

"""
=================================================
settings_models.py - Cleanup Stage 配置模型 (v4.6)
=================================================

**模块功能**:

- **核心职责**: 定义 Cleanup Stage (`kd_tool.stages.cleanup`) 所需的配置设置模型。
- **v4.6 核心变更**:
    - **[架构指令]** `CleanupStageSettings` 从原 `schemas` 目录迁移至此。

---
"""

# --- Python 标准库及第三方库导入 ---
from typing import Optional, Literal, Any, Dict
from pathlib import Path

# --- Pydantic 导入 ---
from pydantic import BaseModel, Field, model_validator

# --- 项目内部模块导入 ---
# [指令] 从中央 schemas 目录导入共享枚举
from ....schemas.enums import DecisionType #


# ==============================================================================
# 清理阶段配置 (CleanupStageSettings)
# ==============================================================================
# [架构师说明]: CleanupStageSettings 定义了 P09 清理阶段的行为。
#               它之前位于 schemas/settings_models.py。

class CleanupStageSettings(BaseModel):
    """
    P09 - 清理阶段的配置。
    **架构师说明**: 此阶段将执行 `DecisionStage` 产生的决策。
                   其配置将涉及具体的文件操作（标记、移动、删除）等。
                   **安全第一**: 默认配置应采用最安全的方式 (mark_only)。
    """
    enabled: bool = Field(
        default=True,
        description="是否启用 P09 - 清理阶段。"
    )

    action_map: Dict[DecisionType, Literal['mark_only', 'move_to_trash', 'permanent_delete', 'ignore']] = Field(
        default_factory=lambda: {
            DecisionType.DELETE: 'mark_only', # **默认**: 对标记为 DELETE 的仅做标记
            DecisionType.KEEP: 'ignore',      # 忽略 KEEP
            DecisionType.UNDECIDED: 'ignore', # 忽略 UNDECIDED
            DecisionType.IGNORE_PAIR: 'ignore' # 忽略 IGNORE_PAIR
        },
        description="""
        决策类型到具体清理动作的映射。
        - 'mark_only': (最安全) 仅更新数据库中 FileRecordDTO 的状态为 'MARKED_FOR_DELETION'。
        - 'move_to_trash': 将物理文件移动到指定的 'trash_directory' 并更新状态。
        - 'permanent_delete': (危险!) **永久删除**物理文件并更新状态 (或删除记录)。
        - 'ignore': 对此决策类型不执行任何操作。
        **规范**: 必须为所有 DecisionType 提供映射 (或有默认处理)。
        """
    )

    trash_directory: Optional[Path] = Field(
        default=None,
        description="""
        垃圾箱目录的路径。
        **规范**: 如果 `action_map` 中有任何值设为 'move_to_trash'，此字段 **必须** 提供且必须是有效目录。
        """
    )

    @model_validator(mode='after')
    @classmethod
    def check_trash_dir_if_needed(cls, data: Any) -> Any:
        """**验证器**: 如果需要移动到垃圾箱，确保垃圾箱目录已提供。"""
        if isinstance(data, cls): # 确保 data 是 CleanupStageSettings 的实例
            needs_trash = any(action == 'move_to_trash' for action in data.action_map.values())
            if needs_trash and not data.trash_directory:
                raise ValueError("如果 'action_map' 中包含 'move_to_trash'，则 'trash_directory' 必须提供。")
            # **编码要求**: 在实际实现中，还应验证 trash_directory 是否存在且可写。
        return data

    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True # 允许 Path 和枚举类型