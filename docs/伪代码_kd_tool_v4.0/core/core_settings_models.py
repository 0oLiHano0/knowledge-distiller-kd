# kd_tool/core/core_settings_models.py (v4.6 - OrchestratorSettings 迁移版)
# -*- coding: utf-8 -*-

"""
=================================================
core_settings_models.py - KD_Tool 核心配置模型 (v4.6)
=================================================

**模块功能**:

- **核心职责**: 定义 `core` 层特有的、关键的配置设置模型。
- **v4.6 核心变更**:
    - **[架构指令]** `OrchestratorSettings` 从原 `schemas` 目录迁移至此。
    - **[架构指令]** `OrchestratorSettings` 是本模块当前唯一的配置模型。

---
"""

# --- Python 标准库及第三方库导入 ---
from typing import List, Literal # [指令] 必须导入所需类型

# --- Pydantic 导入 ---
from pydantic import BaseModel, Field # [指令] 必须导入 Pydantic 核心


# ==============================================================================
# Orchestrator 配置 (OrchestratorSettings)
# ==============================================================================
# [架构师说明]: OrchestratorSettings 定义了流水线调度和执行行为的参数。
#               它之前位于 schemas/settings_models.py。

class OrchestratorSettings(BaseModel):
    """
    Orchestrator 模块的配置设置。
    **规范**: 定义流水线调度和执行行为的参数。
    """
    on_pipeline_error_policy: Literal['HALT_ON_FIRST_ERROR', 'CONTINUE_IGNORING_ERROR'] = Field(
        default='HALT_ON_FIRST_ERROR',
        description="""
        流水线错误处理策略。
        - HALT_ON_FIRST_ERROR: 遇到第一个 Stage 错误时，立即停止整个流水线。
        - CONTINUE_IGNORING_ERROR: 记录错误并继续执行下一个 Stage。
        **编码要求**: Orchestrator 的 `run` 方法必须根据此策略进行错误处理。
        """
    )
    default_task_id_prefix: str = Field(
        default='kd_task_',
        description="""
        为生成的 task_id 添加的可选前缀。
        **规范**: 主要用于日志追踪和调试。
        **编码要求**: 在 Orchestrator 创建 PipelineContextDTO 时使用。
        """
    )
    default_stage_order: List[str] = Field(
        default=[
            "prefilter",
            "document_processing",
            "block_merging",
            "md5_analysis",
            "simhash_analysis",
            "semantic_analysis",
            "decision",
            "cleanup"
        ],
        description="""
        默认情况下流水线中各个阶段的执行顺序和名称。
        **规范**: 这里的名称 **必须** 与 `ApplicationBuilder` 中注册 Stage 时使用的键名一致。
        **编码要求**: Orchestrator 将按此列表顺序执行 Stage。
        """
    )

    class Config:
        extra = 'forbid' # **规范**: 禁止未知字段。