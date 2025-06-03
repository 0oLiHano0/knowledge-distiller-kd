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

from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict


class OrchestratorSettings(BaseModel):
    """
    Orchestrator 模块的配置设置。
    **规范**: 定义流水线调度和执行行为的参数。
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    on_pipeline_error_policy: Literal[
        "HALT_ON_FIRST_ERROR", "CONTINUE_IGNORING_ERROR"
    ] = Field(
        default="HALT_ON_FIRST_ERROR",
        description="""
        流水线错误处理策略。
        - HALT_ON_FIRST_ERROR: 遇到第一个 Stage 错误时，立即停止整个流水线。
        - CONTINUE_IGNORING_ERROR: 记录错误并继续执行下一个 Stage。
        **编码要求**: Orchestrator 的 `run` 方法必须根据此策略进行错误处理。
        """,
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
            "cleanup",
        ],
        description="""
        默认情况下流水线中各个阶段的执行顺序和名称。
        **规范**: 这里的名称 **必须** 与 `ApplicationBuilder` 中注册 Stage 时使用的键名一致。
        **编码要求**: Orchestrator 将按此列表顺序执行 Stage。
        """,
    )
