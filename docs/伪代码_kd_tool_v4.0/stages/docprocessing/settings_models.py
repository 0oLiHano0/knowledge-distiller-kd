# kd_tool/stages/docprocessing/settings_models.py (v4.6 - DocProcessing Settings 迁移版)
# -*- coding: utf-8 -*-

"""
=================================================
settings_models.py - DocumentProcessing Stage 配置模型 (v4.6)
=================================================

**模块功能**:

- **核心职责**: 定义 DocumentProcessing Stage (`kd_tool.stages.docprocessing`) 所需的配置设置模型。
- **v4.6 核心变更**:
    - **[架构指令]** `DocumentProcessingStageSettings` 从原 `schemas` 目录迁移至此。

---
"""

# --- Python 标准库及第三方库导入 ---
from typing import List, Literal

# --- Pydantic 导入 ---
from pydantic import BaseModel, Field


# ==============================================================================
# 文档处理阶段配置 (DocumentProcessingStageSettings)
# ==============================================================================
# [架构师说明]: DocumentProcessingStageSettings 定义了 P03 文档处理阶段（原始提取）的行为。
#               它之前位于 schemas/settings_models.py。

class DocumentProcessingStageSettings(BaseModel):
    """P03 - 文档处理阶段 (原始提取) 的配置模型。"""
    enabled: bool = Field(
        default=True,
        description="是否启用 P03 - 文档处理阶段。"
    )
    parsing_strategy: Literal['auto', 'fast', 'hi_res'] = Field(
        default='auto',
        description="底层解析库 (如 `unstructured`) 使用的解析策略。"
    )
    supported_extensions: List[str] = Field(
        default=[".md", ".txt", ".docx", ".pdf"],
        description="此阶段尝试处理的文件扩展名列表。"
    )
    class Config:
        extra = 'forbid'