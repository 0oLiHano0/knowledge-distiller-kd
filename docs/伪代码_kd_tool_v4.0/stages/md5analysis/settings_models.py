# kd_tool/stages/md5analysis/settings_models.py (v4.6 - MD5Analysis Settings 迁移版)
# -*- coding: utf-8 -*-

"""
=================================================
settings_models.py - MD5Analysis Stage 配置模型 (v4.6)
=================================================

**模块功能**:

- **核心职责**: 定义 MD5Analysis Stage (`kd_tool.stages.md5analysis`) 所需的配置设置模型。
- **v4.6 核心变更**:
    - **[架构指令]** `MD5AnalysisStageSettings` 从原 `schemas` 目录迁移至此。

---
"""

# --- Pydantic 导入 ---
from pydantic import BaseModel, Field


# ==============================================================================
# MD5 分析阶段配置 (MD5AnalysisStageSettings)
# ==============================================================================
# [架构师说明]: MD5AnalysisStageSettings 定义了 P05 MD5 分析阶段的行为。
#               它之前位于 schemas/settings_models.py。

class MD5AnalysisStageSettings(BaseModel):
    """P05 - MD5 分析阶段的配置。"""
    enabled: bool = Field(
        default=True,
        description="是否启用 P05 - MD5 分析阶段 (用于精确去重)。"
    )
    # [指令] coding 阶段：如果 MD5 分析阶段未来需要更多配置项（例如，是否强制重新计算MD5），
    #       可以在此处添加。目前保持简单。

    class Config:
        extra = 'forbid'