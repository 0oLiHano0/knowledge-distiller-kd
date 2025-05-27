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
from pydantic import BaseModel, Field


class MD5AnalysisStageSettings(BaseModel):
    """P05 - MD5 分析阶段的配置。"""
    enabled: bool = Field(default=True, description=
        '是否启用 P05 - MD5 分析阶段 (用于精确去重)。')


    class Config:
        extra = 'forbid'
