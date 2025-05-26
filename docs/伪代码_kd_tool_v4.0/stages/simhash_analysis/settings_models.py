# kd_tool/stages/simhash_analysis/settings_models.py (v4.6 - SimHash Settings 迁移版)
# -*- coding: utf-8 -*-

"""
=================================================
settings_models.py - SimHashAnalysis Stage 配置模型 (v4.6)
=================================================

**模块功能**:

- **核心职责**: 定义 SimHashAnalysis Stage (`kd_tool.stages.simhash_analysis`) 
              所需的配置设置模型。
- **v4.6 核心变更**:
    - **[架构指令]** `SimHashAnalysisStageSettings` 从原 `schemas` 目录迁移至此。

---
"""

# --- Python 标准库及第三方库导入 ---
from typing import Literal, Any # [指令] 导入所需类型

# --- Pydantic 导入 ---
from pydantic import BaseModel, Field, model_validator, conint # [指令] 导入 Pydantic 核心


# ==============================================================================
# SimHash 分析阶段配置 (SimHashAnalysisStageSettings)
# ==============================================================================
# [架构师说明]: SimHashAnalysisStageSettings 定义了 P06 SimHash 分析阶段的行为。
#               它之前位于 schemas/settings_models.py。

class SimHashAnalysisStageSettings(BaseModel):
    """P06 - SimHash 分析阶段的配置。"""
    enabled: bool = Field(
        default=True,
        description="是否启用 P06 - SimHash 分析阶段 (用于近似去重)。"
    )
    hash_bits: Literal[64, 128] = Field(
        default=64,
        description="""
        SimHash 指纹的位数。
        **规范**: 必须是 64 或 128。64 位速度更快，128 位精度更高。
        **编码要求**: SimHash 适配器和 Stage 必须处理此配置。
        """
    )
    hamming_distance_threshold: conint(ge=0, le=128) = Field( # type: ignore
        default=3,
        description="""
        SimHash 汉明距离阈值。
        **规范**: 两个块的汉明距离 <= 此阈值时，被视为相似。
                 取值范围必须在 [0, hash_bits] 之间。
        **编码要求**: Stage 必须使用此阈值过滤比较结果。
        """
    )
    force_recalculate: bool = Field(
        default=False,
        description="是否强制重新计算所有块的 SimHash 值，即使它们已存在。**规范**: 用于调试或策略变更。"
    )

    @model_validator(mode='after')
    @classmethod
    def check_threshold_within_bits(cls, data: Any) -> Any:
        """**验证器**: 确保汉明距离阈值不超过哈希位数。"""
        if isinstance(data, cls): # 确保 data 是 SimHashAnalysisStageSettings 的实例
            if data.hamming_distance_threshold > data.hash_bits:
                raise ValueError(
                    f"汉明距离阈值 ({data.hamming_distance_threshold}) "
                    f"不能大于哈希位数 ({data.hash_bits})。"
                )
        return data

    class Config:
        extra = 'forbid'