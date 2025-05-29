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
from pydantic import BaseModel, Field, model_validator, ConfigDict
from math import ceil


class SimHashAnalysisStageSettings(BaseModel):
    """P06 - SimHash 分析阶段的配置。"""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = Field(default=True, description='是否启用 P06 - SimHash 分析阶段 (用于近似去重)。')
    hash_bits: int = Field(default=64, description="SimHash 指纹的位数。64 或 128。")
    hamming_distance_ratio: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="汉明距离阈值（标准化为0~1）。实际阈值=ratio*hash_bits，建议0.01~0.2"
    )
    force_recalculate: bool = Field(default=False, description='是否强制重新计算所有块的 SimHash 值。')

    @property
    def hamming_distance_threshold(self) -> int:
        """返回实际汉明距离阈值（向上取整）"""
        return max(0, min(self.hash_bits, int(ceil(self.hamming_distance_ratio * self.hash_bits))))

    @model_validator(mode='after')
    @classmethod
    def check_ratio(cls, data):
        if isinstance(data, cls):
            if not (0.0 <= data.hamming_distance_ratio <= 1.0):
                raise ValueError("hamming_distance_ratio 必须在 0~1 之间")
        return data