"""
=================================================
settings_models.py - SemanticAnalysis Stage 配置模型 (v4.6)
=================================================

**模块功能**:

- **核心职责**: 定义 SemanticAnalysis Stage (`kd_tool.stages.semantic_analysis`)
              所需的配置设置模型。
- **v4.6 核心变更**:
    - **[架构指令]** `SemanticAnalysisStageSettings` 从原 `schemas` 目录迁移至此。

---
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, PositiveInt, ConfigDict


class SemanticAnalysisStageSettings(BaseModel):
    """
    P07 - 语义分析阶段的配置。
    **规范**: 定义语义分析模型、阈值和执行参数。
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    enabled: bool = Field(
        default=True, description="是否启用 P07 - 语义分析阶段 (用于语义去重)。"
    )
    model_name_or_path: str = Field(
        default="shibing624/text2vec-base-chinese",
        description="""
        语义分析模型名称 (来自 Hugging Face 等) 或本地路径。
        **规范**: 需要选择适合中文且性能可接受的模型。适配器将使用此路径加载模型。
        """,
    )
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="""
        语义相似度得分阈值。
        **规范**: [0.0, 1.0] 范围。两个块的余弦相似度 >= 此阈值时，被视为相似。
        """,
    )
    batch_size: PositiveInt = Field(
        default=32,
        description="""
        向量嵌入批处理大小。
        **规范**: 影响性能和显存占用。适配器应支持按此批次大小处理文本。
        """,
    )
    device: Optional[str] = Field(
        default=None,
        description="""
        运行模型设备 (e.g., 'cpu', 'cuda', 'cuda:0')。
        **规范**: 如果为 None，库通常会自动选择。适配器应能处理此参数。
        """,
    )
    comparison_strategy: Literal["all_pairs", "pre_filtered"] = Field(
        default="pre_filtered",
        description="""
        比较策略。
        - 'all_pairs': 比较所有内容块 (计算成本极高)。
        - 'pre_filtered': (推荐) 仅比较那些未被 MD5 或 SimHash 识别为完全/高度相似的块对。
        **编码要求**: Stage 必须根据此策略决定哪些块对需要进行语义比较。
        """,
    )
