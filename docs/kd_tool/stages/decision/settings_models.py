"""
=================================================
settings_models.py - Decision Stage 配置模型 (v4.6)
=================================================

**模块功能**:

- **核心职责**: 定义 Decision Stage (`kd_tool.stages.decision`) 所需的配置设置模型。
- **v4.6 核心变更**:
    - **[架构指令]** `DecisionRule` 和 `DecisionStageSettings` 从原 `schemas` 目录迁移至此。

---
"""
from typing import List, Optional, Any
from pydantic import BaseModel, Field, confloat
from ....schemas.enums import DecisionType


class DecisionRule(BaseModel):
    """
    定义一条决策规则。
    **规范**: 用于描述当满足某些分析条件时，应采取何种决策。
    """
    md5_score: Optional[confloat(ge=0.0, le=1.0)] = Field(default=None,
        description='触发此规则的 MD5 分数 (通常是 1.0)。如果为 None，则不考虑 MD5。')
    simhash_similarity_min: Optional[confloat(ge=0.0, le=1.0)] = Field(default
        =None, description='触发此规则的 SimHash 最小相似度。如果为 None，则不考虑。')
    semantic_similarity_min: Optional[confloat(ge=0.0, le=1.0)] = Field(default
        =None, description='触发此规则的语义最小相似度。如果为 None，则不考虑。')
    decision_to_apply: DecisionType = Field(..., description=
        '当满足以上所有（非 None）条件时，要应用的决策。')
    rule_priority: int = Field(default=0, description=
        '规则优先级。**规范**: 数字越大，优先级越高。用于处理一个块对可能匹配多条规则的情况。')


    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True


class DecisionStageSettings(BaseModel):
    """
    P08 - 决策阶段的配置。
    """
    enabled: bool = Field(default=True, description='是否启用 P08 - 决策阶段。')
    rules: List[DecisionRule] = Field(default_factory=lambda : [
        DecisionRule(md5_score=1.0, decision_to_apply=DecisionType.DELETE,
        rule_priority=100), DecisionRule(semantic_similarity_min=0.95,
        decision_to_apply=DecisionType.DELETE, rule_priority=90),
        DecisionRule(simhash_similarity_min=0.97, decision_to_apply=
        DecisionType.UNDECIDED, rule_priority=80), DecisionRule(
        semantic_similarity_min=0.85, decision_to_apply=DecisionType.
        UNDECIDED, rule_priority=70)], description=
        """
        决策规则列表。
        **规范**: DecisionStage 将按优先级从高到低评估这些规则。
                 对于每一对分析结果，将应用第一个匹配的、优先级最高的规则。
        """
        )
    default_decision: DecisionType = Field(default=DecisionType.KEEP,
        description=
        """
        如果没有任何规则匹配分析结果，则应用的默认决策。
        **规范**: 通常设置为 'KEEP' 或 'UNDECIDED'。
        """
        )
    process_undecided: bool = Field(default=False, description=
        "是否为 'UNDECIDED' 的结果创建 UserDecisionDTO。**规范**: 如果为 False，则只有明确的决策会被记录。"
        )


    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True
