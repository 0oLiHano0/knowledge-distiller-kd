# kd_tool/stages/decision/errors.py (v4.6 - 确认导入)
# -*- coding: utf-8 -*-

"""
=================================================
errors.py - P08 决策阶段错误定义 (v4.6)
=================================================

**模块功能**:

- 定义 P08 决策阶段可能抛出的特定异常。
- 所有异常必须继承自 `core.errors.KDToolError`。

---
"""

from ...core.errors import KDToolError

class DecisionError(KDToolError):
    """决策阶段的基础错误类型。"""
    pass

class RuleEvaluationError(DecisionError):
    """当评估决策规则时发生错误时抛出。"""
    def __init__(self, pair_id: str, rule: Any, original_error: Exception):
        self.pair_id = pair_id
        self.rule = str(rule) # 转换为字符串以避免循环引用或复杂性
        self.original_error = original_error
        super().__init__(
            f"为分析对 '{pair_id}' 评估规则 '{self.rule}' 时发生错误: {original_error}"
        )

class MissingAnalysisDataError(DecisionError):
    """当决策所需的分析数据缺失时抛出。"""
    pass