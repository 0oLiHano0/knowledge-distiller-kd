"""
=================================================
errors.py - P06 SimHash 分析阶段错误定义 (v4.5)
=================================================

**模块功能**:

- 定义 P06 SimHash 分析阶段可能抛出的特定异常。
- 所有异常必须继承自 `core.errors.KDToolError`。

---
"""
from kd_tool.core.errors import KDToolError


class SimHashAnalysisError(KDToolError):
    """SimHash 分析阶段的基础错误类型。"""
    pass


class SimHashCalculationError(SimHashAnalysisError):
    """当为某个内容块计算 SimHash 值失败时抛出。"""

    def __init__(self, block_id: str, original_error: Exception):
        self.block_id = block_id
        self.original_error = original_error
        super().__init__(
            f"为内容块 '{block_id}' 计算 SimHash 时发生错误: {original_error}")


class SimHashAdapterError(SimHashAnalysisError):
    """当 SimHash 适配器 (例如，与外部库交互) 发生错误时抛出。"""
    pass


class SimHashComparisonError(SimHashAnalysisError):
    """当比较 SimHash 值时发生意外错误时抛出。"""
    pass
