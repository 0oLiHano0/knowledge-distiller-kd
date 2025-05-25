```python

# kd_tool/stages/semantic_analysis/errors.py
# -*- coding: utf-8 -*-

"""
=================================================
errors.py - P07 语义分析阶段错误定义 (v4.5)
=================================================

**模块功能**:

- 定义 P07 语义分析阶段可能抛出的特定异常。
- 所有异常必须继承自 `core.errors.KDToolError`。

---
"""

from ...core.errors import KDToolError

class SemanticAnalysisError(KDToolError):
    """语义分析阶段的基础错误类型。"""
    pass

class ModelLoadingError(SemanticAnalysisError):
    """当加载语义模型失败时抛出。"""
    def __init__(self, model_path: str, original_error: Exception):
        self.model_path = model_path
        self.original_error = original_error
        super().__init__(
            f"加载语义模型 '{model_path}' 时发生错误: {original_error}"
        )

class EmbeddingCalculationError(SemanticAnalysisError):
    """当为文本计算嵌入向量失败时抛出。"""
    def __init__(self, block_id: str, original_error: Exception):
        self.block_id = block_id
        self.original_error = original_error
        super().__init__(
            f"为内容块 '{block_id}' 计算语义嵌入时发生错误: {original_error}"
        )

class SimilarityCalculationError(SemanticAnalysisError):
    """当计算向量间相似度失败时抛出。"""
    pass

class SemanticAdapterError(SemanticAnalysisError):
    """当语义分析适配器发生错误时抛出。"""
    pass

```