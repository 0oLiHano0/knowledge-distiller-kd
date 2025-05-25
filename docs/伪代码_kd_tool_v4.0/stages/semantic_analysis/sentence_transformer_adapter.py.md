```python

# kd_tool/stages/semantic_analysis/sentence_transformer_adapter.py
# -*- coding: utf-8 -*-

"""
=================================================
sentence_transformer_adapter.py - 基于 `sentence-transformers` 的适配器实现 (v4.5)
=================================================

**模块功能**:

- 实现了 `SemanticAdapterInterface`，使用 `sentence-transformers` 库提供具体功能。
- **规范**: 必须处理库可能抛出的异常，并转换为我们定义的 `SemanticAnalysisError`。
- **依赖**: 需要安装 `sentence-transformers` 和 `torch` (或 `tensorflow`)。

---
"""

from typing import List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
except ImportError:
    print("警告: `sentence-transformers` 库未安装。SentenceTransformerAdapter 将无法工作。")
    SentenceTransformer = None
    cos_sim = None

from .adapter_interface import SemanticAdapterInterface
from .errors import ModelLoadingError, EmbeddingCalculationError, SimilarityCalculationError, SemanticAdapterError

class SentenceTransformerAdapter(SemanticAdapterInterface):
    """
    使用 `sentence-transformers` 库实现语义分析功能的适配器。
    """

    def __init__(self):
        self._model: Optional[SentenceTransformer] = None

    def load_model(self, model_name_or_path: str, device: Optional[str] = None) -> None:
        """加载模型。"""
        if SentenceTransformer is None:
            raise SemanticAdapterError("`sentence-transformers` 库未安装。")
            
        try:
            self._model = SentenceTransformer(model_name_or_path, device=device)
        except Exception as e:
            raise ModelLoadingError(model_path=model_name_or_path, original_error=e)

    def calculate_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        """计算嵌入向量。"""
        if self._model is None:
            raise SemanticAdapterError("模型尚未加载。请先调用 `load_model`。")

        try:
            # `sentence-transformers` 的 encode 方法支持批量处理和设备选择。
            embeddings = self._model.encode(
                texts, 
                batch_size=batch_size, 
                show_progress_bar=False # 在工具中通常关闭进度条
            )
            return np.array(embeddings) # 确保返回 numpy 数组
        except Exception as e:
            # 此处需要更细致的错误处理，区分是哪个块出错
            raise EmbeddingCalculationError(block_id="<batch>", original_error=e)

    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """计算相似度矩阵。"""
        if cos_sim is None:
            raise SemanticAdapterError("`sentence-transformers.util.cos_sim` 未找到。")
            
        if self._model is None:
            raise SemanticAdapterError("模型尚未加载。")

        try:
            # 使用 sentence-transformers 提供的 cos_sim 计算矩阵
            similarity_matrix = cos_sim(embeddings, embeddings).cpu().numpy()
            return similarity_matrix
        except Exception as e:
            raise SimilarityCalculationError(f"计算相似度矩阵时出错: {e}")

    def calculate_pair_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算单对相似度。"""
        if cos_sim is None:
             raise SemanticAdapterError("`sentence-transformers.util.cos_sim` 未找到。")

        try:
            similarity = cos_sim(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0, 0].item()
            return float(similarity)
        except Exception as e:
             raise SimilarityCalculationError(f"计算单对相似度时出错: {e}")

```