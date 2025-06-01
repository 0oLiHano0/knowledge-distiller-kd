"""
=================================================
sentence_transformer_adapter.py - 基于 `sentence-transformers` 的适配器实现 (v4.6)
=================================================

**模块功能**:

- 实现了 `SemanticAdapterInterface`，使用 `sentence-transformers` 库提供具体功能。
- **规范**: 必须处理库可能抛出的异常，并转换为我们定义的 `SemanticAnalysisError`。
- **依赖**: 需要安装 `sentence-transformers` 和 `torch` (或 `tensorflow`)。

---
"""
from typing import List, Optional, Any # 实际运行时可能不需要Any
import numpy as np
import threading
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
except ImportError:
    print('警告: `sentence-transformers` 库未安装。SentenceTransformerAdapter 将无法工作。')
    SentenceTransformer = None
    cos_sim = None
from kd_tool.stages.semantic_analysis.adapter_interface import SemanticAdapterInterface
from kd_tool.stages.semantic_analysis.errors import ModelLoadingError, EmbeddingCalculationError, SimilarityCalculationError, SemanticAdapterError


class SentenceTransformerAdapter(SemanticAdapterInterface):
    """
    使用 `sentence-transformers` 库实现语义分析功能的适配器。
    """

    def __init__(self, model_name, device):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model_loaded(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = SentenceTransformer(self._model_name, device=self._device)

    def calculate_embeddings(self, texts, batch_size):
        self._ensure_model_loaded()
        return self._model.encode(texts, batch_size=batch_size)

    def calculate_pair_similarity(self, emb1, emb2):
        # 这里假设emb1和emb2是numpy数组
        from numpy import dot
        from numpy.linalg import norm
        return float(dot(emb1, emb2) / (norm(emb1) * norm(emb2)))

    def calculate_similarity_matrix(self, embeddings: np.ndarray) ->np.ndarray:
        """计算相似度矩阵。"""
        if cos_sim is None:
            raise SemanticAdapterError(
                '`sentence-transformers.util.cos_sim` 未找到。')
        if self._model is None:
            raise SemanticAdapterError('模型尚未加载。')
        try:
            similarity_matrix = cos_sim(embeddings, embeddings).cpu().numpy()
            return similarity_matrix
        except Exception as e:
            raise SimilarityCalculationError(f'计算相似度矩阵时出错: {e}')

    def calculate_pair_similarity(self, embedding1: np.ndarray, embedding2:
        np.ndarray) ->float:
        """计算单对相似度。"""
        if cos_sim is None:
            raise SemanticAdapterError(
                '`sentence-transformers.util.cos_sim` 未找到。')
        try:
            similarity = cos_sim(embedding1.reshape(1, -1), embedding2.
                reshape(1, -1))[0, 0].item()
            return float(similarity)
        except Exception as e:
            raise SimilarityCalculationError(f'计算单对相似度时出错: {e}')
