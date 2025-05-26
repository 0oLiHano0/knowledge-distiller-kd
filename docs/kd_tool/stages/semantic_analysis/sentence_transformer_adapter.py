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
from typing import List, Optional
import numpy as np
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

    def __init__(self):
        self._model: Optional[SentenceTransformer] = None

    def load_model(self, model_name_or_path: str, device: Optional[str]=None
        ) ->None:
        """加载模型。"""
        if SentenceTransformer is None:
            raise SemanticAdapterError('`sentence-transformers` 库未安装。')
        try:
            self._model = SentenceTransformer(model_name_or_path, device=device
                )
        except Exception as e:
            raise ModelLoadingError(model_path=model_name_or_path,
                original_error=e)

    def calculate_embeddings(self, texts: List[str], batch_size: int
        ) ->np.ndarray:
        """计算嵌入向量。"""
        if self._model is None:
            raise SemanticAdapterError('模型尚未加载。请先调用 `load_model`。')
        try:
            embeddings = self._model.encode(texts, batch_size=batch_size,
                show_progress_bar=False)
            return np.array(embeddings)
        except Exception as e:
            raise EmbeddingCalculationError(block_id='<batch>',
                original_error=e)

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
