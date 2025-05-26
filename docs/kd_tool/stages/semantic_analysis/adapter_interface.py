"""
=================================================
adapter_interface.py - 语义分析适配器接口定义 (v4.5)
=================================================

**模块功能**:

- 定义语义分析服务的抽象接口。
- 任何具体的语义模型实现 (例如，基于 `sentence-transformers`) 都必须实现此接口。
- **规范**: 接口应关注核心功能：加载模型、计算嵌入、计算相似度。

---
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np


class SemanticAdapterInterface(ABC):
    """
    语义分析适配器的抽象基类。
    """

    @abstractmethod
    def load_model(self, model_name_or_path: str, device: Optional[str]=None
        ) ->None:
        """
        加载指定的语义模型到指定的设备。
        **规范**: 
        - 如果模型已加载，可以跳过或重新加载（根据实现）。
        - 必须处理模型加载失败的情况，并抛出 `ModelLoadingError`。
        
        **参数**:
            model_name_or_path (str): 模型名称或路径。
            device (Optional[str]): 运行设备 (e.g., 'cpu', 'cuda')。
        """
        pass

    @abstractmethod
    def calculate_embeddings(self, texts: List[str], batch_size: int
        ) ->np.ndarray:
        """
        为一批文本计算语义嵌入向量。
        
        **参数**:
            texts (List[str]): 需要计算嵌入的文本列表。
            batch_size (int): 批处理大小。
            
        **返回**:
            np.ndarray: 一个 2D numpy 数组，形状为 (len(texts), embedding_dim)，
                        每一行代表一个文本的嵌入向量。
                        
        **可能抛出**:
            EmbeddingCalculationError: 如果计算过程中发生错误。
            SemanticAdapterError: 如果模型未加载。
        """
        pass

    @abstractmethod
    def calculate_similarity_matrix(self, embeddings: np.ndarray) ->np.ndarray:
        """
        计算一批嵌入向量之间的余弦相似度矩阵。
        
        **参数**:
            embeddings (np.ndarray): 形状为 (n, embedding_dim) 的嵌入向量数组。
            
        **返回**:
            np.ndarray: 一个 2D numpy 数组，形状为 (n, n)，
                        其中 `matrix[i, j]` 是第 i 个向量和第 j 个向量的余弦相似度。
                        **规范**: 矩阵应该是对称的，对角线为 1.0。
                        
        **可能抛出**:
            SimilarityCalculationError: 如果计算过程中发生错误。
        """
        pass

    @abstractmethod
    def calculate_pair_similarity(self, embedding1: np.ndarray, embedding2:
        np.ndarray) ->float:
        """
        计算两个嵌入向量之间的余弦相似度。
        
        **参数**:
            embedding1 (np.ndarray): 第一个嵌入向量 (1D 数组)。
            embedding2 (np.ndarray): 第二个嵌入向量 (1D 数组)。
            
        **返回**:
            float: 两个向量之间的余弦相似度 (0.0 到 1.0)。
            
        **可能抛出**:
            SimilarityCalculationError: 如果计算过程中发生错误。
        """
        pass
