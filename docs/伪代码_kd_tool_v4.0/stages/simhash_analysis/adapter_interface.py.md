```python

# kd_tool/stages/simhash_analysis/adapter_interface.py
# -*- coding: utf-8 -*-

"""
=================================================
adapter_interface.py - SimHash 适配器接口定义 (v4.5)
=================================================

**模块功能**:

- 定义 SimHash 计算和比较服务的抽象接口。
- 任何具体的 SimHash 实现 (例如，基于 `simhash` 库) 都必须实现此接口。
- **规范**: 接口应关注核心功能，保持最小化和稳定性。

---
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

class SimHashAdapterInterface(ABC):
    """
    SimHash 适配器的抽象基类。
    定义了计算 SimHash 指纹和计算汉明距离的标准方法。
    """

    @abstractmethod
    def calculate_simhash(self, text: str, hash_bits: int) -> str:
        """
        计算给定文本的 SimHash 指纹。

        **参数**:
            text (str): 需要计算 SimHash 的输入文本。
                        **规范**: 接口实现不负责文本预处理，应使用 `analysis_text`。
            hash_bits (int): 期望的 SimHash 位数 (必须是 64 或 128)。

        **返回**:
            str: 计算出的 SimHash 指纹 (十六进制字符串)。
                 **规范**: 64 位返回 16 个字符，128 位返回 32 个字符。

        **可能抛出**:
            SimHashCalculationError: 如果计算过程中发生错误。
        """
        pass

    @abstractmethod
    def calculate_hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        计算两个 SimHash 指纹之间的汉明距离。

        **参数**:
            hash1 (str): 第一个 SimHash 指纹 (十六进制字符串)。
            hash2 (str): 第二个 SimHash 指纹 (十六进制字符串)。
                         **规范**: 两个哈希值的位数必须相同。

        **返回**:
            int: 两个指纹之间的汉明距离。

        **可能抛出**:
            ValueError: 如果两个哈希值的长度（位数）不一致。
            SimHashComparisonError: 如果比较过程中发生其他错误。
        """
        pass

    @abstractmethod
    def find_similar_pairs(
        self,
        block_hashes: List[Tuple[str, str]], 
        threshold: int,
        hash_bits: int
    ) -> List[Tuple[str, str, int]]:
        """
        (可选但推荐) 在一批哈希值中高效地找出汉明距离小于阈值的对。
        这可以避免 O(n^2) 的完全比较，尤其是在数据量大时。
        如果无法高效实现，Stage 也可以自行进行 O(n^2) 比较。

        **参数**:
            block_hashes (List[Tuple[str, str]]): 一个列表，每个元组包含 (block_id, simhash_value)。
            threshold (int): 汉明距离阈值。
            hash_bits (int): 哈希位数。

        **返回**:
            List[Tuple[str, str, int]]: 一个列表，每个元组包含 (block_id_1, block_id_2, hamming_distance)。
        """
        pass

```