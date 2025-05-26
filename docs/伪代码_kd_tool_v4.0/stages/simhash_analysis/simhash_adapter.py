# kd_tool/stages/simhash_analysis/simhash_adapter.py (v4.6 - Schema 路径更新版)
# -*- coding: utf-8 -*-

"""
=================================================
simhash_adapter.py - 基于 `simhash` 库的适配器实现 (v4.6)
=================================================

**模块功能**:

- 实现了 `SimHashAdapterInterface`，使用 `simhash` 库提供具体功能。
- **规范**: 必须处理 `simhash` 库可能抛出的异常，并转换为我们定义的 `SimHashAnalysisError`。
- **依赖**: 需要安装 `simhash` 库。

---
"""

from typing import List, Tuple
# 伪代码: 假设我们有 simhash 库
try:
    from simhash import Simhash, SimhashIndex
except ImportError:
    # 在实际项目中，这里应该有更健壮的处理或安装提示
    print("警告: `simhash` 库未安装。SimHashAdapter 将无法工作。")
    Simhash = None
    SimhashIndex = None

from .adapter_interface import SimHashAdapterInterface
from .errors import SimHashCalculationError, SimHashComparisonError

class SimhashLibAdapter(SimHashAdapterInterface):
    """
    使用 `simhash` 库实现 SimHash 功能的适配器。
    """

    def calculate_simhash(self, text: str, hash_bits: int) -> str:
        """计算 SimHash 指纹。"""
        if Simhash is None:
            raise SimHashAdapterError("`simhash` 库未安装。")
        if hash_bits not in [64, 128]:
            raise ValueError(f"不支持的哈希位数: {hash_bits}。只支持 64 或 128。")

        try:
            # `simhash` 库默认使用 64 位。如果需要 128，需要检查库是否支持或调整。
            # 此处伪代码假设 Simhash 构造函数或 get_simhash 方法能处理位数，
            # 或者我们在这里进行适配。为简单起见，假设它默认 64 位。
            # 实际实现时需要仔细核对 `simhash` 库的行为。
            # 【注意】`simhash` 库可能需要对中文进行特定处理（如 jieba 分词）。
            #       此处伪代码简化了这一步，但在实际实现中至关重要。
            
            # 假设 Simhash 需要分词后的词列表
            # words = self._tokenize(text) # <-- 需要一个分词方法
            # simhash_obj = Simhash(words, f=hash_bits) 
            
            # 如果 Simhash 直接接受字符串（并内置分词或基于字符）
            simhash_obj = Simhash(text, f=hash_bits)

            # 将 Simhash 对象的 value (通常是 int) 转换为十六进制字符串
            # 确保填充到正确的长度 (64位 -> 16 Hex, 128位 -> 32 Hex)
            hex_format = f'0{hash_bits // 4}x' # e.g., '016x' for 64 bits
            return format(simhash_obj.value, hex_format)

        except Exception as e:
            # 捕获 `simhash` 库可能抛出的任何异常
            raise SimHashCalculationError(block_id="<unknown>", original_error=e)

    def calculate_hamming_distance(self, hash1_hex: str, hash2_hex: str) -> int:
        """计算汉明距离。"""
        if Simhash is None:
            raise SimHashAdapterError("`simhash` 库未安装。")
        
        if len(hash1_hex) != len(hash2_hex):
            raise ValueError("SimHash 指纹长度不一致，无法比较。")

        try:
            # 将十六进制字符串转换回整数
            hash1_int = int(hash1_hex, 16)
            hash2_int = int(hash2_hex, 16)

            # 创建 Simhash 对象（或直接使用其 distance 方法，如果支持）
            # Simhash 对象的构造可能需要位数，从长度推断
            bits = len(hash1_hex) * 4
            simhash1 = Simhash(hash1_int, f=bits)
            simhash2 = Simhash(hash2_int, f=bits)

            return simhash1.distance(simhash2)

        except Exception as e:
            raise SimHashComparisonError(f"比较 {hash1_hex} 和 {hash2_hex} 时出错: {e}")

    def find_similar_pairs(
        self,
        block_hashes: List[Tuple[str, str]],
        threshold: int,
        hash_bits: int
    ) -> List[Tuple[str, str, int]]:
        """使用 SimhashIndex 高效查找相似对。"""
        if SimhashIndex is None:
             raise SimHashAdapterError("`simhash` 库的 SimhashIndex 未找到或未安装。")

        if not block_hashes:
            return []

        try:
            # 创建 SimhashIndex
            # `k` 参数是汉明距离阈值
            index = SimhashIndex([], f=hash_bits, k=threshold)

            # 向索引中添加对象
            for block_id, hash_value in block_hashes:
                simhash_obj = Simhash(int(hash_value, 16), f=hash_bits)
                index.add(block_id, simhash_obj)

            # 查找相似项
            similar_pairs = []
            
            # SimhashIndex.get_near_dups 返回的是 block_id 列表
            # 我们需要自己处理，确保不重复且是 (id1, id2, distance) 格式
            
            checked_pairs = set()

            for block_id, hash_value in block_hashes:
                simhash_obj = Simhash(int(hash_value, 16), f=hash_bits)
                near_dups = index.get_near_dups(simhash_obj)
                
                for dup_id in near_dups:
                    if block_id == dup_id:
                        continue # 跳过自身
                    
                    # 确保对 (id1, id2) 的顺序不敏感，避免重复
                    pair = tuple(sorted((block_id, dup_id)))
                    if pair in checked_pairs:
                        continue
                        
                    # 找到另一个块的哈希值并计算精确距离
                    dup_hash_value = next(h for i, h in block_hashes if i == dup_id)
                    distance = self.calculate_hamming_distance(hash_value, dup_hash_value)
                    
                    # 再次确认距离是否在阈值内 (get_near_dups 可能有误差)
                    if distance <= threshold:
                        similar_pairs.append((block_id, dup_id, distance))
                        checked_pairs.add(pair)
                        
            return similar_pairs

        except Exception as e:
            raise SimHashComparisonError(f"使用 SimhashIndex 查找相似对时出错: {e}")

    # def _tokenize(self, text: str) -> List[str]:
    #     """
    #     【关键实现细节】文本分词方法。
    #     对于中文，必须使用像 jieba 这样的库。
    #     返回词语列表。
    #     """
    #     # 伪代码: import jieba; return list(jieba.cut(text))
    #     # 实际实现时需要处理停用词等。
    #     # 为简化伪代码，此处返回基于空格的分词（对中文无效）
    #     return text.split()