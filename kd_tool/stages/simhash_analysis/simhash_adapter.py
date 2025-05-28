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
try:
    from simhash import Simhash, SimhashIndex
except ImportError:
    print('警告: `simhash` 库未安装。SimHashAdapter 将无法工作。')
    Simhash = None
    SimhashIndex = None
from kd_tool.stages.simhash_analysis.adapter_interface import SimHashAdapterInterface
from kd_tool.stages.simhash_analysis.errors import SimHashCalculationError, SimHashComparisonError, SimHashAdapterError


class SimhashLibAdapter(SimHashAdapterInterface):
    """
    使用 `simhash` 库实现 SimHash 功能的适配器。
    """

    def calculate_simhash(self, text: str, hash_bits: int) ->str:
        """计算 SimHash 指纹。"""
        if Simhash is None:
            raise SimHashAdapterError('`simhash` 库未安装。')
        if hash_bits not in [64, 128]:
            raise ValueError(f'不支持的哈希位数: {hash_bits}。只支持 64 或 128。')
        try:
            simhash_obj = Simhash(text, f=hash_bits)
            hex_format = f'0{hash_bits // 4}x'
            return format(simhash_obj.value, hex_format)
        except Exception as e:
            raise SimHashCalculationError(block_id='<unknown>',
                original_error=e)

    def calculate_hamming_distance(self, hash1_hex: str, hash2_hex: str) ->int:
        """计算汉明距离。"""
        if Simhash is None:
            raise SimHashAdapterError('`simhash` 库未安装。')
        if len(hash1_hex) != len(hash2_hex):
            raise ValueError('SimHash 指纹长度不一致，无法比较。')
        try:
            hash1_int = int(hash1_hex, 16)
            hash2_int = int(hash2_hex, 16)
            bits = len(hash1_hex) * 4
            simhash1 = Simhash(hash1_int, f=bits)
            simhash2 = Simhash(hash2_int, f=bits)
            return simhash1.distance(simhash2)
        except Exception as e:
            raise SimHashComparisonError(
                f'比较 {hash1_hex} 和 {hash2_hex} 时出错: {e}')

    def find_similar_pairs(self, block_hashes: List[Tuple[str, str]],
        threshold: int, hash_bits: int) ->List[Tuple[str, str, int]]:
        """使用 SimhashIndex 高效查找相似对。"""
        if SimhashIndex is None:
            raise SimHashAdapterError('`simhash` 库的 SimhashIndex 未找到或未安装。')
        if not block_hashes:
            return []
        try:
            index = SimhashIndex([], f=hash_bits, k=threshold)
            for block_id, hash_value in block_hashes:
                simhash_obj = Simhash(int(hash_value, 16), f=hash_bits)
                index.add(block_id, simhash_obj)
            similar_pairs = []
            checked_pairs = set()
            for block_id, hash_value in block_hashes:
                simhash_obj = Simhash(int(hash_value, 16), f=hash_bits)
                near_dups = index.get_near_dups(simhash_obj)
                for dup_id in near_dups:
                    if block_id == dup_id:
                        continue
                    pair = tuple(sorted((block_id, dup_id)))
                    if pair in checked_pairs:
                        continue
                    dup_hash_value = next(h for i, h in block_hashes if i ==
                        dup_id)
                    distance = self.calculate_hamming_distance(hash_value,
                        dup_hash_value)
                    if distance <= threshold:
                        similar_pairs.append((block_id, dup_id, distance))
                        checked_pairs.add(pair)
            return similar_pairs
        except Exception as e:
            raise SimHashComparisonError(f'使用 SimhashIndex 查找相似对时出错: {e}')
