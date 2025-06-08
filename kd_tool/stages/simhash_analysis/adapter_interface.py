"""
====================开发指引======================
kd_tool/stages/simhash_analysis/adapter_interface.py - v0.1
=================================================

**【文件定位】**
- 路径：kd_tool/stages/simhash_analysis/adapter_interface.py
- 所属：simhash_analysis 阶段模块，适配器接口层。
- 作用：为SimHash算法适配器定义标准抽象接口，供具体实现类继承，服务于文本块级去重流程。

**【模块职责（SRP）】**
- 唯一职责：定义SimHash相关算法的标准接口，确保不同实现的可插拔与可替换。

**【依赖关系与注入】**
- 依赖：无外部依赖，仅依赖标准库（abc、typing）及本阶段专属异常（通过绝对导入）。
- 注入方式：具体实现类通过工厂注入到Stage层。
- Mock点：需为接口定义Mock实现，便于Stage层单元测试。

**【输入输出规范】**
- `calculate_simhash(text: str, hash_bits: int) -> str`
  - 输入：原始文本（str）、SimHash位数（int，64或128）
  - 输出：SimHash指纹（十六进制字符串）
  - 异常：SimHashCalculationError
- `calculate_hamming_distance(hash1: str, hash2: str) -> int`
  - 输入：两个SimHash指纹（str）
  - 输出：汉明距离（int）
  - 异常：ValueError、SimHashComparisonError
- `find_similar_pairs(block_hashes: List[Tuple[str, str]], threshold: int, hash_bits: int) -> List[Tuple[str, str, int]]`
  - 输入：区块ID与SimHash值对列表、阈值、位数
  - 输出：相似对及距离列表
  - 异常：实现可自定义抛出SimHashAnalysisError子类

**【核心架构约束】**
- 禁止直接实例化依赖，禁止业务逻辑与存储耦合。
- 所有方法参数与返回值必须类型注解。
- 仅定义接口，不包含任何实现逻辑。
- 关键类与方法需补充WHY/WHAT/HOW三段式注释。
- 异常类型必须通过绝对导入引用`kd_tool.stages.simhash_analysis.errors`。
- 日志记录由具体实现负责，接口不涉及日志。

**【接口与DTO规范】**
- 仅暴露抽象方法，不含实现。
- DTO/异常类需在专用模块定义并绝对导入。
- 若需结构化数据，后续通过Pydantic DTO扩展。

**【日志与安全】**
- 本接口不涉及日志与安全，具体实现需遵循日志上下文绑定与敏感信息处理规范。

**【任务清单】**
1. 在`adapter_interface.py`顶部及关键类/方法补充WHY/WHAT/HOW三段式注释。
2. 通过绝对导入引用`kd_tool.stages.simhash_analysis.errors`中的异常类型。
3. 明确接口方法签名与注释，确保类型注解与文档规范。
4. 规划Mock实现与接口测试用例，确保可测试性。
5. 检查所有导入均为绝对导入，符合架构规范。
6. 预留DTO/Pydantic扩展点，便于未来结构化数据传递。

**【其他说明】**
- 若需支持多种SimHash算法或第三方库，均应通过本接口扩展。
- 若需支持分布式/批量高效查重，可在`find_similar_pairs`方法扩展参数与返回结构。
- 若后续异常类型增多，建议在`errors.py`中分层管理，保持异常体系清晰。

"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class SimHashAdapterInterface(ABC):
    """
    WHY: 统一SimHash算法适配器接口，支持多实现可插拔，服务于文本块级去重等场景。
    WHAT: 约定SimHash指纹计算、汉明距离计算及批量相似对查找的标准方法签名与异常规范。
    """

    @abstractmethod
    def calculate_simhash(self, text: str, hash_bits: int) -> str:
        """
        WHY: 计算文本的SimHash指纹，便于后续相似性检测。
        WHAT: 输入原始文本和目标位数，输出SimHash十六进制指纹。
        HOW:
            - 实现不负责文本预处理，调用方需保证输入已规范化。
            - 仅支持64或128位SimHash。
        参数:
            text (str): 输入文本（需预处理）。
            hash_bits (int): SimHash位数（64或128）。
        返回:
            str: SimHash指纹（16或32位十六进制字符串）。
        异常:
            SimHashCalculationError: 计算失败时抛出。
        """
        pass

    @abstractmethod
    def calculate_hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        WHY: 比较两个SimHash指纹的相似性，支持去重与聚类。
        WHAT: 输入两个等长SimHash指纹，输出汉明距离。
        HOW:
            - 两个指纹长度必须一致。
        参数:
            hash1 (str): 第一个SimHash指纹。
            hash2 (str): 第二个SimHash指纹。
        返回:
            int: 汉明距离。
        异常:
            ValueError: 指纹长度不一致时抛出。
            SimHashComparisonError: 其他比较错误时抛出。
        """
        pass

    @abstractmethod
    def find_similar_pairs(
        self, block_hashes: List[Tuple[str, str]], threshold: int, hash_bits: int
    ) -> List[Tuple[str, str, int]]:
        """
        WHY: 批量查找汉明距离小于阈值的区块对，提升大规模去重效率。
        WHAT: 输入区块ID与SimHash值对列表、阈值和位数，输出所有相似对及其距离。
        HOW:
            - 推荐高效实现，若无法优化可由Stage层O(n^2)遍历。
        参数:
            block_hashes (List[Tuple[str, str]]): (区块ID, SimHash值)列表。
            threshold (int): 汉明距离阈值。
            hash_bits (int): SimHash位数。
        返回:
            List[Tuple[str, str, int]]: (区块ID1, 区块ID2, 汉明距离)列表。
        异常:
            SimHashAnalysisError及其子类: 实现可自定义抛出。
        """
        pass
