```python

# kd_tool/stages/simhash_analysis/simhash_analysis_stage.py
# -*- coding: utf-8 -*-

"""
=================================================
simhash_analysis_stage.py - P06 SimHash 分析阶段实现 (v4.5)
=================================================

**模块功能**:

- 实现 `StageInterface`，执行 SimHash 分析流程。
- **职责**:
    1. 为 `ContentBlockDTO` 计算 SimHash 指纹 (如果不存在或需要强制计算)。
    2. 将计算出的 SimHash 指纹更新到 `ContentBlockDTO` 和存储中。
    3. 比较块之间的 SimHash 指纹。
    4. 生成 `AnalysisResultDTO` 并添加到 `PipelineContextDTO` 和存储中。
- **规范**:
    - **必须**通过构造函数注入 `Logger`、`StorageInterface`、`SimHashAnalysisStageSettings` 和 `SimHashAdapterInterface`。
    - **严禁**包含任何 SimHash 计算的**具体**实现细节，**必须**使用 `SimHashAdapterInterface`。
    - **必须**处理 `SimHashAnalysisError` 并记录到 `PipelineContextDTO`。
    - **必须**使用 `run_logger` 进行日志记录。

---
"""

from typing import List, Dict, Tuple
from loguru import Logger

from ...core.interfaces import StageInterface, StorageInterface
from ...schemas.dtos import PipelineContextDTO, ContentBlockDTO, AnalysisResultDTO
from ...schemas.settings_models import SimHashAnalysisStageSettings
from ...schemas.enums import AnalysisType
from .adapter_interface import SimHashAdapterInterface
from .errors import SimHashAnalysisError, SimHashCalculationError

class SimHashAnalysisStage(StageInterface):
    """
    P06 - SimHash 分析阶段。
    负责计算内容块的 SimHash 指纹，并找出相似的内容块对。
    """

    def __init__(self,
                 logger: Logger,
                 storage: StorageInterface,
                 settings: SimHashAnalysisStageSettings,
                 adapter: SimHashAdapterInterface):
        """
        **规范**: 构造函数，**必须**通过 DI 注入所有依赖。
        """
        self._logger = logger.bind(stage="SimHashAnalysisStage") # 绑定阶段上下文
        self._storage = storage
        self._settings = settings
        self._adapter = adapter

    def process(self, context: PipelineContextDTO) -> PipelineContextDTO:
        """
        执行 SimHash 分析流水线阶段。

        **参数**:
            context (PipelineContextDTO): 流水线上下文。

        **返回**:
            PipelineContextDTO: 更新后的流水线上下文。
        """
        run_logger = context.run_logger.bind(stage="SimHashAnalysisStage") # 使用带 task_id 的 logger
        run_logger.info("开始执行 SimHash 分析阶段...")

        if not self._settings.enabled:
            run_logger.warning("SimHash 分析阶段已禁用，跳过处理。")
            return context

        try:
            # 1. 获取需要处理的内容块
            blocks_to_process = self._get_blocks_to_process(context)
            if not blocks_to_process:
                run_logger.info("没有需要进行 SimHash 计算或比较的内容块。")
                return context
                
            run_logger.info(f"找到 {len(blocks_to_process)} 个内容块需要进行 SimHash 处理。")

            # 2. 计算 SimHash (如果需要)
            blocks_with_hash, calculation_errors = self._calculate_hashes(blocks_to_process, run_logger)
            for error in calculation_errors: context.add_error(error)

            # 3. 更新存储中的 SimHash 值 (如果计算了新的)
            if blocks_with_hash: # 只更新计算了哈希的
                 self._update_storage_hashes(blocks_with_hash, run_logger, context)

            # 4. 获取所有 *相关* 内容块 (包括本次计算的和之前已有的) 用于比较
            all_blocks_for_comparison = self._get_all_blocks_for_comparison(context)
            run_logger.info(f"共有 {len(all_blocks_for_comparison)} 个内容块参与 SimHash 比较。")

            # 5. 比较 SimHash 并生成结果
            analysis_results = self._compare_hashes_and_generate_results(all_blocks_for_comparison, run_logger)
            run_logger.info(f"生成了 {len(analysis_results)} 个 SimHash 分析结果。")

            # 6. 添加结果到上下文并更新存储
            for result in analysis_results:
                context.add_analysis_result(result)
            
            if analysis_results:
                self._save_analysis_results(analysis_results, run_logger, context)

            run_logger.success("SimHash 分析阶段执行完毕。")

        except Exception as e:
            error = SimHashAnalysisError(f"SimHash 分析阶段发生未知错误: {e}")
            run_logger.exception(error)
            context.add_error(error)
            # 根据 Orchestrator 的策略，这里可能需要重新抛出或处理

        return context

    def _get_blocks_to_process(self, context: PipelineContextDTO) -> List[ContentBlockDTO]:
        """获取需要计算或比较 SimHash 的内容块。"""
        # 规范: 如果设置了 force_recalculate，则返回所有块。
        # 否则，只返回 simhash_value 为 None 的块。
        # 考虑: 是否只处理尚未进行 SimHash 分析的块？目前逻辑是先算哈希，再比较。
        if self._settings.force_recalculate:
            return list(context.content_blocks.values())
        else:
            return [b for b in context.content_blocks.values() if b.simhash_value is None]

    def _calculate_hashes(self, 
                          blocks: List[ContentBlockDTO], 
                          logger: Logger) -> Tuple[List[ContentBlockDTO], List[SimHashAnalysisError]]:
        """批量计算内容块的 SimHash。"""
        calculated_blocks = []
        errors = []
        hash_bits = self._settings.hash_bits

        for block in blocks:
            if not block.analysis_text:
                logger.warning(f"内容块 {block.block_id} 没有 `analysis_text`，跳过 SimHash 计算。")
                continue

            try:
                # 使用适配器计算哈希
                hash_value = self._adapter.calculate_simhash(block.analysis_text, hash_bits)
                block.simhash_value = hash_value
                calculated_blocks.append(block)
                logger.trace(f"为内容块 {block.block_id} 计算 SimHash: {hash_value}")
            except Exception as e:
                err = SimHashCalculationError(block_id=block.block_id, original_error=e)
                logger.error(f"为内容块 {block.block_id} 计算 SimHash 失败: {err}")
                errors.append(err)
        
        return calculated_blocks, errors

    def _update_storage_hashes(self, blocks: List[ContentBlockDTO], logger: Logger, context: PipelineContextDTO) -> None:
        """将计算出的 SimHash 值持久化到存储。"""
        logger.info(f"正在将 {len(blocks)} 个新的 SimHash 值更新到存储...")
        try:
            # 规范: 存储接口应支持批量更新 ContentBlockDTO。
            self._storage.save_content_blocks(blocks, context.task_id)
            logger.debug(f"成功更新 {len(blocks)} 个 SimHash 值。")
        except Exception as e:
            err = SimHashAnalysisError(f"更新 SimHash 值到存储时失败: {e}")
            logger.error(err)
            context.add_error(err)

    def _get_all_blocks_for_comparison(self, context: PipelineContextDTO) -> List[ContentBlockDTO]:
        """获取所有具有 SimHash 值的块用于比较。"""
        return [b for b in context.content_blocks.values() if b.simhash_value is not None]

    def _compare_hashes_and_generate_results(self, 
                                             blocks: List[ContentBlockDTO], 
                                             logger: Logger) -> List[AnalysisResultDTO]:
        """比较 SimHash 值并生成 AnalysisResultDTO。"""
        results = []
        threshold = self._settings.hamming_distance_threshold
        hash_bits = self._settings.hash_bits

        if len(blocks) < 2:
            return []

        block_hashes = [(b.block_id, b.simhash_value) for b in blocks if b.simhash_value]

        # 检查所有哈希值的位数是否一致
        first_hash_len = len(block_hashes[0][1])
        if not all(len(h) == first_hash_len for _, h in block_hashes):
            logger.error("发现 SimHash 位数不一致的内容块，无法进行比较。")
            # TODO: 可以在此处添加错误到 context
            return []

        # 检查哈希位数是否与设置匹配
        if first_hash_len * 4 != hash_bits:
            logger.error(f"内容块的哈希位数 ({first_hash_len * 4}) 与设置 ({hash_bits}) 不符。")
            return []
            
        logger.info(f"正在使用汉明距离阈值 {threshold} 比较 {len(block_hashes)} 个 SimHash...")

        try:
            # 方案 A: 使用适配器的高效查找 (如果实现)
            if hasattr(self._adapter, 'find_similar_pairs') and callable(self._adapter.find_similar_pairs):
                logger.debug("使用适配器的 `find_similar_pairs` 方法进行比较。")
                similar_pairs = self._adapter.find_similar_pairs(block_hashes, threshold, hash_bits)
            # 方案 B: 在 Stage 中进行 O(n^2) 比较 (备用)
            else:
                 logger.warning("SimHash 适配器未提供 `find_similar_pairs`，将进行 O(n^2) 比较。")
                 similar_pairs = self._brute_force_comparison(block_hashes, threshold, logger)

            # 将找到的对转换为 AnalysisResultDTO
            task_id = blocks[0].task_id # 所有块应该有相同的 task_id
            for b1_id, b2_id, distance in similar_pairs:
                similarity_score = 1.0 - (distance / hash_bits)
                result = AnalysisResultDTO(
                    block_id_1=b1_id,
                    block_id_2=b2_id,
                    analysis_type=AnalysisType.SIMHASH,
                    score=similarity_score,
                    details={
                        "hamming_distance": distance,
                        "hash_bits": hash_bits,
                        "threshold": threshold
                    },
                    task_id=task_id
                )
                results.append(result)

        except Exception as e:
             err = SimHashComparisonError(f"比较 SimHash 时发生错误: {e}")
             logger.error(err)
             # TODO: 可以在此处添加错误到 context
        
        return results

    def _brute_force_comparison(self, 
                                 block_hashes: List[Tuple[str, str]], 
                                 threshold: int, 
                                 logger: Logger) -> List[Tuple[str, str, int]]:
        """简单的 O(n^2) 暴力比较方法。"""
        pairs = []
        n = len(block_hashes)
        for i in range(n):
            for j in range(i + 1, n):
                b1_id, h1 = block_hashes[i]
                b2_id, h2 = block_hashes[j]
                try:
                    distance = self._adapter.calculate_hamming_distance(h1, h2)
                    if distance <= threshold:
                        pairs.append((b1_id, b2_id, distance))
                except Exception as e:
                    logger.warning(f"比较 {b1_id} 和 {b2_id} 时出错: {e}")
        return pairs

    def _save_analysis_results(self, 
                               results: List[AnalysisResultDTO], 
                               logger: Logger, 
                               context: PipelineContextDTO) -> None:
        """将分析结果持久化到存储。"""
        logger.info(f"正在将 {len(results)} 个 SimHash 分析结果保存到存储...")
        try:
            # 规范: 存储接口应支持批量保存 AnalysisResultDTO。
            self._storage.save_analysis_results(results, context.task_id)
            logger.debug(f"成功保存 {len(results)} 个分析结果。")
        except Exception as e:
            err = SimHashAnalysisError(f"保存 SimHash 分析结果到存储时失败: {e}")
            logger.error(err)
            context.add_error(err)

```