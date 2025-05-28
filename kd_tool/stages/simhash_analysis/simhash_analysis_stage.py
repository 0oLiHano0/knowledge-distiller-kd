"""
=================================================
simhash_analysis_stage.py - P06 SimHash 分析阶段实现 (v4.6)
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
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.core.interfaces import StageInterface, StorageInterface
from kd_tool.core.core_dtos import PipelineContextDTO
from kd_tool.schemas.dtos import ContentBlockDTO, AnalysisResultDTO
from kd_tool.schemas.enums import AnalysisType
from kd_tool.stages.simhash_analysis.settings_models import SimHashAnalysisStageSettings
from kd_tool.stages.simhash_analysis.adapter_interface import SimHashAdapterInterface
from kd_tool.stages.simhash_analysis.errors import SimHashAnalysisError, SimHashCalculationError, SimHashComparisonError


class SimHashAnalysisStage(StageInterface):
    """
    P06 - SimHash 分析阶段。
    负责计算内容块的 SimHash 指纹，并找出相似的内容块对。
    """

    def __init__(self, logger: LoggerProtocol, storage: StorageInterface, settings:
        SimHashAnalysisStageSettings, adapter: SimHashAdapterInterface):
        """
        **规范**: 构造函数，**必须**通过 DI 注入所有依赖。
        """
        self._logger = logger.bind(stage='SimHashAnalysisStage')
        self._storage = storage
        self._settings = settings
        self._adapter = adapter

    def process(self, context: PipelineContextDTO) ->PipelineContextDTO:
        """
        执行 SimHash 分析流水线阶段。
        **[指令]** 必须使用 `context.run_logger` 进行日志记录。
        **[指令]** 创建 `ContentBlockDTO` (更新时) 和 `AnalysisResultDTO` 时 **严禁** 包含 `task_id` 字段。
        **[指令]** 调用 `storage.save_content_blocks` 和 `storage.save_analysis_results` 时
                  **不再需要** 传递 `task_id` 参数 (除非存储接口方法本身需要，但目前设计是不需要)。
        """
        run_logger = context.run_logger.bind(stage_name=self.__class__.__name__
            )
        run_logger.info('开始执行 SimHash 分析阶段...')
        if not self._settings.enabled:
            run_logger.warning('SimHash 分析阶段已禁用，跳过处理。')
            return context
        try:
            blocks_to_process = self._get_blocks_to_process(context)
            if not blocks_to_process:
                run_logger.info('没有需要进行 SimHash 计算或比较的内容块。')
                return context
            run_logger.info(f'找到 {len(blocks_to_process)} 个内容块需要进行 SimHash 处理。'
                )
            blocks_with_hash, calculation_errors = self._calculate_hashes(
                blocks_to_process, run_logger)
            for error in calculation_errors:
                context.add_error(error)
            if blocks_with_hash:
                self._update_storage_hashes(blocks_with_hash, run_logger,
                    context)
            all_blocks_for_comparison = self._get_all_blocks_for_comparison(
                context)
            run_logger.info(
                f'共有 {len(all_blocks_for_comparison)} 个内容块参与 SimHash 比较。')
            analysis_results = self._compare_hashes_and_generate_results(
                all_blocks_for_comparison, run_logger, context.task_id)
            run_logger.info(f'生成了 {len(analysis_results)} 个 SimHash 分析结果。')
            for result in analysis_results:
                context.add_analysis_result(result)
            if analysis_results:
                self._save_analysis_results(analysis_results, run_logger,
                    context)
            run_logger.success('SimHash 分析阶段执行完毕。')
        except Exception as e:
            error = SimHashAnalysisError(f'SimHash 分析阶段发生未知错误: {e}',
                original_exception=e)
            run_logger.exception(error)
            context.add_error(error)
        return context

    def _get_blocks_to_process(self, context: PipelineContextDTO) ->List[
        ContentBlockDTO]:
        if self._settings.force_recalculate:
            return list(context.content_blocks.values())
        else:
            return [b for b in context.content_blocks.values() if b.
                simhash_value is None]

    def _calculate_hashes(self, blocks: List[ContentBlockDTO], logger: LoggerProtocol
        ) ->Tuple[List[ContentBlockDTO], List[SimHashAnalysisError]]:
        calculated_blocks = []
        errors = []
        hash_bits = self._settings.hash_bits
        for block in blocks:
            if not block.analysis_text:
                logger.warning(
                    f'内容块 {block.block_id} 没有 `analysis_text`，跳过 SimHash 计算。')
                continue
            try:
                hash_value = self._adapter.calculate_simhash(block.
                    analysis_text, hash_bits)
                block.simhash_value = hash_value
                calculated_blocks.append(block)
                logger.trace(f'为内容块 {block.block_id} 计算 SimHash: {hash_value}')
            except Exception as e:
                err = SimHashCalculationError(block_id=block.block_id,
                    original_error=e)
                logger.error(f'为内容块 {block.block_id} 计算 SimHash 失败: {err}')
                errors.append(err)
        return calculated_blocks, errors

    def _update_storage_hashes(self, blocks: List[ContentBlockDTO], logger:
        LoggerProtocol, context: PipelineContextDTO) ->None:
        logger.info(f'正在将 {len(blocks)} 个新的 SimHash 值更新到存储...')
        try:
            self._storage.save_content_blocks(blocks)
            logger.debug(f'成功更新 {len(blocks)} 个 SimHash 值。')
        except Exception as e:
            err = SimHashAnalysisError(f'更新 SimHash 值到存储时失败: {e}',
                original_exception=e)
            logger.error(err)
            context.add_error(err)

    def _get_all_blocks_for_comparison(self, context: PipelineContextDTO
        ) ->List[ContentBlockDTO]:
        return [b for b in context.content_blocks.values() if b.
            simhash_value is not None]

    def _compare_hashes_and_generate_results(self, blocks: List[
        ContentBlockDTO], logger: LoggerProtocol, task_id: str) ->List[
        AnalysisResultDTO]:
        results = []
        threshold = self._settings.hamming_distance_threshold
        hash_bits = self._settings.hash_bits
        if len(blocks) < 2:
            return []
        block_hashes = [(b.block_id, b.simhash_value) for b in blocks if b.
            simhash_value]
        if not block_hashes:
            return []
        first_hash_len = len(block_hashes[0][1])
        if not all(len(h) == first_hash_len for _, h in block_hashes):
            logger.error('发现 SimHash 位数不一致的内容块，无法进行比较。')
            return []
        if first_hash_len * 4 != hash_bits:
            logger.error(
                f'内容块的哈希位数 ({first_hash_len * 4}) 与设置 ({hash_bits}) 不符。')
            return []
        logger.info(
            f'正在使用汉明距离阈值 {threshold} 比较 {len(block_hashes)} 个 SimHash...')
        try:
            if hasattr(self._adapter, 'find_similar_pairs') and callable(self
                ._adapter.find_similar_pairs):
                logger.debug('使用适配器的 `find_similar_pairs` 方法进行比较。')
                similar_pairs = self._adapter.find_similar_pairs(block_hashes,
                    threshold, hash_bits)
            else:
                logger.warning(
                    'SimHash 适配器未提供 `find_similar_pairs`，将进行 O(n^2) 比较。')
                similar_pairs = self._brute_force_comparison(block_hashes,
                    threshold, logger)
            for b1_id, b2_id, distance in similar_pairs:
                similarity_score = 1.0 - distance / hash_bits
                result = AnalysisResultDTO(block_id_1=b1_id, block_id_2=
                    b2_id, analysis_type=AnalysisType.SIMHASH, score=
                    similarity_score, details={'hamming_distance': distance,
                    'hash_bits': hash_bits, 'threshold': threshold})
                results.append(result)
        except Exception as e:
            err = SimHashComparisonError(f'比较 SimHash 时发生错误: {e}',
                original_exception=e)
            logger.error(err)
        return results

    def _brute_force_comparison(self, block_hashes: List[Tuple[str, str]],
        threshold: int, logger: LoggerProtocol) ->List[Tuple[str, str, int]]:
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
                    logger.warning(f'比较 {b1_id} 和 {b2_id} 时出错: {e}')
        return pairs

    def _save_analysis_results(self, results: List[AnalysisResultDTO],
        logger: LoggerProtocol, context: PipelineContextDTO) ->None:
        logger.info(f'正在将 {len(results)} 个 SimHash 分析结果保存到存储...')
        try:
            self._storage.save_analysis_results(results)
            logger.debug(f'成功保存 {len(results)} 个分析结果。')
        except Exception as e:
            err = SimHashAnalysisError(f'保存 SimHash 分析结果到存储时失败: {e}',
                original_exception=e)
            logger.error(err)
            context.add_error(err)
