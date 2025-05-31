"""
=================================================
md5_analysis_stage.py - P05 MD5 分析阶段 (v4.6)
=================================================

**模块功能**:

# ------------------------------------------------------------------------------
# 文件名: kd_tool/stages/md5analysis/md5_analysis_stage.py
# 模块: MD5 分析阶段 (MD5AnalysisStage)
# 描述:
#   此模块负责计算内容块的 MD5 哈希值，并在同一阶段内找出 MD5 完全相同的块，
#   生成相应的成对分析结果 (AnalysisResultDTO)。
#   它实现了计算与比较的双重职责，遵循我们最终的架构决策。
# 架构约束:
#   - 必须实现 StageInterface。
#   - 必须通过构造函数接收 Logger, StorageInterface, MD5AnalysisStageSettings。
#   - 必须使用 StorageInterface 进行所有 *外部* 数据持久化操作 (但此阶段主要操作 Context)。
#   - 必须抛出 MD5InputError 或 MD5CalculationError。
#   - 必须是无状态的。
# ------------------------------------------------------------------------------
"""
from typing import List, Dict, Optional
from kd_tool.logging.protocols import LoggerProtocol
import itertools
from kd_tool.core.interfaces import StageInterface
from kd_tool.core.core_dtos import PipelineContextDTO
from kd_tool.schemas.dtos import ContentBlockDTO, AnalysisResultDTO
from kd_tool.schemas.enums import AnalysisType
from kd_tool.stages.md5analysis.settings_models import MD5AnalysisStageSettings
from kd_tool.stages.md5analysis.errors import MD5InputError, MD5CalculationError, MD5AnalysisError
from kd_tool.storage.storage_interface import StorageInterface


class MD5AnalysisStage(StageInterface):
    """
    MD5 分析阶段实现 (计算与比较合并)。

    负责计算内容块的 MD5 哈希值，找出重复项，并生成 AnalysisResultDTO。
    """

    def __init__(self, logger: LoggerProtocol, storage: StorageInterface, settings:
        MD5AnalysisStageSettings) ->None:
        """
        构造函数 - 严格执行依赖注入。
        """
        self._logger = logger.bind(stage='MD5AnalysisStage')
        self._storage = storage
        self._settings = settings
        self._logger.info('MD5AnalysisStage (Combined) initialized.')

    def process(self, context: PipelineContextDTO) ->PipelineContextDTO:
        """
        执行 MD5 计算与比较流程。
        """
        logger = context.run_logger.bind(stage_name=self.__class__.__name__)
        logger.info("MD5 分析阶段开始...")
        # 检查阶段是否启用
        if not self._settings.enabled:
            logger.warning('MD5AnalysisStage is disabled. Skipping.')
            return context
        content_blocks_to_check = list(context.content_blocks.values())
        if not content_blocks_to_check:
            logger.info('No content blocks found in context. Skipping.')
            return context
        logger.info(
            f'Found {len(content_blocks_to_check)} blocks to check for MD5.')
        md5_map: Dict[str, List[str]] = {}
        processed_count = 0
        error_count = 0
        logger.debug('Phase 1: Calculating MD5 hashes...')
        for block in content_blocks_to_check:
            try:
                logger.trace(
                    f'Calculating MD5 for block_id: {block.block_id}...')
                if not block.analysis_text:
                    raise MD5InputError(block.block_id,
                        'Analysis text is empty or None.')
                md5_hash = self._calculate_block_md5(block.analysis_text,
                    block.block_id)
                context.content_blocks[block.block_id].text_hash_md5 = md5_hash
                md5_map.setdefault(md5_hash, []).append(block.block_id)
                processed_count += 1
            except (MD5InputError, MD5CalculationError) as e:
                logger.error(
                    f'MD5 processing error for block {block.block_id}: {e}')
                context.add_error(e)
                error_count += 1
            except Exception as e:
                logger.exception(
                    f'Unexpected error processing block {block.block_id} for MD5.'
                    )
                context.add_error(MD5CalculationError(block.block_id,
                    original_exception=e))
                error_count += 1
        logger.info(
            f'Phase 1 finished. Calculated MD5 for {processed_count} blocks, Errors: {error_count}.'
            )
        logger.debug('Phase 2: Finding duplicates and generating results...')
        duplicate_groups_found = 0
        results_generated = 0
        for md5_hash, block_ids in md5_map.items():
            if len(block_ids) > 1:
                duplicate_groups_found += 1
                logger.debug(
                    f'Found duplicate group (MD5: {md5_hash}): {block_ids}')
                for block_id_1, block_id_2 in itertools.combinations(block_ids,
                    2):
                    try:
                        analysis_result = AnalysisResultDTO(block_id_1=
                            block_id_1, block_id_2=block_id_2,
                            analysis_type=AnalysisType.MD5, score=1.0,
                            details={'md5_hash': md5_hash})
                        context.add_analysis_result(analysis_result)
                        results_generated += 1
                    except Exception as e:
                        logger.error(
                            f'Error generating MD5 result for pair ({block_id_1}, {block_id_2}): {e}'
                            )
                        context.add_error(MD5AnalysisError(
                            f'Failed to generate result for pair ({block_id_1}, {block_id_2})'
                            , original_exception=e))
                        error_count += 1
        logger.info(
            f'Phase 2 finished. Found {duplicate_groups_found} duplicate groups, generated {results_generated} results.'
            )
        logger.info(
            f'Combined MD5 analysis process finished. Total errors: {error_count}.'
            )
        return context

    def _calculate_block_md5(self, text: str, block_id: str) ->str:
        """
        计算文本的 MD5 哈希值 (伪代码)。

        架构说明:
            - **此处无需展示真实的 hashlib 实现细节**。
            - **coding 阶段要求**: 必须使用 'utf-8' 编码。
            - **coding 阶段要求**: 必须捕获 `UnicodeEncodeError` 并抛出 `MD5CalculationError`。
        """
        try:
            md5_hash = f'dummy_md5_for_{block_id}'
            return md5_hash
        except UnicodeEncodeError as e:
            raise MD5CalculationError(block_id, original_exception=e)
        except Exception as e:
            raise MD5CalculationError(block_id, original_exception=e)
