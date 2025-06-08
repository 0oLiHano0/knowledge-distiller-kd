"""
=================================================
semantic_analysis_stage.py - P07 语义分析阶段实现 (v4.6)
=================================================

**模块功能**:

- 实现 `StageInterface`，执行语义分析流程。
- **职责**:
    1. 筛选需要进行语义比较的内容块对 (根据 `comparison_strategy`)。
    2. 使用 `SemanticAdapterInterface` 计算嵌入向量。
    3. 使用 `SemanticAdapterInterface` 计算相似度。
    4. 生成 `AnalysisResultDTO` 并添加到 `PipelineContextDTO` 和存储中。
- **规范**:
    - **必须**通过 DI 注入 `Logger`、`StorageInterface`、`SemanticAnalysisStageSettings` 和 `SemanticAdapterInterface`。
    - **严禁**包含任何模型加载、嵌入计算的具体实现，**必须**使用适配器。
    - **必须**处理 `SemanticAnalysisError` 并记录。
    - **必须**考虑性能问题，尤其是比较策略。

---
"""

from typing import List, Dict, Tuple
from uuid import UUID
from kd_tool.logging.protocols import LoggerProtocol
import numpy as np
from kd_tool.core.interfaces import StageInterface
from kd_tool.core.core_dtos import PipelineContextDTO
from kd_tool.schemas.dtos import ContentBlockDTO, AnalysisResultDTO
from kd_tool.schemas.enums import AnalysisType
from kd_tool.stages.semantic_analysis.settings_models import (
    SemanticAnalysisStageSettings,
)
from kd_tool.stages.semantic_analysis.adapter_interface import SemanticAdapterInterface
from kd_tool.stages.semantic_analysis.errors import (
    SemanticAnalysisError,
    ModelLoadingError,
    EmbeddingCalculationError,
    SimilarityCalculationError,
)


class SemanticAnalysisStage(StageInterface):
    """
    P07 - 语义分析阶段。
    负责内容块的语义相似度分析。

    性能警告：默认批量比较为O(n^2)暴力实现，数据量大时不可用。必须实现降采样、索引或近似算法以满足性能目标。
    """

    def __init__(
        self,
        logger: LoggerProtocol,
        settings: SemanticAnalysisStageSettings,
        adapter: SemanticAdapterInterface,
    ):
        """构造函数，通过 DI 注入所有依赖。"""
        self._logger = logger.bind(stage="SemanticAnalysisStage")
        self._settings = settings
        self._adapter = adapter

    def process(self, context: PipelineContextDTO) -> PipelineContextDTO:
        """
        执行语义分析流水线阶段。
        **[指令]** 必须使用 `context.run_logger` 进行日志记录。
        **[指令]** 创建 `AnalysisResultDTO` 时 **严禁** 包含 `task_id` 字段。
        **[指令]** 严禁直接调用 `storage` 进行写入操作，**必须**通过 `PipelineContextDTO` 进行状态同步。
        """
        run_logger = context.run_logger.bind(stage_name=self.__class__.__name__)
        run_logger.info("开始执行语义分析阶段...")
        if not self._settings.enabled:
            run_logger.warning("语义分析阶段已禁用，跳过处理。")
            return context
        try:
            blocks_to_process = self._get_blocks_to_process(context)
            if len(blocks_to_process) < 2:
                run_logger.info("没有足够的内容块进行语义比较。")
                return context
            run_logger.info(f"找到 {len(blocks_to_process)} 个内容块需要进行语义分析。")
            block_map = {b.block_id: b for b in blocks_to_process}
            block_ids = [b.block_id for b in blocks_to_process]
            texts_to_embed = [
                b.analysis_text
                for b in blocks_to_process
                if b.analysis_text is not None and b.analysis_text.strip() != ""
            ]
            valid_blocks_for_embedding = [
                b
                for b in blocks_to_process
                if b.analysis_text is not None and b.analysis_text.strip() != ""
            ]
            valid_block_ids_for_embedding = [
                b.block_id for b in valid_blocks_for_embedding
            ]
            if len(texts_to_embed) < 2:
                run_logger.info("有效内容块（含analysis_text）不足以进行比较。")
                return context
            run_logger.debug(f"正在为 {len(texts_to_embed)} 个文本计算嵌入向量...")
            embeddings = self._adapter.calculate_embeddings(
                texts_to_embed, self._settings.batch_size
            )
            embedding_map = {
                block_id: emb
                for block_id, emb in zip(valid_block_ids_for_embedding, embeddings)
            }
            run_logger.debug("嵌入向量计算完成。")
            pairs_to_compare = self._get_pairs_to_compare(
                valid_block_ids_for_embedding, context, run_logger
            )
            run_logger.info(
                f"确定了 {len(pairs_to_compare)} 对内容块需要进行语义比较。"
            )
            analysis_results = self._calculate_pairs_similarity(
                pairs_to_compare, embedding_map, run_logger
            )
            run_logger.info(f"生成了 {len(analysis_results)} 个语义分析结果。")
            for result in analysis_results:
                context.add_analysis_result(result)
            run_logger.success("语义分析阶段执行完毕。")
        except ModelLoadingError as mle:
            run_logger.error(f"语义模型加载失败: {mle}")
            context.add_error(mle)
        except EmbeddingCalculationError as ece:
            run_logger.error(f"嵌入向量计算失败: {ece}")
            context.add_error(ece)
        except SimilarityCalculationError as sce:
            run_logger.error(f"相似度计算失败: {sce}")
            context.add_error(sce)
        except Exception as e:
            error = SemanticAnalysisError(
                f"语义分析阶段发生未知错误: {e}", original_exception=e
            )
            run_logger.exception(str(error))
            context.add_error(error)
        return context

    def _get_blocks_to_process(
        self, context: PipelineContextDTO
    ) -> List[ContentBlockDTO]:
        return [
            b
            for b in context.content_blocks.values()
            if b.analysis_text and b.analysis_text.strip() != ""
        ]

    def _get_pairs_to_compare(
        self, block_ids: List[str], context: PipelineContextDTO, logger: LoggerProtocol
    ) -> List[Tuple[str, str]]:
        """
        获取需要比较的块对。

        性能警告：
        - 当前实现为 O(n^2) 复杂度，适用于小规模数据
        - 大规模数据（>1000块）时，建议实现降采样或近似算法
        - 考虑使用向量索引（如 FAISS）优化相似度计算
        """
        pairs = []
        n = len(block_ids)
        if self._settings.comparison_strategy == "all_pairs":
            logger.warning("使用 'all_pairs' 策略进行语义比较，计算量可能很大！")
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((block_ids[i], block_ids[j]))
            return pairs
        logger.info("使用 'pre_filtered' 策略筛选语义比较对...")
        existing_matches_pair_ids = set()
        for block_id_current in block_ids:
            if block_id_current in context.analysis_results:
                for analysis_type in [AnalysisType.MD5, AnalysisType.SIMHASH]:
                    if analysis_type in context.analysis_results[block_id_current]:
                        for result in context.analysis_results[block_id_current][
                            analysis_type
                        ]:
                            if (
                                result.analysis_type == AnalysisType.MD5
                                and result.score == 1.0
                            ):
                                existing_matches_pair_ids.add(result.pair_analysis_id)
                            elif (
                                result.analysis_type == AnalysisType.SIMHASH
                                and result.score is not None
                                and result.score >= self._settings.similarity_threshold
                            ):
                                existing_matches_pair_ids.add(result.pair_analysis_id)
        for i in range(n):
            for j in range(i + 1, n):
                b1_id, b2_id = block_ids[i], block_ids[j]
                temp_pair_id_md5 = AnalysisResultDTO._make_id(
                    b1_id, b2_id, AnalysisType.MD5
                )
                temp_pair_id_simhash = AnalysisResultDTO._make_id(
                    b1_id, b2_id, AnalysisType.SIMHASH
                )
                if (
                    temp_pair_id_md5 not in existing_matches_pair_ids
                    and temp_pair_id_simhash not in existing_matches_pair_ids
                ):
                    pairs.append((b1_id, b2_id))
        return pairs

    def _calculate_pairs_similarity(
        self,
        pairs: List[Tuple[str, str]],
        embedding_map: Dict[str, np.ndarray],
        logger: LoggerProtocol,
    ) -> List[AnalysisResultDTO]:
        results = []
        threshold = self._settings.similarity_threshold
        for b1_id, b2_id in pairs:
            try:
                emb1 = embedding_map.get(b1_id)
                emb2 = embedding_map.get(b2_id)
                if emb1 is None or emb2 is None:
                    logger.warning(
                        f"无法找到 {b1_id} 或 {b2_id} 的嵌入向量，跳过比较。"
                    )
                    continue
                similarity = self._adapter.calculate_pair_similarity(emb1, emb2)
                if similarity >= threshold:
                    result = AnalysisResultDTO(
                        pair_analysis_id=AnalysisResultDTO._make_id(
                            b1_id, b2_id, AnalysisType.SEMANTIC
                        ),
                        block_id_1=b1_id,
                        block_id_2=b2_id,
                        analysis_type=AnalysisType.SEMANTIC,
                        score=similarity,
                        details={
                            "model_name": self._settings.model_name_or_path,
                            "threshold": threshold,
                        },
                    )
                    results.append(result)
            except Exception as e:
                logger.error(f"计算 {b1_id} 和 {b2_id} 语义相似度时出错: {e}")
        return results
