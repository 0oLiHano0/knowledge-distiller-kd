# kd_tool/stages/semantic_analysis/semantic_analysis_stage.py (v4.6 - Schema 路径与 task_id 更新版)
# -*- coding: utf-8 -*-

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

from typing import List, Dict, Tuple # Tuple 已在原文件中
from uuid import UUID # 导入UUID，因为_compare_hashes_and_generate_results中 AnalysisResultDTO 创建时需要
from loguru import Logger
import numpy as np

# --- 核心模块导入 ---
from ....core.interfaces import StageInterface, StorageInterface
from ....core.dtos import PipelineContextDTO      # <-- [指令] 已更新
from ....schemas.dtos import ContentBlockDTO, AnalysisResultDTO # <-- [指令] 已更新 (来自中央 schemas, 已移除 task_id)
from ....schemas.enums import AnalysisType        # <-- [指令] 已更新 (来自中央 schemas)

# --- Stage 内部导入 ---
from .settings_models import SemanticAnalysisStageSettings # <-- [指令] 已更新为本地导入
from .adapter_interface import SemanticAdapterInterface
from .errors import SemanticAnalysisError, ModelLoadingError, EmbeddingCalculationError, SimilarityCalculationError # 导入所有需要的错误


class SemanticAnalysisStage(StageInterface): #
    """
    P07 - 语义分析阶段。
    负责计算内容块的语义相似度，并找出相似的内容块对。
    """

    def __init__(self,
                 logger: Logger,
                 storage: StorageInterface,
                 settings: SemanticAnalysisStageSettings, # <-- [指令] 类型已更新
                 adapter: SemanticAdapterInterface): #
        """构造函数，通过 DI 注入所有依赖。"""
        self._logger = logger.bind(stage="SemanticAnalysisStage") #
        self._storage = storage #
        self._settings = settings #
        self._adapter = adapter #
        self._is_model_loaded = False #

    def process(self, context: PipelineContextDTO) -> PipelineContextDTO: #
        """
        执行语义分析流水线阶段。
        **[指令]** 必须使用 `context.run_logger` 进行日志记录。
        **[指令]** 创建 `AnalysisResultDTO` 时 **严禁** 包含 `task_id` 字段。
        **[指令]** 调用 `storage.save_analysis_results` 时
                  **不再需要** 传递 `task_id` 参数。
        """
        run_logger = context.run_logger.bind(stage_name=self.__class__.__name__) #
        run_logger.info("开始执行语义分析阶段...") #

        if not self._settings.enabled: #
            run_logger.warning("语义分析阶段已禁用，跳过处理。") #
            return context #

        try: #
            self._ensure_model_loaded(run_logger) #

            blocks_to_process = self._get_blocks_to_process(context) #
            if len(blocks_to_process) < 2: #
                run_logger.info("没有足够的内容块进行语义比较。") #
                return context #
            
            run_logger.info(f"找到 {len(blocks_to_process)} 个内容块需要进行语义分析。") #

            block_map = {b.block_id: b for b in blocks_to_process} #
            block_ids = [b.block_id for b in blocks_to_process] #
            texts_to_embed = [b.analysis_text for b in blocks_to_process if b.analysis_text is not None and b.analysis_text.strip() != ""] # 确保 analysis_text有效
            
            valid_blocks_for_embedding = [b for b in blocks_to_process if b.analysis_text is not None and b.analysis_text.strip() != ""]
            valid_block_ids_for_embedding = [b.block_id for b in valid_blocks_for_embedding]


            if len(texts_to_embed) < 2: #
                 run_logger.info("有效内容块（含analysis_text）不足以进行比较。") #
                 return context #

            run_logger.debug(f"正在为 {len(texts_to_embed)} 个文本计算嵌入向量...") #
            embeddings = self._adapter.calculate_embeddings(texts_to_embed, self._settings.batch_size) #
            embedding_map = {block_id: emb for block_id, emb in zip(valid_block_ids_for_embedding, embeddings)} #
            run_logger.debug("嵌入向量计算完成。") #

            pairs_to_compare = self._get_pairs_to_compare(valid_block_ids_for_embedding, context, run_logger) # 使用有效ID列表
            run_logger.info(f"确定了 {len(pairs_to_compare)} 对内容块需要进行语义比较。") #

            # [指令] AnalysisResultDTO 不再需要 task_id，其创建时由 pair_analysis_id, block_ids, analysis_type 决定
            analysis_results = self._calculate_pairs_similarity( #
                pairs_to_compare, embedding_map, run_logger # 移除 task_id 参数
            )
            run_logger.info(f"生成了 {len(analysis_results)} 个语义分析结果。") #

            for result in analysis_results: #
                context.add_analysis_result(result) #
            
            if analysis_results: #
                self._save_analysis_results(analysis_results, run_logger, context) #

            run_logger.success("语义分析阶段执行完毕。") #

        except ModelLoadingError as mle: # 更具体的异常捕获
            run_logger.error(f"语义模型加载失败: {mle}", exc_info=True)
            context.add_error(mle)
        except EmbeddingCalculationError as ece:
            run_logger.error(f"嵌入向量计算失败: {ece}", exc_info=True)
            context.add_error(ece)
        except SimilarityCalculationError as sce:
            run_logger.error(f"相似度计算失败: {sce}", exc_info=True)
            context.add_error(sce)
        except Exception as e: #
            error = SemanticAnalysisError(f"语义分析阶段发生未知错误: {e}", original_exception=e) #
            run_logger.exception(error) #
            context.add_error(error) #

        return context #

    def _ensure_model_loaded(self, logger: Logger): #
        if not self._is_model_loaded: #
            logger.info(f"正在加载语义模型: {self._settings.model_name_or_path}...") #
            # [指令] 适配器的 load_model 方法应处理 ModelLoadingError
            self._adapter.load_model(self._settings.model_name_or_path, self._settings.device) #
            self._is_model_loaded = True #
            logger.info("语义模型加载成功。") #

    def _get_blocks_to_process(self, context: PipelineContextDTO) -> List[ContentBlockDTO]: #
        # [指令] 只选择包含有效 analysis_text 的块
        return [b for b in context.content_blocks.values() if b.analysis_text and b.analysis_text.strip() != ""]

    def _get_pairs_to_compare(self,
                              block_ids: List[str],
                              context: PipelineContextDTO,
                              logger: Logger) -> List[Tuple[str, str]]: #
        # ... (内部逻辑基本不变, 确保 AnalysisResultDTO._calculate_pair_analysis_id 调用正确)
        pairs = [] #
        n = len(block_ids) #

        if self._settings.comparison_strategy == "all_pairs": #
            logger.warning("使用 'all_pairs' 策略进行语义比较，计算量可能很大！") #
            for i in range(n): #
                for j in range(i + 1, n): #
                    pairs.append((block_ids[i], block_ids[j])) #
            return pairs #

        logger.info("使用 'pre_filtered' 策略筛选语义比较对...") #
        existing_matches_pair_ids = set() #
        
        for block_id_current in block_ids: #
            if block_id_current in context.analysis_results: #
                for analysis_type in [AnalysisType.MD5, AnalysisType.SIMHASH]: #
                    if analysis_type in context.analysis_results[block_id_current]: #
                        for result in context.analysis_results[block_id_current][analysis_type]: #
                             # [指令] 仅当分数表明高度相似时才过滤。
                             # MD5 匹配 (score=1.0)
                             if result.analysis_type == AnalysisType.MD5 and result.score == 1.0: #
                                 existing_matches_pair_ids.add(result.pair_analysis_id) #
                             # SimHash 相似度高于阈值 (注意：SimHash 的 score 是 1 - (dist/bits))
                             # Semantic Stage 不应直接依赖 SimHash 的具体配置，
                             # 但可以有一个通用概念，如 SimHash 结果的 score > 0.9 (假设高相似)
                             elif result.analysis_type == AnalysisType.SIMHASH and result.score is not None and result.score >= self._settings.similarity_threshold : # 或一个独立的 prefilter_threshold
                                 existing_matches_pair_ids.add(result.pair_analysis_id) #
        for i in range(n): #
            for j in range(i + 1, n): #
                b1_id, b2_id = block_ids[i], block_ids[j] #
                # 为当前对计算一个临时的 pair_analysis_id 以便在 existing_matches_pair_ids 中检查
                # 注意：这只是一种检查方式，如果 AnalysisResultDTO.pair_analysis_id 的生成方式涉及 task_id，
                # 而 task_id 已从 DTO 中移除，则需要确保此处的ID生成逻辑与存储中的ID一致。
                # 当前 _calculate_pair_analysis_id 不涉及 task_id，所以是安全的。
                temp_pair_id_md5 = AnalysisResultDTO._calculate_pair_analysis_id(b1_id, b2_id, AnalysisType.MD5) #
                temp_pair_id_simhash = AnalysisResultDTO._calculate_pair_analysis_id(b1_id, b2_id, AnalysisType.SIMHASH) #

                if temp_pair_id_md5 not in existing_matches_pair_ids and \
                   temp_pair_id_simhash not in existing_matches_pair_ids: #
                    pairs.append((b1_id, b2_id)) #
        return pairs #

    def _calculate_pairs_similarity(self,
                                  pairs: List[Tuple[str, str]],
                                  embedding_map: Dict[str, np.ndarray],
                                  # task_id: UUID, # <-- [指令] 移除 task_id 参数
                                  logger: Logger) -> List[AnalysisResultDTO]: #
        results = [] #
        threshold = self._settings.similarity_threshold #

        for b1_id, b2_id in pairs: #
            try: #
                emb1 = embedding_map.get(b1_id) #
                emb2 = embedding_map.get(b2_id) #

                if emb1 is None or emb2 is None: #
                    logger.warning(f"无法找到 {b1_id} 或 {b2_id} 的嵌入向量，跳过比较。") #
                    continue #

                similarity = self._adapter.calculate_pair_similarity(emb1, emb2) #

                if similarity >= threshold: #
                    result = AnalysisResultDTO( #
                        block_id_1=b1_id, #
                        block_id_2=b2_id, #
                        analysis_type=AnalysisType.SEMANTIC, #
                        score=similarity, #
                        details={ #
                            "model_name": self._settings.model_name_or_path, #
                            "threshold": threshold #
                        }
                        # task_id 字段已从 DTO 移除
                    )
                    results.append(result) #
            except Exception as e: #
                logger.error(f"计算 {b1_id} 和 {b2_id} 语义相似度时出错: {e}") #
                # 可以在此处将错误添加到 context (通过包装成 SemanticAnalysisError)
                # context.add_error(SimilarityCalculationError(f"Failed for pair {b1_id}-{b2_id}", original_exception=e))
        
        return results #

    def _save_analysis_results(self,
                               results: List[AnalysisResultDTO],
                               logger: Logger,
                               context: PipelineContextDTO) -> None: #
        """将分析结果持久化到存储。"""
        logger.info(f"正在将 {len(results)} 个语义分析结果保存到存储...") #
        try: #
            # [指令] StorageInterface.save_analysis_results 不再需要 task_id
            self._storage.save_analysis_results(results) #
            logger.debug(f"成功保存 {len(results)} 个分析结果。") #
        except Exception as e: #
            err = SemanticAnalysisError(f"保存语义分析结果到存储时失败: {e}", original_exception=e) #
            logger.error(err) #
            context.add_error(err) #