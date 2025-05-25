```python

# kd_tool/stages/semantic_analysis/semantic_analysis_stage.py
# -*- coding: utf-8 -*-

"""
=================================================
semantic_analysis_stage.py - P07 语义分析阶段实现 (v4.5)
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
from loguru import Logger
import numpy as np

from ...core.interfaces import StageInterface, StorageInterface
from ...schemas.dtos import PipelineContextDTO, ContentBlockDTO, AnalysisResultDTO
from ...schemas.settings_models import SemanticAnalysisStageSettings
from ...schemas.enums import AnalysisType
from .adapter_interface import SemanticAdapterInterface
from .errors import SemanticAnalysisError

class SemanticAnalysisStage(StageInterface):
    """
    P07 - 语义分析阶段。
    负责计算内容块的语义相似度，并找出相似的内容块对。
    """

    def __init__(self,
                 logger: Logger,
                 storage: StorageInterface,
                 settings: SemanticAnalysisStageSettings,
                 adapter: SemanticAdapterInterface):
        """构造函数，通过 DI 注入所有依赖。"""
        self._logger = logger.bind(stage="SemanticAnalysisStage")
        self._storage = storage
        self._settings = settings
        self._adapter = adapter
        self._is_model_loaded = False # 标记模型是否已加载

    def process(self, context: PipelineContextDTO) -> PipelineContextDTO:
        """执行语义分析流水线阶段。"""
        run_logger = context.run_logger.bind(stage="SemanticAnalysisStage")
        run_logger.info("开始执行语义分析阶段...")

        if not self._settings.enabled:
            run_logger.warning("语义分析阶段已禁用，跳过处理。")
            return context

        try:
            # 1. 确保模型已加载
            self._ensure_model_loaded(run_logger)

            # 2. 获取需要处理的内容块
            blocks_to_process = self._get_blocks_to_process(context)
            if len(blocks_to_process) < 2:
                run_logger.info("没有足够的内容块进行语义比较。")
                return context
            
            run_logger.info(f"找到 {len(blocks_to_process)} 个内容块需要进行语义分析。")

            # 3. 计算所有块的嵌入向量 (批量)
            block_map = {b.block_id: b for b in blocks_to_process}
            block_ids = [b.block_id for b in blocks_to_process]
            texts = [b.analysis_text for b in blocks_to_process if b.analysis_text]
            
            if len(texts) != len(blocks_to_process):
                run_logger.warning("部分内容块缺少 `analysis_text`，将被跳过。")
                # 需要重新构建 block_ids 和 block_map 以匹配 texts
                valid_blocks = [b for b in blocks_to_process if b.analysis_text]
                block_ids = [b.block_id for b in valid_blocks]
                block_map = {b.block_id: b for b in valid_blocks}

            if len(block_ids) < 2:
                 run_logger.info("有效内容块不足以进行比较。")
                 return context

            run_logger.debug(f"正在为 {len(texts)} 个文本计算嵌入向量...")
            embeddings = self._adapter.calculate_embeddings(texts, self._settings.batch_size)
            embedding_map = {block_id: emb for block_id, emb in zip(block_ids, embeddings)}
            run_logger.debug("嵌入向量计算完成。")

            # 4. 确定需要比较的块对 (根据策略)
            pairs_to_compare = self._get_pairs_to_compare(block_ids, context, run_logger)
            run_logger.info(f"确定了 {len(pairs_to_compare)} 对内容块需要进行语义比较。")

            # 5. 逐对计算相似度并生成结果
            analysis_results = self._calculate_pairs_similarity(
                pairs_to_compare, embedding_map, context.task_id, run_logger
            )
            run_logger.info(f"生成了 {len(analysis_results)} 个语义分析结果。")

            # 6. 添加结果到上下文并更新存储
            for result in analysis_results:
                context.add_analysis_result(result)
            
            if analysis_results:
                self._save_analysis_results(analysis_results, run_logger, context)

            run_logger.success("语义分析阶段执行完毕。")

        except Exception as e:
            error = SemanticAnalysisError(f"语义分析阶段发生未知错误: {e}")
            run_logger.exception(error)
            context.add_error(error)

        return context

    def _ensure_model_loaded(self, logger: Logger):
        """确保语义模型已加载，如果未加载则加载。"""
        if not self._is_model_loaded:
            logger.info(f"正在加载语义模型: {self._settings.model_name_or_path}...")
            self._adapter.load_model(self._settings.model_name_or_path, self._settings.device)
            self._is_model_loaded = True
            logger.info("语义模型加载成功。")

    def _get_blocks_to_process(self, context: PipelineContextDTO) -> List[ContentBlockDTO]:
        """获取所有需要处理的内容块。"""
        # 简化: 目前返回所有块，比较策略在 _get_pairs_to_compare 中处理。
        return list(context.content_blocks.values())

    def _get_pairs_to_compare(self, 
                              block_ids: List[str], 
                              context: PipelineContextDTO, 
                              logger: Logger) -> List[Tuple[str, str]]:
        """根据比较策略，确定需要进行语义比较的块对。"""
        pairs = []
        n = len(block_ids)

        if self._settings.comparison_strategy == "all_pairs":
            logger.warning("使用 'all_pairs' 策略，计算量可能很大！")
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((block_ids[i], block_ids[j]))
            return pairs

        # "pre_filtered" 策略
        logger.info("使用 'pre_filtered' 策略筛选比较对...")
        existing_matches = set()
        
        # 收集所有 MD5 和 SimHash 匹配的对 (pair_analysis_id)
        for block_id in block_ids:
            if block_id in context.analysis_results:
                for analysis_type in [AnalysisType.MD5, AnalysisType.SIMHASH]:
                    if analysis_type in context.analysis_results[block_id]:
                        for result in context.analysis_results[block_id][analysis_type]:
                             # 只有当分数表示高度相似时才过滤 (MD5=1.0, SimHash>threshold)
                             # MD5 结果的 score 就是 1.0 (匹配)
                             if result.analysis_type == AnalysisType.MD5 and result.score == 1.0:
                                 existing_matches.add(result.pair_analysis_id)
                             # SimHash 结果需要检查是否高于阈值 (我们存储的是相似度)
                             elif result.analysis_type == AnalysisType.SIMHASH:
                                 # 我们需要从 settings 获取 simhash 阈值来计算相似度阈值
                                 #  或者，我们可以直接认为任何 SimHash 结果都过滤掉，
                                 #  因为 SimHash 本身就是粗筛。此处采用后者简化。
                                 existing_matches.add(result.pair_analysis_id)

        # 生成所有可能的对，并排除已匹配的对
        for i in range(n):
            for j in range(i + 1, n):
                b1_id, b2_id = block_ids[i], block_ids[j]
                # 计算这个对的 MD5 和 SimHash pair_id
                md5_pair_id = AnalysisResultDTO._calculate_pair_analysis_id(b1_id, b2_id, AnalysisType.MD5)
                simhash_pair_id = AnalysisResultDTO._calculate_pair_analysis_id(b1_id, b2_id, AnalysisType.SIMHASH)

                # 如果这个对没有被 MD5 或 SimHash 匹配，则添加到待比较列表
                if md5_pair_id not in existing_matches and simhash_pair_id not in existing_matches:
                    pairs.append((b1_id, b2_id))

        return pairs

    def _calculate_pairs_similarity(self, 
                                  pairs: List[Tuple[str, str]], 
                                  embedding_map: Dict[str, np.ndarray],
                                  task_id: UUID,
                                  logger: Logger) -> List[AnalysisResultDTO]:
        """逐对计算相似度并生成结果。"""
        results = []
        threshold = self._settings.similarity_threshold

        for b1_id, b2_id in pairs:
            try:
                emb1 = embedding_map.get(b1_id)
                emb2 = embedding_map.get(b2_id)

                if emb1 is None or emb2 is None:
                    logger.warning(f"无法找到 {b1_id} 或 {b2_id} 的嵌入向量，跳过比较。")
                    continue

                similarity = self._adapter.calculate_pair_similarity(emb1, emb2)

                if similarity >= threshold:
                    result = AnalysisResultDTO(
                        block_id_1=b1_id,
                        block_id_2=b2_id,
                        analysis_type=AnalysisType.SEMANTIC,
                        score=similarity,
                        details={
                            "model_name": self._settings.model_name_or_path,
                            "threshold": threshold
                        },
                        task_id=task_id
                    )
                    results.append(result)
            except Exception as e:
                logger.error(f"计算 {b1_id} 和 {b2_id} 语义相似度时出错: {e}")
                # 可以在此处添加错误到 context
        
        return results

    def _save_analysis_results(self, 
                               results: List[AnalysisResultDTO], 
                               logger: Logger, 
                               context: PipelineContextDTO) -> None:
        """将分析结果持久化到存储。"""
        logger.info(f"正在将 {len(results)} 个语义分析结果保存到存储...")
        try:
            self._storage.save_analysis_results(results, context.task_id)
            logger.debug(f"成功保存 {len(results)} 个分析结果。")
        except Exception as e:
            err = SemanticAnalysisError(f"保存语义分析结果到存储时失败: {e}")
            logger.error(err)
            context.add_error(err)

```