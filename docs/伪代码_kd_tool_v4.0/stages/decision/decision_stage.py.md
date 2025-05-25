```python

# kd_tool/stages/decision/decision_stage.py
# -*- coding: utf-8 -*-

"""
=================================================
decision_stage.py - P08 决策阶段实现 (v4.6)
=================================================

**模块功能**:

- 实现 `StageInterface`，执行决策流程。
- **职责**:
    1. 收集所有 `AnalysisResultDTO`。
    2. 对每一个独特的块对，根据 `DecisionStageSettings` 中的规则进行评估。
    3. 生成 `UserDecisionDTO`。
    4. 将 `UserDecisionDTO` 添加到 `PipelineContextDTO` 和存储中。
- **规范**:
    - **必须**通过 DI 注入 `Logger`、`StorageInterface` 和 `DecisionStageSettings`。
    - **必须**处理 `DecisionError` 并记录。
    - 决策逻辑**必须**清晰且可测试。

---
"""

from typing import List, Dict, Set, Tuple, Optional
from loguru import Logger
from uuid import UUID

from ...core.interfaces import StageInterface, StorageInterface
from ...schemas.dtos import PipelineContextDTO, AnalysisResultDTO, UserDecisionDTO
from ...schemas.settings_models import DecisionStageSettings, DecisionRule
from ...schemas.enums import AnalysisType, DecisionType
from .errors import DecisionError, RuleEvaluationError, MissingAnalysisDataError

class DecisionStage(StageInterface):
    """
    P08 - 决策阶段。
    负责根据分析结果生成决策建议。
    """

    def __init__(self,
                 logger: Logger,
                 storage: StorageInterface,
                 settings: DecisionStageSettings):
        """构造函数，通过 DI 注入所有依赖。"""
        self._logger = logger.bind(stage="DecisionStage")
        self._storage = storage
        self._settings = settings
        # 对规则按优先级排序，高的在前
        self._sorted_rules = sorted(
            self._settings.rules, 
            key=lambda r: r.rule_priority, 
            reverse=True
        )

    def process(self, context: PipelineContextDTO) -> PipelineContextDTO:
        """执行决策流水线阶段。"""
        run_logger = context.run_logger.bind(stage="DecisionStage")
        run_logger.info("开始执行决策阶段...")

        if not self._settings.enabled:
            run_logger.warning("决策阶段已禁用，跳过处理。")
            return context

        try:
            # 1. 收集所有唯一的分析结果对
            unique_pairs = self._get_unique_pairs_with_results(context)
            if not unique_pairs:
                run_logger.info("没有找到需要进行决策的分析结果对。")
                return context
            
            run_logger.info(f"找到 {len(unique_pairs)} 对独特的块进行决策评估。")

            # 2. 为每个对应用规则并生成决策
            decisions_to_add = []
            for (b1_id, b2_id), results_map in unique_pairs.items():
                try:
                    decision = self._apply_rules(results_map, run_logger)
                    
                    # 如果有决策，并且不是 'KEEP' (或需要处理 UNDECIDED)
                    if decision != DecisionType.KEEP or self._settings.process_undecided:
                         # 我们需要一个 pair_analysis_id。由于一个块对可能有多个分析结果，
                         # 我们需要选择一个作为代表，或者创建一个新的决策 ID。
                         # **决策**: 我们将基于第一个分析结果的 pair_id 来创建 UserDecisionDTO。
                         #            如果一个决策覆盖了多个分析结果，这可能需要注意。
                         #            更好的方法可能是 UserDecisionDTO 直接关联 (b1_id, b2_id)。
                         #            【v4.6 架构决策】让 UserDecisionDTO 关联到 *代表性* 的
                         #            pair_analysis_id，通常是优先级最高的那个分析结果的 ID。
                         #            或者，如果 UserDecisionDTO 可以不关联 pair_id，
                         #            而是直接关联 (b1, b2) 会更干净。
                         #            当前 DTO 设计是关联 pair_id。我们先用 MD5 > SimHash > Semantic 的 ID。
                         
                         representative_result = self._get_representative_result(results_map)
                         if representative_result:
                            user_decision = UserDecisionDTO(
                                pair_analysis_id=representative_result.pair_analysis_id,
                                decision=decision,
                                decided_by="system_rules_v1", # 标记为系统决策
                                notes=f"基于 {len(self._settings.rules)} 条规则自动生成。",
                                task_id=context.task_id
                            )
                            decisions_to_add.append(user_decision)
                         else:
                             run_logger.warning(f"无法为块对 ({b1_id}, {b2_id}) 找到代表性分析结果。")

                except Exception as e:
                    err = RuleEvaluationError(
                        pair_id=f"{b1_id}-{b2_id}", 
                        rule="<multiple>", 
                        original_error=e
                    )
                    run_logger.error(err)
                    context.add_error(err)
            
            run_logger.info(f"生成了 {len(decisions_to_add)} 个用户决策。")

            # 3. 添加决策到上下文并保存
            for decision in decisions_to_add:
                context.add_user_decision(decision)
                
            if decisions_to_add:
                self._save_user_decisions(decisions_to_add, run_logger, context)

            run_logger.success("决策阶段执行完毕。")

        except Exception as e:
            error = DecisionError(f"决策阶段发生未知错误: {e}")
            run_logger.exception(error)
            context.add_error(error)

        return context

    def _get_unique_pairs_with_results(self, context: PipelineContextDTO) -> Dict[Tuple[str, str], Dict[AnalysisType, AnalysisResultDTO]]:
        """
        从 context 中收集所有唯一的块对及其对应的所有分析结果。
        返回一个字典，键是排序后的 (block_id_1, block_id_2) 元组，
        值是另一个字典，键是 AnalysisType，值是对应的 AnalysisResultDTO。
        """
        pairs: Dict[Tuple[str, str], Dict[AnalysisType, AnalysisResultDTO]] = {}
        
        for block_id, analyses in context.analysis_results.items():
            for analysis_type, results_list in analyses.items():
                for result in results_list:
                    # 确保块对顺序一致
                    pair_key = tuple(sorted((result.block_id_1, result.block_id_2)))
                    
                    if pair_key not in pairs:
                        pairs[pair_key] = {}
                        
                    # 只存储每个类型一个结果 (通常只有一个)
                    # 如果有多个 (不应该发生)，则取第一个
                    if analysis_type not in pairs[pair_key]:
                         pairs[pair_key][analysis_type] = result
                         
        return pairs

    def _get_representative_result(self, 
                                 results_map: Dict[AnalysisType, AnalysisResultDTO]
                                 ) -> Optional[AnalysisResultDTO]:
        """选择一个代表性的 AnalysisResultDTO 以获取 pair_analysis_id。"""
        if AnalysisType.MD5 in results_map:
            return results_map[AnalysisType.MD5]
        if AnalysisType.SIMHASH in results_map:
            return results_map[AnalysisType.SIMHASH]
        if AnalysisType.SEMANTIC in results_map:
            return results_map[AnalysisType.SEMANTIC]
        return None


    def _apply_rules(self, 
                     results_map: Dict[AnalysisType, AnalysisResultDTO], 
                     logger: Logger) -> DecisionType:
        """
        对单个块对的分析结果应用排序后的规则。
        返回第一个匹配的规则所指定的决策。
        """
        for rule in self._sorted_rules:
            if self._check_rule_match(rule, results_map):
                logger.trace(f"块对匹配规则 (Prio: {rule.rule_priority}): {rule.decision_to_apply}")
                return rule.decision_to_apply
        
        logger.trace("块对未匹配任何规则，应用默认决策。")
        return self._settings.default_decision

    def _check_rule_match(self, 
                          rule: DecisionRule, 
                          results_map: Dict[AnalysisType, AnalysisResultDTO]) -> bool:
        """检查单个规则是否与给定的分析结果匹配。"""
        
        # 检查 MD5 条件
        if rule.md5_score is not None:
            md5_result = results_map.get(AnalysisType.MD5)
            if md5_result is None or md5_result.score != rule.md5_score:
                return False # MD5 条件不满足

        # 检查 SimHash 条件
        if rule.simhash_similarity_min is not None:
            simhash_result = results_map.get(AnalysisType.SIMHASH)
            if simhash_result is None or simhash_result.score is None or simhash_result.score < rule.simhash_similarity_min:
                return False # SimHash 条件不满足

        # 检查 Semantic 条件
        if rule.semantic_similarity_min is not None:
            semantic_result = results_map.get(AnalysisType.SEMANTIC)
            if semantic_result is None or semantic_result.score is None or semantic_result.score < rule.semantic_similarity_min:
                return False # Semantic 条件不满足

        # 如果所有（非 None）条件都满足，则规则匹配
        return True


    def _save_user_decisions(self, 
                             decisions: List[UserDecisionDTO], 
                             logger: Logger, 
                             context: PipelineContextDTO) -> None:
        """将用户决策持久化到存储。"""
        logger.info(f"正在将 {len(decisions)} 个用户决策保存到存储...")
        try:
            # 规范: 存储接口应支持批量保存 UserDecisionDTO。
            self._storage.save_user_decisions(decisions, context.task_id)
            logger.debug(f"成功保存 {len(decisions)} 个用户决策。")
        except Exception as e:
            err = DecisionError(f"保存用户决策到存储时失败: {e}")
            logger.error(err)
            context.add_error(err)


```