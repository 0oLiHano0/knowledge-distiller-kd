# kd_tool/stages/decision/decision_stage.py (v4.6 - Schema 路径与 task_id 更新版)
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

from typing import List, Dict, Set, Tuple, Optional, Any # Any 用于 _check_rule_match 中的 rule 类型
from uuid import UUID # 导入 UUID
from loguru import Logger

# --- 核心模块导入 ---
from ....core.interfaces import StageInterface, StorageInterface
from ....core.dtos import PipelineContextDTO      # <-- [指令] 已更新
from ....schemas.dtos import AnalysisResultDTO, UserDecisionDTO # <-- [指令] 已更新 (来自中央 schemas, 已移除 task_id)
from ....schemas.enums import AnalysisType, DecisionType        # <-- [指令] 已更新 (来自中央 schemas)

# --- Stage 内部导入 ---
from .settings_models import DecisionStageSettings, DecisionRule # <-- [指令] 已更新为本地导入
from .errors import DecisionError, RuleEvaluationError, MissingAnalysisDataError # <-- [指令] 本地错误导入


class DecisionStage(StageInterface): #
    """
    P08 - 决策阶段。
    负责根据分析结果生成决策建议。
    """

    def __init__(self,
                 logger: Logger,
                 storage: StorageInterface,
                 settings: DecisionStageSettings): # <-- [指令] 类型已更新
        """构造函数，通过 DI 注入所有依赖。"""
        self._logger = logger.bind(stage="DecisionStage") #
        self._storage = storage #
        self._settings = settings #
        self._sorted_rules = sorted( #
            self._settings.rules,
            key=lambda r: r.rule_priority,
            reverse=True
        )

    def process(self, context: PipelineContextDTO) -> PipelineContextDTO: #
        """
        执行决策流水线阶段。
        **[指令]** 必须使用 `context.run_logger` 进行日志记录。
        **[指令]** 创建 `UserDecisionDTO` 时 **严禁** 包含 `task_id` 字段。
        **[指令]** 调用 `storage.save_user_decisions` (后续阶段6会修改此方法为批量)。
        """
        run_logger = context.run_logger.bind(stage_name=self.__class__.__name__) #
        run_logger.info("开始执行决策阶段...") #

        if not self._settings.enabled: #
            run_logger.warning("决策阶段已禁用，跳过处理。") #
            return context #

        try: #
            unique_pairs = self._get_unique_pairs_with_results(context) #
            if not unique_pairs: #
                run_logger.info("没有找到需要进行决策的分析结果对。") #
                return context #
            
            run_logger.info(f"找到 {len(unique_pairs)} 对独特的块进行决策评估。") #

            decisions_to_add: List[UserDecisionDTO] = [] #
            for (b1_id, b2_id), results_map in unique_pairs.items(): #
                try: #
                    decision_type_to_apply = self._apply_rules(results_map, run_logger) #
                    
                    # [指令] 根据 process_undecided 设置决定是否处理 UNDECIDED
                    if decision_type_to_apply != DecisionType.KEEP or \
                       (decision_type_to_apply == DecisionType.UNDECIDED and self._settings.process_undecided) or \
                       (decision_type_to_apply != DecisionType.UNDECIDED and decision_type_to_apply != DecisionType.KEEP) : # 明确的DELETE或IGNORE_PAIR也处理
                         
                         representative_result = self._get_representative_result(results_map) #
                         if representative_result: #
                            user_decision = UserDecisionDTO( #
                                pair_analysis_id=representative_result.pair_analysis_id, #
                                decision=decision_type_to_apply, #
                                decided_by="system_rules_v1", #
                                notes=f"基于 {len(self._settings.rules)} 条规则自动生成。" #
                                # task_id 字段已从 DTO 移除
                            )
                            decisions_to_add.append(user_decision) #
                         else: #
                             run_logger.warning(f"无法为块对 ({b1_id}, {b2_id}) 找到代表性分析结果，无法生成决策。") #

                except RuleEvaluationError as ree: # 捕获我们定义的具体错误
                    run_logger.error(ree) #
                    context.add_error(ree) #

                except Exception as e: #
                    err = RuleEvaluationError( #
                        pair_id=f"{b1_id}-{b2_id}", #
                        rule="<multiple_during_application>", #
                        original_error=e #
                    )
                    run_logger.error(err) #
                    context.add_error(err) #
            
            run_logger.info(f"生成了 {len(decisions_to_add)} 个用户决策。") #

            for decision_dto in decisions_to_add: #
                context.add_user_decision(decision_dto) #
                
            if decisions_to_add: #
                # [指令] _save_user_decisions 方法将在阶段6中更新为批量处理
                self._save_user_decisions(decisions_to_add, run_logger, context) #

            run_logger.success("决策阶段执行完毕。") #

        except MissingAnalysisDataError as mae: # 更具体的错误捕获
            run_logger.error(f"决策阶段因缺少分析数据而中止: {mae}", exc_info=True)
            context.add_error(mae)
        except Exception as e: #
            error = DecisionError(f"决策阶段发生未知错误: {e}", original_exception=e) #
            run_logger.exception(error) #
            context.add_error(error) #

        return context #

    def _get_unique_pairs_with_results(self, context: PipelineContextDTO) -> Dict[Tuple[str, str], Dict[AnalysisType, AnalysisResultDTO]]: #
        # ... (内部逻辑基本不变) ...
        pairs: Dict[Tuple[str, str], Dict[AnalysisType, AnalysisResultDTO]] = {} #
        if not context.analysis_results: # 添加检查，如果 analysis_results 为空
            return pairs
        for block_id, analyses in context.analysis_results.items(): #
            for analysis_type, results_list in analyses.items(): #
                for result in results_list: #
                    pair_key = tuple(sorted((result.block_id_1, result.block_id_2))) #
                    if pair_key not in pairs: #
                        pairs[pair_key] = {} #
                    if analysis_type not in pairs[pair_key]: #
                         pairs[pair_key][analysis_type] = result #
        return pairs #

    def _get_representative_result(self,
                                 results_map: Dict[AnalysisType, AnalysisResultDTO]
                                 ) -> Optional[AnalysisResultDTO]: #
        # ... (内部逻辑不变) ...
        if AnalysisType.MD5 in results_map: #
            return results_map[AnalysisType.MD5] #
        if AnalysisType.SIMHASH in results_map: #
            return results_map[AnalysisType.SIMHASH] #
        if AnalysisType.SEMANTIC in results_map: #
            return results_map[AnalysisType.SEMANTIC] #
        return None #

    def _apply_rules(self,
                     results_map: Dict[AnalysisType, AnalysisResultDTO],
                     logger: Logger) -> DecisionType: #
        # ... (内部逻辑基本不变) ...
        for rule in self._sorted_rules: #
            if self._check_rule_match(rule, results_map): #
                logger.trace(f"块对匹配规则 (Prio: {rule.rule_priority}): {rule.decision_to_apply.value}") #
                return rule.decision_to_apply #
        logger.trace("块对未匹配任何规则，应用默认决策。") #
        return self._settings.default_decision #

    def _check_rule_match(self,
                          rule: DecisionRule, # 类型应为 DecisionRule
                          results_map: Dict[AnalysisType, AnalysisResultDTO]) -> bool: #
        # ... (内部逻辑基本不变) ...
        if rule.md5_score is not None: #
            md5_result = results_map.get(AnalysisType.MD5) #
            if md5_result is None or md5_result.score is None or md5_result.score != rule.md5_score: # 增加对 score is None 的检查
                return False #
        if rule.simhash_similarity_min is not None: #
            simhash_result = results_map.get(AnalysisType.SIMHASH) #
            if simhash_result is None or simhash_result.score is None or simhash_result.score < rule.simhash_similarity_min: #
                return False #
        if rule.semantic_similarity_min is not None: #
            semantic_result = results_map.get(AnalysisType.SEMANTIC) #
            if semantic_result is None or semantic_result.score is None or semantic_result.score < rule.semantic_similarity_min: #
                return False #
        return True #

    def _save_user_decisions(self,
                             decisions: List[UserDecisionDTO],
                             logger: Logger,
                             context: PipelineContextDTO) -> None: #
        """将用户决策持久化到存储。"""
        # [指令] 此方法将在阶段6中更新为调用 StorageInterface 的批量保存方法
        logger.info(f"正在将 {len(decisions)} 个用户决策保存到存储...") #
        try: #
            # [指令] StorageInterface.save_user_decisions 不再需要 task_id
            # self._storage.save_user_decisions(decisions, context.task_id) # 旧调用
            self._storage.save_user_decisions(decisions) # 新调用 (阶段6会修改接口和实现)
            logger.debug(f"成功保存 {len(decisions)} 个用户决策。") #
        except Exception as e: #
            err = DecisionError(f"保存用户决策到存储时失败: {e}", original_exception=e) #
            logger.error(err) #
            context.add_error(err) #