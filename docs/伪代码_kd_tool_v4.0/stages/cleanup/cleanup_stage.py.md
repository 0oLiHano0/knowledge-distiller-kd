```python

# kd_tool/stages/cleanup/cleanup_stage.py
# -*- coding: utf-8 -*-

"""
=================================================
cleanup_stage.py - P09 清理阶段实现 (v4.7)
=================================================

**模块功能**:

- 实现 `StageInterface`，执行清理流程。
- **职责**:
    1. 读取 `UserDecisionDTO`。
    2. 确定哪些文件需要执行操作。
    3. 根据 `CleanupStageSettings` 执行操作 (标记、移动、删除)。
    4. 更新 `FileRecordDTO` 状态。
- **架构挑战**: 将基于 *块对* 的决策转换为基于 *文件* 的操作。
               **初步策略**: 如果一个文件的 *所有* 内容块都被标记为 DELETE (或与被标记为 DELETE 的块高度相似)，
                            则该文件可以被清理。如果一个文件包含 KEEP 和 DELETE 块，则不能删除文件。
                            本伪代码将简化此逻辑，主要演示流程。

---
"""

from typing import List, Dict, Set
from loguru import Logger
from pathlib import Path

from ...core.interfaces import StageInterface, StorageInterface
from ...schemas.dtos import PipelineContextDTO, UserDecisionDTO, FileRecordDTO
from ...schemas.settings_models import CleanupStageSettings
from ...schemas.enums import DecisionType, ProcessingStatus
from .adapter_interface import FileSystemAdapterInterface
from .errors import CleanupError, DecisionResolutionError, FileOperationError
from ..decision.decision_stage import DecisionStage # 可能需要 DecisionStage 的逻辑或 DTO

class CleanupStage(StageInterface):
    """
    P09 - 清理阶段。
    负责执行决策结果。
    """

    def __init__(self,
                 logger: Logger,
                 storage: StorageInterface,
                 settings: CleanupStageSettings,
                 fs_adapter: FileSystemAdapterInterface):
        """构造函数，通过 DI 注入所有依赖。"""
        self._logger = logger.bind(stage="CleanupStage")
        self._storage = storage
        self._settings = settings
        self._fs_adapter = fs_adapter

    def process(self, context: PipelineContextDTO) -> PipelineContextDTO:
        """执行清理流水线阶段。"""
        run_logger = context.run_logger.bind(stage="CleanupStage")
        run_logger.info("开始执行清理阶段...")

        if not self._settings.enabled:
            run_logger.warning("清理阶段已禁用，跳过处理。")
            return context

        try:
            # 1. 获取所有决策
            decisions = list(context.user_decisions.values())
            if not decisions:
                run_logger.info("没有用户决策需要处理。")
                return context

            # 2. **核心难点**: 确定要删除的文件
            #    这需要一个复杂的逻辑：
            #    a. 找出所有 `DecisionType.DELETE` 的决策。
            #    b. 找到这些决策涉及的所有 `block_id`。
            #    c. 找到这些 `block_id` 对应的 `file_id`。
            #    d. 对于每个 `file_id`，检查其 *所有* `block_id` 是否都应该被删除。
            #    e. 只有当一个文件的 *所有* 块都可删除时，才将该文件标记为可删除。
            #    **伪代码简化**: 我们假设能得到一个 `files_to_delete_ids` 列表。
            files_to_process = self._resolve_files_from_decisions(context, run_logger)

            # 3. 执行操作
            updated_records = []
            for file_id, action in files_to_process.items():
                record = context.file_records.get(file_id)
                if not record:
                    run_logger.warning(f"找不到文件记录 {file_id}，无法执行清理。")
                    continue

                try:
                    new_status = self._execute_action(record, action, run_logger)
                    if new_status:
                        record.processing_status = new_status
                        updated_records.append(record)
                except Exception as e:
                    err = CleanupError(f"处理文件 {file_id} ({record.original_path}) 时出错: {e}")
                    run_logger.error(err)
                    context.add_error(err)
                    record.processing_status = ProcessingStatus.CLEANUP_FAILED
                    updated_records.append(record)

            # 4. 更新存储
            if updated_records:
                run_logger.info(f"正在将 {len(updated_records)} 个文件记录的状态更新到存储...")
                # **规范**: StorageInterface 需要支持批量更新 FileRecordDTO 状态。
                self._storage.save_file_records(updated_records, context.task_id) # 假设 save_file_records 支持更新

            run_logger.success("清理阶段执行完毕。")

        except Exception as e:
            error = CleanupError(f"清理阶段发生未知错误: {e}")
            run_logger.exception(error)
            context.add_error(error)

        return context

    def _resolve_files_from_decisions(self, 
                                    context: PipelineContextDTO, 
                                    logger: Logger) -> Dict[str, str]:
        """
        【关键且复杂的逻辑 - 伪代码简化】
        根据 UserDecisionDTO 决定每个文件的最终清理动作。
        返回一个字典 {file_id: action_string}。
        """
        logger.debug("正在解析决策以确定文件操作...")
        file_actions: Dict[str, str] = {}
        
        # 1. 找出所有决定为 DELETE 的块
        delete_block_ids: Set[str] = set()
        for decision in context.user_decisions.values():
            action = self._settings.action_map.get(decision.decision, 'ignore')
            if action != 'ignore': # 我们只关心需要操作的决策
                 # 需要从 pair_analysis_id 找到 block_ids
                 # 这需要反向查找或更好的上下文结构。
                 # **简化**: 假设我们能找到与 DELETE 相关的 block_id。
                 # 在真实实现中，可能需要遍历 AnalysisResultDTO 来找到 block_id。
                 # 这里我们假设 DecisionStage 已经将 DELETE 决策关联到了具体的 block_id。
                 # 或者我们直接从 context.user_decisions 找出所有 DELETE 决策。
                 if decision.decision == DecisionType.DELETE:
                     # 尝试找到对应的 AnalysisResultDTO
                     result = next((r for analyses in context.analysis_results.values() 
                                      for results_list in analyses.values() 
                                      for r in results_list if r.pair_analysis_id == decision.pair_analysis_id), None)
                     if result:
                         # **重要**: 我们需要决定删除哪个。通常是第二个 (block_id_2)。
                         # 这是一个巨大的简化！真实决策可能更复杂。
                         delete_block_ids.add(result.block_id_2) 
                         
        # 2. 检查每个文件
        for file_id, file_record in context.file_records.items():
            file_blocks = [b for b in context.content_blocks.values() if b.file_id == file_id]
            if not file_blocks: continue

            all_blocks_marked_delete = all(b.block_id in delete_block_ids for b in file_blocks)

            if all_blocks_marked_delete:
                # 如果所有块都标记为删除，则对文件应用 'DELETE' 对应的动作
                action = self._settings.action_map.get(DecisionType.DELETE, 'ignore')
                if action != 'ignore':
                    file_actions[file_id] = action

        logger.info(f"解析完成，确定了 {len(file_actions)} 个文件需要执行清理操作。")
        return file_actions


    def _execute_action(self, record: FileRecordDTO, action: str, logger: Logger) -> Optional[ProcessingStatus]:
        """根据配置执行具体的文件操作。"""
        
        original_path = record.original_path
        
        if action == 'mark_only':
            logger.info(f"标记文件 {original_path} 为待删除。")
            return ProcessingStatus.MARKED_FOR_DELETION

        elif action == 'move_to_trash':
            if not self._settings.trash_directory:
                 raise TrashDirectoryError("垃圾箱目录未配置。")
            
            target_path = self._settings.trash_directory / original_path.name
            logger.info(f"移动文件 {original_path} 到垃圾箱 {target_path}...")
            
            if self._fs_adapter.file_exists(original_path):
                 self._fs_adapter.move_file(original_path, target_path)
                 logger.debug(f"文件 {original_path} 移动成功。")
                 return ProcessingStatus.MOVED_TO_TRASH
            else:
                 logger.warning(f"文件 {original_path} 不存在，无法移动，但仍标记为已移动。")
                 return ProcessingStatus.MOVED_TO_TRASH # 或许应该有 'SOURCE_MISSING' 状态？

        elif action == 'permanent_delete':
            logger.warning(f"**永久删除**文件 {original_path}...")
            
            if self._fs_adapter.file_exists(original_path):
                self._fs_adapter.delete_file(original_path)
                logger.debug(f"文件 {original_path} 永久删除成功。")
                return ProcessingStatus.PERMANENTLY_DELETED
            else:
                 logger.warning(f"文件 {original_path} 不存在，无法删除，但仍标记为已删除。")
                 return ProcessingStatus.PERMANENTLY_DELETED

        else: # 'ignore' or unknown
            return None # 状态不变

```