"""
=================================================
cleanup_stage.py - P09 清理阶段实现 (v4.6)
=================================================

**模块功能**:

- 实现 `StageInterface`，执行清理流程。
- **职责**:
    1. 读取 `UserDecisionDTO`，根据决策结果确定每个文件的最终清理动作。
    2. 通过文件系统适配器（fs_adapter）对物理文件执行实际操作（如标记、移动到回收站、永久删除）。
    3. 将每个文件的处理结果（如"已标记删除""已移动到回收站""已永久删除"）同步到 context.file_records。
    4. 不直接进行数据库/存储写入，所有状态变更仅写入 context，由 Orchestrator 统一调度和持久化。
- **架构意义**:
    - CleanupStage 是业务动作的唯一执行者，是决策落地和物理操作的桥梁。
    - 保证业务闭环，实现物理世界与数据世界的同步。
    - 便于审计、回滚和幂等性控制。

---
"""

from typing import List, Dict, Set, Optional
from pathlib import Path
from kd_tool.logging.protocols import LoggerProtocol
import datetime
from kd_tool.core.interfaces import StageInterface
from kd_tool.core.core_dtos import PipelineContextDTO
from kd_tool.schemas.dtos import FileRecordDTO, UserDecisionDTO
from kd_tool.schemas.enums import DecisionType, ProcessingStatus
from kd_tool.stages.cleanup.settings_models import CleanupStageSettings
from kd_tool.stages.cleanup.adapter_interface import FileSystemAdapterInterface
from kd_tool.stages.cleanup.errors import (
    CleanupError,
    FileOperationError,
    TrashDirectoryError,
)
from kd_tool.storage.storage_interface import StorageInterface


class CleanupStage(StageInterface):
    """
    P09 - 清理阶段。
    负责根据上游决策结果，执行实际的文件清理动作（如标记、移动、删除），并将结果同步到 context。
    不直接写库，所有状态变更由 Orchestrator 统一持久化。
    """

    def __init__(
        self,
        logger: LoggerProtocol,
        settings: CleanupStageSettings,
        fs_adapter: FileSystemAdapterInterface,
    ):
        """构造函数，通过 DI 注入所有依赖。"""
        self._logger = logger.bind(stage="CleanupStage")
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
            decisions = list(context.user_decisions.values())
            if not decisions:
                run_logger.info("没有用户决策需要处理。")
                return context
            files_to_process = self._resolve_files_from_decisions(context, run_logger)
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
                    err = CleanupError(
                        f"处理文件 {file_id} ({record.original_path}) 时出错: {e}"
                    )
                    run_logger.error(err)
                    context.add_error(err)
                    record.processing_status = ProcessingStatus.CLEANUP_FAILED
                    updated_records.append(record)
            if updated_records:
                run_logger.info(
                    f"将 {len(updated_records)} 个文件记录的状态更新到 context..."
                )
                for record in updated_records:
                    context.file_records[record.file_id] = record
            run_logger.success("清理阶段执行完毕。")
        except Exception as e:
            error = CleanupError(f"清理阶段发生未知错误: {e}")
            run_logger.exception(error)
            context.add_error(error)
        return context

    def _resolve_files_from_decisions(
        self, context: PipelineContextDTO, logger: LoggerProtocol
    ) -> Dict[str, str]:
        """
        【关键且复杂的逻辑 - 伪代码简化】
        根据 UserDecisionDTO 决定每个文件的最终清理动作。
        返回一个字典 {file_id: action_string}。
        """
        logger.debug("正在解析决策以确定文件操作...")
        file_actions: Dict[str, str] = {}
        delete_block_ids: Set[str] = set()
        for decision in context.user_decisions.values():
            action = self._settings.action_map.get(decision.decision, "ignore")
            if action != "ignore":
                if decision.decision == DecisionType.DELETE:
                    result = next(
                        (
                            r
                            for analyses in context.analysis_results.values()
                            for results_list in analyses.values()
                            for r in results_list
                            if r.pair_analysis_id == decision.pair_analysis_id
                        ),
                        None,
                    )
                    if result:
                        delete_block_ids.add(result.block_id_2)
        for file_id, file_record in context.file_records.items():
            file_blocks = [
                b for b in context.content_blocks.values() if b.file_id == file_id
            ]
            if not file_blocks:
                continue
            all_blocks_marked_delete = all(
                b.block_id in delete_block_ids for b in file_blocks
            )
            if all_blocks_marked_delete:
                action = self._settings.action_map.get(DecisionType.DELETE, "ignore")
                if action != "ignore":
                    file_actions[file_id] = action
        logger.info(f"解析完成，确定了 {len(file_actions)} 个文件需要执行清理操作。")
        return file_actions

    def _execute_action(
        self, record: FileRecordDTO, action: str, logger: LoggerProtocol
    ) -> Optional[ProcessingStatus]:
        """根据配置执行具体的文件操作。"""
        original_path = record.original_path
        if action == "mark_only":
            logger.info(f"标记文件 {original_path} 为待删除。")
            return ProcessingStatus.MARKED_FOR_DELETION
        elif action == "move_to_trash":
            if not self._settings.trash_directory:
                raise TrashDirectoryError("垃圾箱目录未配置。")
            target_path = self._settings.trash_directory / original_path.name
            logger.info(f"移动文件 {original_path} 到垃圾箱 {target_path}...")
            if self._fs_adapter.file_exists(original_path):
                self._fs_adapter.move_file(original_path, target_path)
                logger.debug(f"文件 {original_path} 移动成功。")
                return ProcessingStatus.MOVED_TO_TRASH
            else:
                logger.warning(
                    f"文件 {original_path} 不存在，无法移动，但仍标记为已移动。"
                )
                return ProcessingStatus.MOVED_TO_TRASH
        elif action == "permanent_delete":
            logger.warning(f"**永久删除**文件 {original_path}...")
            if self._fs_adapter.file_exists(original_path):
                self._fs_adapter.delete_file(original_path)
                logger.debug(f"文件 {original_path} 永久删除成功。")
                return ProcessingStatus.PERMANENTLY_DELETED
            else:
                logger.warning(
                    f"文件 {original_path} 不存在，无法删除，但仍标记为已删除。"
                )
                return ProcessingStatus.PERMANENTLY_DELETED
        else:
            return None
