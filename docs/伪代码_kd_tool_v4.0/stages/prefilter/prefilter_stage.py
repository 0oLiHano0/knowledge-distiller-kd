# kd_tool/stages/prefilter/prefilter_stage.py (v4.6 - Schema 路径与 task_id 更新版)
# -*- coding: utf-8 -*-

"""
=================================================
prefilter_stage.py.md - PrefilterStage 实现 (v4.6)
=================================================

**模块功能**:

- 负责执行文件级预过滤（如去重）的阶段。
- **v4.6 核心变更**:
    - **[架构指令]** 导入路径已更新，以反映 DTOs 和 Settings Models 的新位置。
    - **[架构指令]** 创建 `FileRecordDTO` 时，不再包含 `task_id` 字段。
      （`task_id` 由 `PipelineContextDTO` 管理）。
    - **[架构指令]** 调用 `StorageInterface.register_files` 时，不再传递 `task_id`。

**架构师说明**:
- **[规范] 状态精确**: Stage 在完成其核心职责后，应尽可能精确地更新 DTO 的
             `processing_status`，以准确反映其处理结果，为下游 Stage
             提供清晰的判断依据。

---
"""

from loguru import Logger
from typing import List, Dict
from pathlib import Path
from uuid import UUID # [指令] 保持 UUID 导入，因为 PipelineContextDTO.task_id 是 UUID

# --- 核心模块导入 ---
# [指令] 根据你的项目结构调整这些核心导入的相对路径
# 假设 prefilter stage 位于 kd_tool/stages/prefilter/
from ....core.interfaces import StageInterface, StorageInterface
from ....core.dtos import PipelineContextDTO      # <-- [指令] 已更新为从 core.dtos 导入
from ....schemas.dtos import FileRecordDTO        # <-- [指令] 保持从中央 schemas 导入 (已移除task_id)
from ....schemas.enums import ProcessingStatus    # <-- [指令] 保持从中央 schemas 导入

# --- Stage 内部导入 ---
from .settings_models import PrefilterStageSettings # <-- [指令] 已更新为本地导入
from .adapter_interface import CzkawkaAdapterInterface
from .dtos import CzkawkaScanOutputDTO              # <-- [指令] 已更新为本地导入
from .errors import PrefilterError                  # <-- [指令] 本地错误导入


class PrefilterStage(StageInterface):
    """
    负责执行文件级预过滤（如去重）的阶段。
    """

    def __init__(self,
                 logger: Logger,
                 settings: PrefilterStageSettings, # <-- [指令] 类型已更新为本地导入的模型
                 storage: StorageInterface,
                 czkawka_adapter: CzkawkaAdapterInterface):
        """
        **[指令]** 构造函数，**必须** 通过 DI 注入所有依赖。
        """
        self._logger = logger.bind(stage_name=self.__class__.__name__)
        self._settings = settings
        self._storage = storage
        self._adapter = czkawka_adapter
        self._logger.info("PrefilterStage initialized.")

    def process(self, context: PipelineContextDTO) -> PipelineContextDTO: #
        """
        执行预过滤流程。
        **[指令]** 必须使用 `context.run_logger` 进行日志记录。
        **[指令]** 从 `context.task_id` 获取当前任务ID。
        **[指令]** 创建 `FileRecordDTO` 时 **严禁** 包含 `task_id` 字段。
        """
        run_logger: Logger = context.run_logger.bind(stage_name=self.__class__.__name__) # 使用 context 中的 logger
        task_id: UUID = context.task_id # 从 context 获取 task_id

        run_logger.info("Starting prefilter stage...") #

        if not self._settings.enabled: #
            run_logger.warning("PrefilterStage is disabled. Skipping.") #
            return context #

        if self._settings.tool != "czkawka": #
            # [指令] 错误应添加到 context.errors 并可选择抛出
            error = PrefilterError(f"Unsupported prefilter tool: {self._settings.tool}") #
            context.add_error(error)
            run_logger.error(error)
            # 根据 Orchestrator 策略，可能不需要在这里抛出，或者抛出以立即停止
            # 假设 Orchestrator 会检查 context.errors
            return context


        try: #
            run_logger.info("Running Czkawka to scan and find duplicates...") #
            scan_output: CzkawkaScanOutputDTO = self._adapter.scan_and_find_duplicates() #
            run_logger.success( #
                f"Czkawka finished. Scanned {len(scan_output.all_scanned_files)} files. " #
                f"Found {len(scan_output.duplicate_groups)} duplicate groups." #
            )

            file_dtos_to_register: List[FileRecordDTO] = [] #

            if self._settings.register_files_in_storage: #
                run_logger.info("Registering scan results in storage...") #
                # [指令] _register_scan_results 不再需要 task_id 参数，因为它不应直接设置到 DTO
                file_dtos_to_register = self._register_scan_results(scan_output, run_logger) #
                run_logger.success(f"Scan results processing complete. Prepared {len(file_dtos_to_register)} files for registration/update.") #
            else: #
                 run_logger.warning("Registering files in storage is disabled by settings.") #

            # [指令] 将处理过的 DTOs 添加/更新到 PipelineContextDTO
            if file_dtos_to_register: #
                run_logger.info(f"Updating PipelineContextDTO with {len(file_dtos_to_register)} FileRecordDTOs...") #
                for dto in file_dtos_to_register:
                    context.add_file_record(dto) # 使用 context 的辅助方法
                run_logger.success("PipelineContextDTO updated with prefilter results.") #
            else: #
                run_logger.info("No new FileRecordDTOs to add to PipelineContextDTO from prefilter stage.") #

            # [指令] 可以在 shared_data 中记录摘要信息
            context.shared_data["prefilter_summary"] = { #
                "scanned_files_count": len(scan_output.all_scanned_files), #
                "duplicate_groups_count": len(scan_output.duplicate_groups), #
                "processed_for_registration_count": len(file_dtos_to_register) #
            }

        except PrefilterError as e: #
            run_logger.error(f"Error during PrefilterStage execution: {e}", exc_info=True) #
            context.add_error(e)
        except Exception as e: #
            run_logger.exception("Unexpected error during PrefilterStage execution.") #
            error = PrefilterError(f"Prefilter execution failed with unexpected error: {e}", original_exception=e) #
            context.add_error(error)

        run_logger.info("Prefilter stage finished.") #
        return context #

    def _register_scan_results(self,
                               scan_output: CzkawkaScanOutputDTO,
                               run_logger: Logger) -> List[FileRecordDTO]: #
        """
        将 Czkawka 扫描到的所有文件信息转换为 FileRecordDTO 列表，并调用存储服务注册。
        **[指令]** 此方法不再接收 `task_id`。
        **[指令]** 创建 `FileRecordDTO` 时 **严禁** 包含 `task_id` 字段。
        **[指令]** 调用 `self._storage.register_files` 时 **严禁** 传递 `task_id` (除非接口明确要求)。
        """
        files_to_register_map: Dict[Path, FileRecordDTO] = {} #
        # 使用字典以路径为键，方便处理重复扫描和更新状态

        processed_paths_from_duplicates = set() #

        run_logger.debug(f"Processing {len(scan_output.duplicate_groups)} duplicate groups...") #
        for group in scan_output.duplicate_groups: #
            original_path = group.original_file.resolve() #

            # 处理原始文件
            if original_path not in files_to_register_map: #
                 files_to_register_map[original_path] = FileRecordDTO( #
                    original_path=original_path, #
                    size_bytes=group.size_bytes, #
                    processing_status=ProcessingStatus.PREPROCESSING_COMPLETED, # 原始文件标记为已完成预处理
                    # task_id=task_id, # <-- [架构指令 v4.6] 移除
                    metadata={"is_original_of_duplicates": True, "czkawka_group_size_bytes": group.size_bytes} #
                )
            processed_paths_from_duplicates.add(original_path) #

            # 处理重复文件
            for dup_path_obj in group.duplicates: #
                dup_path = dup_path_obj.resolve() #
                if dup_path not in files_to_register_map: #
                    files_to_register_map[dup_path] = FileRecordDTO( #
                        original_path=dup_path, #
                        size_bytes=group.size_bytes, #
                        processing_status=ProcessingStatus.DUPLICATE, # 重复文件标记为 DUPLICATE
                        # task_id=task_id, # <-- [架构指令 v4.6] 移除
                        metadata={ #
                            "duplicate_of": str(original_path), #
                            "czkawka_group_size_bytes": group.size_bytes #
                        }
                    )
                processed_paths_from_duplicates.add(dup_path) #

        run_logger.debug(f"Processed {len(processed_paths_from_duplicates)} files from duplicate groups.") #

        # 处理所有扫描到的文件，找出唯一的（未在重复组中处理过的）
        run_logger.debug(f"Processing {len(scan_output.all_scanned_files)} total scanned files to find unique ones...") #
        for file_path_obj in scan_output.all_scanned_files: #
            file_path = file_path_obj.resolve() #
            if file_path not in processed_paths_from_duplicates and file_path not in files_to_register_map: #
                try: #
                    size = file_path.stat().st_size #
                except OSError as e: #
                    size = -1 # 表示获取大小失败
                    run_logger.warning(f"Could not get size for file: {file_path}. Error: {e}") #

                files_to_register_map[file_path] = FileRecordDTO( #
                    original_path=file_path, #
                    size_bytes=size, #
                    processing_status=ProcessingStatus.PREPROCESSING_COMPLETED, # 唯一文件标记为已完成预处理
                    # task_id=task_id, # <-- [架构指令 v4.6] 移除
                    metadata={"is_unique_in_scan": True} #
                )
        # [指令] 获取所有待注册/更新的 DTOs
        file_dtos_list = list(files_to_register_map.values()) #

        if file_dtos_list: #
            run_logger.info(f"Attempting to register/update a total of {len(file_dtos_list)} files in storage...") #
            try: #
                # [指令] StorageInterface.register_files 方法本身不接收 task_id
                # 它处理的是 FileRecordDTO 列表，DTO 内部已不含 task_id
                # Storage 层如果需要 task_id 用于其内部日志或元数据，应从外部更高层（如服务层调用时）传入，
                # 或者在 Storage 方法内部设计一种方式记录操作上下文，但不应污染核心 DTO。
                # 目前的 register_files 签名是正确的。
                # 注意: 此处可能需要事务处理，以确保所有文件要么都注册成功，要么都失败。
                # StorageInterface 提供了 begin_transaction, commit_transaction 等方法。
                # [架构决策]：PrefilterStage 自身不管理事务，由更高层或存储实现内部的 _session_scope 保证原子性。
                #            当前的 _session_scope 在 SQLiteStorage 中是自动提交的。
                returned_dtos_from_storage = self._storage.register_files(file_dtos_list) #
                run_logger.success(f"Bulk registration/update complete. Storage returned {len(returned_dtos_from_storage)} DTOs.") #
                return returned_dtos_from_storage # 返回存储层确认后的 DTOs
            except Exception as e: #
                run_logger.exception("Failed to register/update files in storage.") #
                # 将错误包装后抛出，或添加到 context.errors，取决于错误处理策略
                raise PrefilterError(f"Failed to register/update files in storage: {e}", original_exception=e) from e #
        else: #
            run_logger.warning("No files found to register or update in storage.") #
        return [] #