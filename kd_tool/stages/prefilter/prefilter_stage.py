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
from kd_tool.logging.protocols import LoggerProtocol
from typing import List, Dict
from pathlib import Path
from uuid import UUID
from kd_tool.core.interfaces import StageInterface
from kd_tool.core.core_dtos import PipelineContextDTO
from kd_tool.schemas.dtos import FileRecordDTO
from kd_tool.schemas.enums import ProcessingStatus
from kd_tool.stages.prefilter.settings_models import PrefilterStageSettings
from kd_tool.stages.prefilter.adapter_interface import CzkawkaAdapterInterface
from kd_tool.stages.prefilter.dtos import CzkawkaScanOutputDTO
from kd_tool.stages.prefilter.errors import PrefilterError

class PrefilterStage(StageInterface):
    """
    WHY: 文件级去重阶段。
    WHAT: 负责调用底层去重工具，生成初步文件唯一性判断。
    HOW: 依赖注入logger、settings、adapter。
    """
    def __init__(
        self,
        logger: LoggerProtocol,
        settings: PrefilterStageSettings,
        adapter: CzkawkaAdapterInterface
    ):
        self._logger = logger
        self._settings = settings
        self._adapter = adapter

    def process(self, context: PipelineContextDTO) -> PipelineContextDTO:
        """
        执行预过滤流程。
        **[指令]** 必须使用 `context.run_logger` 进行日志记录。
        **[指令]** 从 `context.task_id` 获取当前任务ID。
        **[指令]** 创建 `FileRecordDTO` 时 **严禁** 包含 `task_id` 字段。
        **[指令]** 严禁直接调用 `storage` 进行写入操作，**必须**通过 `PipelineContextDTO` 进行状态同步。
        """
        run_logger: LoggerProtocol = context.run_logger.bind(stage_name=self.
            __class__.__name__)
        task_id: UUID = context.task_id
        run_logger.info('预过滤阶段开始...')
        if not self._settings.enabled:
            run_logger.warning('预过滤阶段已禁用. 跳过.')
            return context
        if self._settings.tool != 'czkawka':
            error = PrefilterError(
                f'Unsupported prefilter tool: {self._settings.tool}')
            context.add_error(error)
            run_logger.error(error)
            return context
        try:
            run_logger.info('运行 Czkawka 扫描并查找重复项...')
            scan_output: CzkawkaScanOutputDTO = (self._adapter.
                scan_and_find_duplicates())
            run_logger.success(
                f'Czkawka 完成. 扫描了 {len(scan_output.all_scanned_files)} 个文件. 找到了 {len(scan_output.duplicate_groups)} 个重复组.'
                )
            file_dtos_to_register: List[FileRecordDTO] = []
            if self._settings.register_files_in_storage:
                run_logger.info('注册扫描结果到存储...')
                file_dtos_to_register = self._prepare_scan_results(scan_output, run_logger)
                run_logger.success(
                    f'扫描结果处理完成. 准备注册/更新 {len(file_dtos_to_register)} 个文件.'
                    )
            else:
                run_logger.warning(
                    '注册文件到存储已禁用.')
            if file_dtos_to_register:
                run_logger.info(
                    f'更新 PipelineContextDTO 的 FileRecordDTOs... {len(file_dtos_to_register)}'
                    )
                for dto in file_dtos_to_register:
                    context.add_file_record(dto)
                run_logger.success(
                    'PipelineContextDTO 更新完成.')
            else:
                run_logger.info(
                    '预过滤阶段没有新的 FileRecordDTOs 添加到 PipelineContextDTO.'
                    )
            context.shared_data['prefilter_summary'] = {'scanned_files_count':
                len(scan_output.all_scanned_files),
                'duplicate_groups_count': len(scan_output.duplicate_groups),
                'processed_for_registration_count': len(file_dtos_to_register)}
        except PrefilterError as e:
            run_logger.error(f'预过滤阶段执行错误: {e}',
                exc_info=True)
            context.add_error(e)
        except Exception as e:
            run_logger.exception(
                '预过滤阶段执行意外错误.')
            error = PrefilterError(
                f'预过滤执行失败: {e}',
                original_exception=e)
            context.add_error(error)
        run_logger.info('预过滤阶段完成.')
        return context

    def _prepare_scan_results(self, scan_output: CzkawkaScanOutputDTO,
        run_logger: LoggerProtocol) ->List[FileRecordDTO]:
        """
        将 Czkawka 扫描到的所有文件信息转换为 FileRecordDTO 列表，并调用存储服务注册。
        **[指令]** 此方法不再接收 `task_id`。
        **[指令]** 创建 `FileRecordDTO` 时 **严禁** 包含 `task_id` 字段。
        **[指令]** 调用 `self._storage.register_files` 时 **严禁** 传递 `task_id` (除非接口明确要求)。
        """
        files_to_register_map: Dict[Path, FileRecordDTO] = {}
        processed_paths_from_duplicates = set()
        run_logger.debug(
            f'处理 {len(scan_output.duplicate_groups)} 个重复组...'
            )
        for group in scan_output.duplicate_groups:
            original_path = group.original_file.resolve()
            if original_path not in files_to_register_map:
                files_to_register_map[original_path] = FileRecordDTO(
                    original_path=original_path, size_bytes=group.
                    size_bytes, processing_status=ProcessingStatus.
                    PREPROCESSING_COMPLETED, metadata={
                    'is_original_of_duplicates': True,
                    'czkawka_group_size_bytes': group.size_bytes})
            processed_paths_from_duplicates.add(original_path)
            for dup_path_obj in group.duplicates:
                dup_path = dup_path_obj.resolve()
                if dup_path not in files_to_register_map:
                    files_to_register_map[dup_path] = FileRecordDTO(
                        original_path=dup_path, size_bytes=group.size_bytes,
                        processing_status=ProcessingStatus.DUPLICATE,
                        metadata={'duplicate_of': str(original_path),
                        'czkawka_group_size_bytes': group.size_bytes})
                processed_paths_from_duplicates.add(dup_path)
        run_logger.debug(
            f'处理 {len(processed_paths_from_duplicates)} 个文件来自重复组.'
            )
        run_logger.debug(
            f'处理 {len(scan_output.all_scanned_files)} 个扫描文件以查找唯一文件...'
            )
        for file_path_obj in scan_output.all_scanned_files:
            file_path = file_path_obj.resolve()
            if (file_path not in processed_paths_from_duplicates and 
                file_path not in files_to_register_map):
                try:
                    size = file_path.stat().st_size
                except OSError as e:
                    size = -1
                    run_logger.warning(
                        f'无法获取文件大小: {file_path}. 错误: {e}'
                        )
                files_to_register_map[file_path] = FileRecordDTO(original_path
                    =file_path, size_bytes=size, processing_status=
                    ProcessingStatus.PREPROCESSING_COMPLETED, metadata={
                    'is_unique_in_scan': True})
        file_dtos_list = list(files_to_register_map.values())
        if file_dtos_list:
            run_logger.info(
                f'尝试注册/更新 {len(file_dtos_list)} 个文件到存储...'
                )
            return file_dtos_list
        else:
            run_logger.warning(
                '没有文件找到注册或更新到存储.')
        return []
