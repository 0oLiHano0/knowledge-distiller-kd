from typing import List, Dict, Tuple, Union
from kd_tool.logging.protocols import LoggerProtocol
import copy
import uuid
import datetime
from kd_tool.core.interfaces import StageInterface
from kd_tool.core.core_dtos import PipelineContextDTO
from kd_tool.schemas.dtos import ContentBlockDTO
from kd_tool.schemas.enums import BlockType, ProcessingStatus
from kd_tool.stages.blockmerging.settings_models import (
    BlockMergerStageSettings,
    CodeBlockMergeSettings,
    TextBlockMergeSettings,
)
from kd_tool.stages.blockmerging.errors import (
    BlockMergingError,
    MergeRuleConflictError,
    InvalidBlockSequenceError,
    MergingFailedError,
)
from typing import Optional


class BlockMergingStage(StageInterface):
    """
    WHY: 块合并阶段，提升内容块质量，减少冗余，为后续分析提供更优输入。
    WHAT: 按规则合并同类型内容块，更新 PipelineContextDTO。
    HOW: 依赖注入 logger/settings，process 只操作 context，不直接写库。
    """

    def __init__(self, logger: LoggerProtocol, settings: BlockMergerStageSettings):
        """
        WHY: 初始化依赖。
        WHAT: 记录日志器和配置。
        HOW: 通过依赖注入赋值。
        """
        self._logger = logger
        self._settings = settings
        self._logger.info(
            f"初始化完成. 设置: {self._settings.model_dump_json(indent=2)}"
        )

    def process(self, context: PipelineContextDTO) -> PipelineContextDTO:
        """
        WHY: 执行块合并，提升后续分析质量。
        WHAT: 合并同类型内容块，更新 context。
        HOW: 只操作 context，不写库，异常写入 context.errors。
        """
        logger = context.run_logger.bind(stage="BlockMergingStage")
        logger.info("块合并阶段开始...")
        if not self._settings.enabled:
            logger.warning("块合并阶段已禁用. 跳过.")
            return context
        input_blocks_from_context = [block for block in context.content_blocks.values()]
        if not input_blocks_from_context:
            logger.info("没有找到内容块. 跳过.")
            return context
        blocks_by_file_id: Dict[str, List[ContentBlockDTO]] = {}
        for block_dto in input_blocks_from_context:
            if not isinstance(block_dto, ContentBlockDTO):
                logger.warning(f"跳过非 ContentBlockDTO 项: {type(block_dto)}")
                continue
            blocks_by_file_id.setdefault(block_dto.file_id, []).append(block_dto)
        for file_id_for_sort in blocks_by_file_id:
            blocks_by_file_id[file_id_for_sort].sort(
                key=lambda b: (
                    b.order_in_document
                    if b.order_in_document is not None
                    else float("inf")
                )
            )
        if not blocks_by_file_id:
            logger.info("没有找到有效的 ContentBlockDTOs. 跳过合并.")
            return context
        logger.info(
            f"P04 - Processing blocks from {len(blocks_by_file_id)} files for merging."
        )
        final_merged_blocks_map: Dict[str, ContentBlockDTO] = {}
        original_block_count_total = len(input_blocks_from_context)
        error_files_count = 0
        for file_id, original_blocks_in_file in blocks_by_file_id.items():
            logger.debug(
                f"合并块: {file_id} (原始数量: {len(original_blocks_in_file)})"
            )
            current_file_processing_blocks = [
                copy.deepcopy(b) for b in original_blocks_in_file
            ]
            try:
                if (
                    self._settings.code_block_settings.enabled
                    and BlockType.CODE_SNIPPET in self._settings.types_to_attempt_merge
                ):
                    current_file_processing_blocks = self._merge_specific_type_blocks(
                        blocks_to_merge=current_file_processing_blocks,
                        target_type=BlockType.CODE_SNIPPET,
                        merge_settings=self._settings.code_block_settings,
                        file_id_for_log=file_id,
                        run_logger=logger,
                    )
                if self._settings.text_block_settings.enabled:
                    types_for_text_merge = [
                        bt
                        for bt in [BlockType.NARRATIVE_TEXT, BlockType.LIST_ITEM]
                        if bt in self._settings.types_to_attempt_merge
                    ]
                    if types_for_text_merge:
                        current_file_processing_blocks = self._merge_specific_type_blocks(
                            blocks_to_merge=current_file_processing_blocks,
                            target_type=types_for_text_merge,
                            merge_settings=self._settings.text_block_settings,
                            file_id_for_log=file_id,
                            preserve_min_len=self._settings.preserve_blocks_with_min_char_length,
                            run_logger=logger,
                        )
                for final_block in current_file_processing_blocks:
                    final_merged_blocks_map[final_block.block_id] = final_block
                logger.info(
                    f"文件 {file_id}: 原始块 {len(original_blocks_in_file)}, 合并到 {len(current_file_processing_blocks)} 块."
                )
                if file_id in context.file_records:
                    file_record = context.file_records[file_id]
                    file_record.processing_status = (
                        ProcessingStatus.BLOCK_EXTRACTION_COMPLETED
                    )
                    file_record.metadata["p04_merged_block_count"] = len(
                        current_file_processing_blocks
                    )
                    file_record.processing_history.append(
                        {
                            "stage": "P04",
                            "status": file_record.processing_status.value,
                            "timestamp": datetime.datetime.now(
                                datetime.timezone.utc
                            ).isoformat(),
                            "details": f"Original: {len(original_blocks_in_file)}, Merged to: {len(current_file_processing_blocks)} blocks.",
                        }
                    )
            except BlockMergingError as e:
                logger.error(
                    f"控制块合并错误: {file_id}: {e.message}",
                    context_info=e.context_info,
                )
                context.add_error(e)
                error_files_count += 1
                if file_id in context.file_records:
                    context.file_records[file_id].processing_status = (
                        ProcessingStatus.BLOCK_EXTRACTION_FAILED
                    )
                    context.file_records[file_id].metadata["p04_error"] = e.to_dict()
                    context.file_records[file_id].processing_history.append(
                        {
                            "stage": "P04",
                            "status": ProcessingStatus.BLOCK_EXTRACTION_FAILED.value,
                            "timestamp": datetime.datetime.now(
                                datetime.timezone.utc
                            ).isoformat(),
                            "error": e.to_dict(),
                        }
                    )
            except Exception as e:
                logger.exception(f"合并块时发生意外严重错误: {file_id}.")
                wrapped_error = MergingFailedError(
                    reason=f"Unexpected critical error: {str(e)}",
                    original_exception=e,
                    processing_block_ids=[b.block_id for b in original_blocks_in_file],
                )
                context.add_error(wrapped_error)
                error_files_count += 1
                if file_id in context.file_records:
                    context.file_records[file_id].processing_status = (
                        ProcessingStatus.BLOCK_EXTRACTION_FAILED
                    )
                    context.file_records[file_id].metadata[
                        "p04_error"
                    ] = wrapped_error.to_dict()
                    context.file_records[file_id].processing_history.append(
                        {
                            "stage": "P04",
                            "status": ProcessingStatus.BLOCK_EXTRACTION_FAILED.value,
                            "timestamp": datetime.datetime.now(
                                datetime.timezone.utc
                            ).isoformat(),
                            "error": wrapped_error.to_dict(),
                        }
                    )
        context.content_blocks = final_merged_blocks_map
        final_block_count_total = len(final_merged_blocks_map)
        logger.info(
            f"块合并阶段完成. 原始总块数: {original_block_count_total}, 最终合并块数: {final_block_count_total}. 合并时发生错误文件数: {error_files_count}."
        )
        # 文件状态更新
        for file_id, file_record in context.file_records.items():
            file_record.processing_status = ProcessingStatus.BLOCK_EXTRACTION_COMPLETED
        logger.info("P04 - Block Merging Stage finished.")
        return context

    def _merge_specific_type_blocks(
        self,
        input_blocks: List[ContentBlockDTO],
        target_type: Union[BlockType, List[BlockType]],
        merge_settings: Union[CodeBlockMergeSettings, TextBlockMergeSettings],
        file_id_for_log: str,
        run_logger: LoggerProtocol,
        preserve_min_len: Optional[int] = None,
    ) -> List[ContentBlockDTO]:
        """
        WHY: 合并同类型内容块，减少冗余。
        WHAT: 按类型和规则合并，生成新块。
        HOW: 只操作内存数据，不写库，日志详细。
        """
        logger = run_logger.bind(method="_merge_specific_type_blocks")
        logger.debug(
            f"应用特定合并: {str(target_type)} 在 {len(input_blocks)} 块上 for file {file_id_for_log}."
        )
        if not input_blocks or not getattr(merge_settings, "enabled", False):
            return input_blocks
        if isinstance(target_type, BlockType):
            target_types_list = [target_type]
        else:
            target_types_list = target_type
        processed_blocks: List[ContentBlockDTO] = []
        i = 0
        while i < len(input_blocks):
            current_block = input_blocks[i]
            is_target_for_merge = (
                current_block.block_type in target_types_list
                and current_block.block_type in self._settings.types_to_attempt_merge
            )
            should_preserve_due_to_length = False
            if (
                preserve_min_len is not None
                and current_block.block_type == BlockType.NARRATIVE_TEXT
                and len(current_block.text_content) >= preserve_min_len
            ):
                should_preserve_due_to_length = True
            if not is_target_for_merge or should_preserve_due_to_length:
                processed_blocks.append(current_block)
                i += 1
                continue
            accumulated_text_parts = [current_block.text_content]
            accumulated_metadata = [copy.deepcopy(current_block.metadata)]
            last_merged_original_block_index = i
            j = i + 1
            while (
                j < len(input_blocks)
                and input_blocks[j].block_type == current_block.block_type
            ):
                current_accumulated_len = (
                    sum(len(p) for p in accumulated_text_parts)
                    + len(accumulated_text_parts)
                    - 1
                )
                is_short_block = False
                if isinstance(merge_settings, TextBlockMergeSettings):
                    is_short_block = (
                        len(input_blocks[j].text_content)
                        < merge_settings.short_text_char_threshold
                    )
                elif isinstance(merge_settings, CodeBlockMergeSettings):
                    is_short_block = True
                if not is_short_block and not isinstance(
                    merge_settings, CodeBlockMergeSettings
                ):
                    break
                max_len = getattr(
                    merge_settings,
                    "max_merged_text_block_length_char",
                    getattr(
                        merge_settings,
                        "max_merged_code_block_length_char",
                        float("inf"),
                    ),
                )
                if (
                    current_accumulated_len + len(input_blocks[j].text_content) + 1
                    > max_len
                ):
                    break
                accumulated_text_parts.append(input_blocks[j].text_content)
                accumulated_metadata.append(copy.deepcopy(input_blocks[j].metadata))
                last_merged_original_block_index = j
                j += 1
            if last_merged_original_block_index > i:
                merged_text = "\n".join(accumulated_text_parts)
                final_metadata = copy.deepcopy(current_block.metadata)
                final_metadata["p04_merged_from_block_ids"] = [
                    input_blocks[k].block_id
                    for k in range(i, last_merged_original_block_index + 1)
                ]
                final_metadata["p04_merged_raw_element_types"] = [
                    m.get("p03_source_element_type") for m in accumulated_metadata if m
                ]
                new_merged_block = ContentBlockDTO(
                    block_id=f"merged_block_{uuid.uuid4().hex}",
                    file_id=current_block.file_id,
                    text_content=merged_text,
                    analysis_text=merged_text,
                    block_type=current_block.block_type,
                    order_in_document=current_block.order_in_document,
                    page_number=current_block.page_number,
                    metadata=final_metadata,
                )
                processed_blocks.append(new_merged_block)
                logger.trace(
                    f"合并 {last_merged_original_block_index - i + 1} 个块 of type {current_block.block_type} into new block {new_merged_block.block_id} for file {file_id_for_log}."
                )
                i = last_merged_original_block_index + 1
            else:
                processed_blocks.append(current_block)
                i += 1
        for new_order, block in enumerate(processed_blocks):
            block.order_in_document = new_order
        return processed_blocks
