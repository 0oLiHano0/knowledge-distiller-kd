# kd_tool/stages/blockmerging/block_merging_stage.py (v4.6 - Schema 路径与 task_id 更新版)
# -*- coding: utf-8 -*-


# ------------------------------------------------------------------------------
# 文件名: kd_tool/stages/blockmerging/block_merging_stage.py.md
# 模块: P04 - 块合并阶段 (BlockMergingStage)
# 描述:
#   负责根据配置的规则，将 P03 阶段产生的初步 ContentBlockDTO 进行合并和重组，
#   以生成粒度更合适、信息密度更高的块，供后续分析阶段使用。
# 架构约束:
#   - 必须实现 StageInterface。
#   - 必须通过构造函数接收 Logger, StorageInterface (占位), BlockMergingStageSettings。
#   - 合并逻辑必须基于规则和配置，不应引入基于语义模型的复杂计算。
#   - 主要通过更新 PipelineContextDTO 来输出结果。
#   - 错误处理必须清晰，并使用本模块定义的 BlockMergingError 子类。
# ------------------------------------------------------------------------------

from typing import List, Dict, Tuple, Union # <-- [指令] 确保导入所需类型
from loguru import Logger
import copy
import uuid
import datetime #

# --- 核心模块导入 ---
from ....core.interfaces import StageInterface # StageInterface 被继承
# [架构师说明] BlockMergingStage 的 __init__ 接收 storage: StageInterface，这可能是笔误。
# 通常 Stage 会接收 StorageInterface。此处按原样保留 StageInterface，但提示 coding 阶段确认。
# 如果确实是 StorageInterface，则应为 from ....core.interfaces import StorageInterface
from ....core.interfaces import StorageInterface as CoreStorageInterface # 明确区分，假设构造函数用这个
from ....core.dtos import PipelineContextDTO                     # <-- [指令] 已更新
from ....schemas.dtos import ContentBlockDTO                     # <-- [指令] 已更新 (来自中央 schemas, 已移除 task_id)
from ....schemas.enums import BlockType, ProcessingStatus        # <-- [指令] 已更新 (来自中央 schemas)

# --- Stage 内部导入 ---
from .settings_models import ( # <-- [指令] 已更新为本地导入
    BlockMergerStageSettings,
    CodeBlockMergeSettings,
    TextBlockMergeSettings
)
from .errors import ( # <-- [指令] 本地错误导入
    BlockMergingError,
    MergeRuleConflictError,
    InvalidBlockSequenceError,
    MergingFailedError
)

class BlockMergingStage(StageInterface):
    """
    P04 - 块合并阶段实现。
    根据规则合并由 P03 提取的初步内容块。
    """

    def __init__(
        self,
        logger: Logger,
        # [架构师说明] 原为 storage: StageInterface，根据上下文推断应为 StorageInterface
        storage: CoreStorageInterface,
        settings: BlockMergerStageSettings # <-- [指令] 类型已更新为本地导入的模型
    ) -> None: #
        self._logger = logger.bind(stage="BlockMergingStageP04") #
        self._storage = storage #
        self._settings = settings #
        self._logger.info(f"BlockMergingStage (P04) initialized with settings: {self._settings.model_dump_json(indent=2)}") #


    def process(self, context: PipelineContextDTO) -> PipelineContextDTO:
        # 特别注意： ContentBlockDTO 的创建已不再需要 task_id
        task_id_str = context.get_task_id_str() # 获取 task_id 字符串供日志使用
        logger = context.run_logger.bind(stage="BlockMergingStageP04") #

        logger.info("P04 - Block Merging Stage starting...") #

        if not self._settings.enabled: #
            logger.warning("P04 - BlockMergingStage is disabled. Skipping.") #
            return context #

        input_blocks_from_context = [ #
            block for block in context.content_blocks.values() #
        ]

        if not input_blocks_from_context: #
            logger.info("P04 - No content blocks found in context from P03 for merging. Skipping.") #
            return context #

        blocks_by_file_id: Dict[str, List[ContentBlockDTO]] = {} #
        for block_dto in input_blocks_from_context: #
            if not isinstance(block_dto, ContentBlockDTO): #
                logger.warning(f"P04 - Skipping non-ContentBlockDTO item in context.content_blocks: {type(block_dto)}") #
                continue #
            blocks_by_file_id.setdefault(block_dto.file_id, []).append(block_dto) #
        
        for file_id_for_sort in blocks_by_file_id: #
            blocks_by_file_id[file_id_for_sort].sort(key=lambda b: b.order_in_document if b.order_in_document is not None else float('inf')) #

        if not blocks_by_file_id: #
            logger.info("P04 - No valid ContentBlockDTOs found after grouping. Skipping merging.") #
            return context #
            
        logger.info(f"P04 - Processing blocks from {len(blocks_by_file_id)} files for merging.") #

        final_merged_blocks_map: Dict[str, ContentBlockDTO] = {} #
        original_block_count_total = len(input_blocks_from_context) #
        error_files_count = 0 #

        for file_id, original_blocks_in_file in blocks_by_file_id.items(): #
            logger.debug(f"P04 - Merging blocks for file_id: {file_id} (original count: {len(original_blocks_in_file)})") #
            current_file_processing_blocks = [copy.deepcopy(b) for b in original_blocks_in_file] #
            
            try: #
                # 代码块合并
                if self._settings.code_block_settings.enabled and \
                   BlockType.CODE_SNIPPET in self._settings.types_to_attempt_merge: # 使用 CODE_SNIPPET
                    current_file_processing_blocks = self._merge_specific_type_blocks( #
                        blocks_to_merge=current_file_processing_blocks, #
                        target_type=BlockType.CODE_SNIPPET, # 使用 CODE_SNIPPET
                        merge_settings=self._settings.code_block_settings, #
                        file_id_for_log=file_id, #
                        run_logger=logger # 传递 logger
                    )
                
                # 文本块合并
                if self._settings.text_block_settings.enabled: #
                    types_for_text_merge = [ #
                        bt for bt in [BlockType.NARRATIVE_TEXT, BlockType.LIST_ITEM] # 使用 NARRATIVE_TEXT, LIST_ITEM
                        if bt in self._settings.types_to_attempt_merge #
                    ]
                    if types_for_text_merge: #
                        current_file_processing_blocks = self._merge_specific_type_blocks( #
                            blocks_to_merge=current_file_processing_blocks, #
                            target_type=types_for_text_merge, #
                            merge_settings=self._settings.text_block_settings, #
                            file_id_for_log=file_id, #
                            preserve_min_len=self._settings.preserve_blocks_with_min_char_length, #
                            run_logger=logger # 传递 logger
                        )
                
                for final_block in current_file_processing_blocks: #
                    final_merged_blocks_map[final_block.block_id] = final_block #
                
                logger.info(f"P04 - File {file_id}: original blocks {len(original_blocks_in_file)}, merged to {len(current_file_processing_blocks)} blocks.") #
                if file_id in context.file_records: #
                    file_record = context.file_records[file_id] #
                    # file_record.processing_status = ProcessingStatus.MERGED # MERGED 状态不存在
                    file_record.processing_status = ProcessingStatus.BLOCK_EXTRACTION_COMPLETED # 假设合并后仍是块提取完成，待分析
                    file_record.metadata['p04_merged_block_count'] = len(current_file_processing_blocks) #
                    file_record.processing_history.append({ #
                        "stage": "P04", "status": file_record.processing_status.value, #
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), #
                        "details": f"Original: {len(original_blocks_in_file)}, Merged to: {len(current_file_processing_blocks)} blocks." #
                    })

            except BlockMergingError as e: #
                logger.error(f"P04 - Controlled block merging error for file {file_id}: {e.message}", context_info=e.context_info) #
                context.add_error(e) #
                error_files_count += 1 #
                if file_id in context.file_records: #
                    # context.file_records[file_id].processing_status = ProcessingStatus.MERGE_FAILED # MERGE_FAILED 不存在
                    context.file_records[file_id].processing_status = ProcessingStatus.BLOCK_EXTRACTION_FAILED #
                    context.file_records[file_id].metadata['p04_error'] = e.to_dict() #
                    context.file_records[file_id].processing_history.append({ #
                        "stage": "P04", "status": ProcessingStatus.BLOCK_EXTRACTION_FAILED.value, #
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "error": e.to_dict() #
                    })
            except Exception as e: #
                logger.exception(f"P04 - Unexpected critical error merging blocks for file {file_id}.") #
                wrapped_error = MergingFailedError(reason=f"Unexpected critical error: {str(e)}", original_exception=e, processing_block_ids=[b.block_id for b in original_blocks_in_file]) #
                context.add_error(wrapped_error) #
                error_files_count += 1 #
                if file_id in context.file_records: #
                    # context.file_records[file_id].processing_status = ProcessingStatus.MERGE_FAILED # MERGE_FAILED 不存在
                    context.file_records[file_id].processing_status = ProcessingStatus.BLOCK_EXTRACTION_FAILED #
                    context.file_records[file_id].metadata['p04_error'] = wrapped_error.to_dict() #
                    context.file_records[file_id].processing_history.append({ #
                        "stage": "P04", "status": ProcessingStatus.BLOCK_EXTRACTION_FAILED.value, #
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "error": wrapped_error.to_dict() #
                    })

        context.content_blocks = final_merged_blocks_map #
        final_block_count_total = len(final_merged_blocks_map) #
        
        logger.info( #
            f"P04 - Block Merging Stage finished. " #
            f"Original total blocks: {original_block_count_total}, " #
            f"Final merged block count: {final_block_count_total}. " #
            f"Files with errors during merge: {error_files_count}." #
        )
        return context #

    def _merge_specific_type_blocks(
        self,
        input_blocks: List[ContentBlockDTO],
        target_type: Union[BlockType, List[BlockType]],
        merge_settings: Union[CodeBlockMergeSettings, TextBlockMergeSettings],
        file_id_for_log: str,
        run_logger: Logger, # [指令] 接收 run_logger
        preserve_min_len: Optional[int] = None
    ) -> List[ContentBlockDTO]: #
        """
        通用的、按类型合并块的辅助方法。
        **[指令]** 创建新的 `ContentBlockDTO` 时 **严禁** 包含 `task_id` 字段。
        **[指令]** 使用传入的 `run_logger` 进行日志记录。
        """
        # [指令] 使用传入的 run_logger
        logger = run_logger.bind(method="_merge_specific_type_blocks") #
        logger.debug(f"P04 - Applying specific merge for type(s) '{str(target_type)}' on {len(input_blocks)} blocks for file {file_id_for_log}.") #
        
        if not input_blocks or not getattr(merge_settings, 'enabled', False): # 检查 merge_settings 是否启用 #
            return input_blocks #

        if isinstance(target_type, BlockType): #
            target_types_list = [target_type] #
        else: #
            target_types_list = target_type #

        processed_blocks: List[ContentBlockDTO] = [] #
        i = 0 #
        while i < len(input_blocks): #
            current_block = input_blocks[i] #

            is_target_for_merge = current_block.block_type in target_types_list and \
                                  current_block.block_type in self._settings.types_to_attempt_merge #

            should_preserve_due_to_length = False #
            if preserve_min_len is not None and \
               current_block.block_type == BlockType.NARRATIVE_TEXT and \
               len(current_block.text_content) >= preserve_min_len: #
                should_preserve_due_to_length = True #
            
            if not is_target_for_merge or should_preserve_due_to_length: #
                processed_blocks.append(current_block) #
                i += 1 #
                continue #
            
            accumulated_text_parts = [current_block.text_content] #
            accumulated_metadata = [copy.deepcopy(current_block.metadata)] #
            last_merged_original_block_index = i #
            
            j = i + 1 #
            while j < len(input_blocks) and \
                  input_blocks[j].block_type == current_block.block_type: #
                
                current_accumulated_len = sum(len(p) for p in accumulated_text_parts) + len(accumulated_text_parts) -1 #
                
                # 检查是否是短文本块 (仅对 TextBlockMergeSettings)
                is_short_block = False
                if isinstance(merge_settings, TextBlockMergeSettings):
                    is_short_block = len(input_blocks[j].text_content) < merge_settings.short_text_char_threshold
                elif isinstance(merge_settings, CodeBlockMergeSettings):
                    # 对于代码块，可以假设总是尝试合并，除非有其他特定规则（如 max_lines_between_blocks_to_merge）
                    # 这里简化，假设只要类型匹配就尝试合并，长度限制在后面检查
                    is_short_block = True 

                if not is_short_block and not isinstance(merge_settings, CodeBlockMergeSettings): # 如果不是短文本块且不是代码块设置，则不合并
                    break

                # 检查合并后是否超长 (对 TextBlockMergeSettings 和 CodeBlockMergeSettings 都适用，如果Code也定义了max_len)
                max_len = getattr(merge_settings, 'max_merged_text_block_length_char', # TextBlockMergeSettings
                                  getattr(merge_settings, 'max_merged_code_block_length_char', float('inf'))) # CodeBlockMergeSettings (假设有此属性)

                if current_accumulated_len + len(input_blocks[j].text_content) + 1 > max_len: #
                    break #

                accumulated_text_parts.append(input_blocks[j].text_content) #
                accumulated_metadata.append(copy.deepcopy(input_blocks[j].metadata)) #
                last_merged_original_block_index = j #
                j += 1 #
            
            if last_merged_original_block_index > i: #
                merged_text = "\n".join(accumulated_text_parts) #
                final_metadata = copy.deepcopy(current_block.metadata) #
                final_metadata['p04_merged_from_block_ids'] = [input_blocks[k].block_id for k in range(i, last_merged_original_block_index + 1)] #
                final_metadata['p04_merged_raw_element_types'] = [m.get('p03_source_element_type') for m in accumulated_metadata if m] #
                
                new_merged_block = ContentBlockDTO( #
                    block_id=f"merged_block_{uuid.uuid4().hex}", # 使用 uuid.uuid4().hex
                    file_id=current_block.file_id, #
                    text_content=merged_text, #
                    analysis_text=merged_text, #
                    block_type=current_block.block_type, #
                    order_in_document=current_block.order_in_document, #
                    page_number=current_block.page_number, #
                    metadata=final_metadata #
                )
                processed_blocks.append(new_merged_block) #
                logger.trace(f"P04 - Merged {last_merged_original_block_index - i + 1} blocks of type {current_block.block_type} into new block {new_merged_block.block_id} for file {file_id_for_log}.") #
                i = last_merged_original_block_index + 1 #
            else: #
                processed_blocks.append(current_block) #
                i += 1 #
        
        for new_order, block in enumerate(processed_blocks): #
            block.order_in_document = new_order #
            
        return processed_blocks #