```python

# ------------------------------------------------------------------------------
# 文件名: knowledge_distiller_kd/stages/blockmerging/block_merging_stage.py.md
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

from typing import List, Dict, Tuple # <-- 确保导入所需类型
from loguru import Logger
import copy # 用于安全地复制 DTO，避免直接修改 Context 中的原始对象
import uuid # 用于为合并后的新块生成 ID
import datetime # 用于记录处理历史

# 核心接口和 DTOs
from knowledge_distiller_kd.core.interfaces import StageInterface
from knowledge_distiller_kd.schemas.dtos import (
    PipelineContextDTO,
    ContentBlockDTO
)
# 架构约束: BlockMergingStageSettings 从 schemas.settings_models 导入
from knowledge_distiller_kd.schemas.settings_models import BlockMergingStageSettings, CodeBlockMergeSettings, TextBlockMergeSettings
from knowledge_distiller_kd.schemas.enums import BlockType, ProcessingStatus # 导入所需枚举

# 自定义错误 (从同目录的 errors.py 导入)
# 架构约束: 必须使用在 stages/blockmerging/errors.py.md 中定义的错误
from .errors import BlockMergingError, MergeRuleConflictError, InvalidBlockSequenceError, MergingFailedError

class BlockMergingStage(StageInterface):
    """
    P04 - 块合并阶段实现。
    根据规则合并由 P03 提取的初步内容块。
    """

    def __init__(
        self,
        logger: Logger,
        storage: StageInterface, # 存储接口，本阶段通常不直接写入主要数据，但保持签名一致性
        settings: BlockMergingStageSettings # <-- 使用已定义的 Settings DTO
    ) -> None:
        self._logger = logger.bind(stage="BlockMergingStageP04")
        self._storage = storage # 保留以备未来可能需要读取辅助信息（如词典、模式等）
        self._settings = settings
        self._logger.info(f"BlockMergingStage (P04) initialized with settings: {self._settings.model_dump_json(indent=2)}")

    def process(self, context: PipelineContextDTO) -> PipelineContextDTO:
        task_id = context.get_task_id()
        logger = self._logger.bind(task_id=task_id)

        logger.info("P04 - Block Merging Stage starting...")

        if not self._settings.enabled:
            logger.warning("P04 - BlockMergingStage is disabled. Skipping.")
            return context

        # 架构说明: 从 Context 中获取 P03 阶段生成的初步内容块。
        # 这些块的状态应为 BLOCKS_EXTRACTED 或类似，表明它们是 P03 的产出。
        # 我们需要对这些块进行操作，并用合并后的块更新 Context。
        
        input_blocks_from_context = [
            block for block in context.content_blocks.values()
            # 假设 P03 会将状态更新为 BLOCKS_EXTRACTED，P04 只处理这些
            # 或者 P04 处理所有块，但其合并逻辑自身会判断是否需要合并
            # 为简化，我们先获取所有 content_blocks，按文件和顺序组织
        ]

        if not input_blocks_from_context:
            logger.info("P04 - No content blocks found in context from P03 for merging. Skipping.")
            return context

        # 按 file_id 分组并按 order_in_document 排序，确保合并处理的顺序性
        blocks_by_file_id: Dict[str, List[ContentBlockDTO]] = {}
        for block_dto in input_blocks_from_context:
            # 确保 block_dto 是 ContentBlockDTO 的实例 (防御性编程)
            if not isinstance(block_dto, ContentBlockDTO):
                logger.warning(f"P04 - Skipping non-ContentBlockDTO item in context.content_blocks: {type(block_dto)}")
                continue
            blocks_by_file_id.setdefault(block_dto.file_id, []).append(block_dto)
        
        for file_id_for_sort in blocks_by_file_id:
            blocks_by_file_id[file_id_for_sort].sort(key=lambda b: b.order_in_document if b.order_in_document is not None else float('inf'))

        if not blocks_by_file_id: # 再次检查，因为可能所有项都不是ContentBlockDTO
            logger.info("P04 - No valid ContentBlockDTOs found after grouping. Skipping merging.")
            return context
            
        logger.info(f"P04 - Processing blocks from {len(blocks_by_file_id)} files for merging.")

        # 创建一个新的字典来存放合并后的块，以替换 context.content_blocks
        final_merged_blocks_map: Dict[str, ContentBlockDTO] = {}
        original_block_count_total = len(input_blocks_from_context)
        error_files_count = 0

        for file_id, original_blocks_in_file in blocks_by_file_id.items():
            logger.debug(f"P04 - Merging blocks for file_id: {file_id} (original count: {len(original_blocks_in_file)})")
            
            # 深度复制当前文件的块列表，以避免直接修改迭代中的源
            current_file_processing_blocks = [copy.deepcopy(b) for b in original_blocks_in_file]
            
            try:
                # 架构说明: 合并逻辑可以按优先级或类型进行，例如先合并结构最明显的代码块。
                # P03的输出是初步块，P04负责根据规则将其优化。

                # 1. 应用代码块合并规则 (如果启用)
                if self._settings.code_block_settings.enabled and \
                   BlockType.CODE in self._settings.types_to_attempt_merge:
                    current_file_processing_blocks = self._merge_specific_type_blocks(
                        blocks_to_merge=current_file_processing_blocks,
                        target_type=BlockType.CODE,
                        merge_settings=self._settings.code_block_settings,
                        file_id_for_log=file_id
                    )
                
                # 2. 应用文本块 (TEXT, LIST_ITEM 等) 合并规则 (如果启用)
                # 架构说明: 可以为 TEXT 和 LIST_ITEM 设计不同的合并子逻辑或配置。
                # 这里简化为一个通用的文本块合并，具体行为由 TextBlockMergeSettings 控制。
                if self._settings.text_block_settings.enabled:
                    types_for_text_merge = [
                        bt for bt in [BlockType.TEXT, BlockType.LIST_ITEM] 
                        if bt in self._settings.types_to_attempt_merge
                    ]
                    if types_for_text_merge: # 只有当配置中允许合并这些类型时才进行
                        current_file_processing_blocks = self._merge_specific_type_blocks(
                            blocks_to_merge=current_file_processing_blocks,
                            target_type=types_for_text_merge, # 可以传递一个类型列表
                            merge_settings=self._settings.text_block_settings,
                            file_id_for_log=file_id,
                            preserve_min_len=self._settings.preserve_blocks_with_min_char_length
                        )
                
                # 3. (可选) 其他类型的块合并规则可以按需添加...

                # 将此文件处理后的最终块添加到全局的合并后块映射中
                for final_block in current_file_processing_blocks:
                    final_merged_blocks_map[final_block.block_id] = final_block
                
                logger.info(f"P04 - File {file_id}: original blocks {len(original_blocks_in_file)}, merged to {len(current_file_processing_blocks)} blocks.")
                if file_id in context.file_records: # 更新文件处理状态
                    file_record = context.file_records[file_id]
                    file_record.processing_status = ProcessingStatus.MERGED
                    file_record.metadata['p04_merged_block_count'] = len(current_file_processing_blocks)
                    file_record.processing_history.append({
                        "stage": "P04", "status": ProcessingStatus.MERGED.value, 
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), 
                        "details": f"Original: {len(original_blocks_in_file)}, Merged to: {len(current_file_processing_blocks)} blocks."
                    })


            except BlockMergingError as e: # 捕获本阶段定义的错误
                logger.error(f"P04 - Controlled block merging error for file {file_id}: {e.message}", context_info=e.context_info)
                context.add_error(e)
                error_files_count += 1
                if file_id in context.file_records:
                    context.file_records[file_id].processing_status = ProcessingStatus.MERGE_FAILED
                    context.file_records[file_id].metadata['p04_error'] = e.to_dict()
                    context.file_records[file_id].processing_history.append({
                        "stage": "P04", "status": ProcessingStatus.MERGE_FAILED.value, 
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "error": e.to_dict()
                    })
                # 策略：如果一个文件合并失败，它的块将不会出现在 final_merged_blocks_map 中
                # 这意味着它们实际上从流水线中被移除了（或者说，没有被传递到下一步）
                # 另一种策略可以是保留原始块，但这会使后续分析复杂化。当前选择移除。

            except Exception as e: # 捕获意外错误
                logger.exception(f"P04 - Unexpected critical error merging blocks for file {file_id}.")
                wrapped_error = MergingFailedError(reason=f"Unexpected critical error: {str(e)}", original_exception=e, processing_block_ids=[b.block_id for b in original_blocks_in_file])
                context.add_error(wrapped_error)
                error_files_count += 1
                if file_id in context.file_records:
                    context.file_records[file_id].processing_status = ProcessingStatus.MERGE_FAILED
                    context.file_records[file_id].metadata['p04_error'] = wrapped_error.to_dict()
                    context.file_records[file_id].processing_history.append({
                        "stage": "P04", "status": ProcessingStatus.MERGE_FAILED.value, 
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "error": wrapped_error.to_dict()
                    })

        # 用合并后的块完全替换 Context 中的原有块
        context.content_blocks = final_merged_blocks_map
        final_block_count_total = len(final_merged_blocks_map)
        
        logger.info(
            f"P04 - Block Merging Stage finished. "
            f"Original total blocks: {original_block_count_total}, "
            f"Final merged block count: {final_block_count_total}. "
            f"Files with errors during merge: {error_files_count}."
        )
        return context

    def _merge_specific_type_blocks(
        self,
        input_blocks: List[ContentBlockDTO],
        target_type: Union[BlockType, List[BlockType]], # 可以是单个类型或类型列表
        merge_settings: Union[CodeBlockMergeSettings, TextBlockMergeSettings], # 相应的设置
        file_id_for_log: str, # 用于日志
        preserve_min_len: Optional[int] = None # 特定于文本合并的参数
    ) -> List[ContentBlockDTO]:
        """
        通用的、按类型合并块的辅助方法 (高度伪代码化)。

        架构说明:
            - **coding 阶段要求**: 此方法需要被精心实现。它将是 P04 的核心。
            - 接收一个文件中的所有块（已按序）。
            - 根据 `target_type` 筛选要处理的块。
            - 应用 `merge_settings` 中的具体规则进行合并。
            - 返回一个新的块列表，其中包含合并后的块以及未被合并的其他块。
            - 合并后的新块需要生成新的 `block_id` (使用 `uuid.uuid4()` 或类似方式)，
              并合理设置其 `order_in_document`, `text_content`, `analysis_text`, `block_type` (可能变为更通用的类型),
              和 `metadata` (可能需要聚合或选择性继承)。
            - 必须确保原始块的顺序在未合并部分中得到维持。
        """
        logger.debug(f"P04 - Applying specific merge for type(s) '{str(target_type)}' on {len(input_blocks)} blocks for file {file_id_for_log}.")
        
        if not input_blocks or not merge_settings.enabled:
            return input_blocks

        # 将 target_type 统一为列表处理
        if isinstance(target_type, BlockType):
            target_types_list = [target_type]
        else:
            target_types_list = target_type

        processed_blocks: List[ContentBlockDTO] = []
        i = 0
        while i < len(input_blocks):
            current_block = input_blocks[i]

            # 检查当前块是否是目标类型之一，并且在全局配置中允许被尝试合并
            is_target_for_merge = current_block.block_type in target_types_list and \
                                  current_block.block_type in self._settings.types_to_attempt_merge

            # 检查是否因为长度原因需要保留 (主要针对文本块)
            should_preserve_due_to_length = False
            if preserve_min_len is not None and \
               current_block.block_type == BlockType.TEXT and \
               len(current_block.text_content) >= preserve_min_len:
                should_preserve_due_to_length = True
            
            if not is_target_for_merge or should_preserve_due_to_length:
                processed_blocks.append(current_block)
                i += 1
                continue

            # --- 此处开始是针对 target_type 块的合并逻辑 ---
            # 以下是一个非常非常简化的概念性伪代码，实际合并逻辑会复杂得多
            
            # 示例：尝试合并连续的同类型短块 (通用逻辑)
            accumulated_text_parts = [current_block.text_content]
            accumulated_metadata = [copy.deepcopy(current_block.metadata)] # 保存每个块的元数据
            last_merged_original_block_index = i
            
            j = i + 1
            while j < len(input_blocks) and \
                  input_blocks[j].block_type == current_block.block_type: # 必须是相同类型才能初步考虑合并
                
                # 此处应加入更复杂的合并判断条件，例如：
                # - 对于 CODE: 检查 max_lines_between_blocks_to_merge (需要原始元素信息)
                # - 对于 TEXT: 检查 short_text_char_threshold, max_merged_text_block_length_char
                #             以及标点符号等启发式规则。
                # - 简化：我们只合并连续的两个短块作为示例
                
                # 假设我们决定合并 current_block 和 input_blocks[j]
                # (实际中，需要更复杂的条件来决定是否将 input_blocks[j] 加入合并序列)
                if len(input_blocks[j].text_content) < getattr(merge_settings, 'short_text_char_threshold', float('inf')): # 简化的短块判断
                    # 检查合并后是否超长
                    current_accumulated_len = sum(len(p) for p in accumulated_text_parts) + len(accumulated_text_parts) -1
                    max_len = getattr(merge_settings, 'max_merged_text_block_length_char', float('inf'))
                    if current_accumulated_len + len(input_blocks[j].text_content) + 1 > max_len:
                        break # 合并后会超长，停止

                    accumulated_text_parts.append(input_blocks[j].text_content)
                    accumulated_metadata.append(copy.deepcopy(input_blocks[j].metadata))
                    last_merged_original_block_index = j
                    j += 1
                else: # 下一个块不是短块，或者不满足其他合并条件
                    break 
            
            if last_merged_original_block_index > i: # 确实发生了合并
                # 创建新的合并后的 ContentBlockDTO
                merged_text = "\n".join(accumulated_text_parts) # 简单拼接
                
                # 元数据合并策略:
                #   - 可以选择第一个块的元数据作为基础。
                #   - 将所有被合并块的原始 P03 元数据收集起来，放入新块的元数据中。
                #   - 更新或添加 P04 特定的合并信息。
                final_metadata = copy.deepcopy(current_block.metadata) # 以第一个块为基础
                final_metadata['p04_merged_from_block_ids'] = [input_blocks[k].block_id for k in range(i, last_merged_original_block_index + 1)]
                final_metadata['p04_merged_raw_element_types'] = [m.get('p03_source_element_type') for m in accumulated_metadata if m]
                
                new_merged_block = ContentBlockDTO(
                    block_id=f"merged_block_{uuid.uuid4()}", # 生成新 ID
                    file_id=current_block.file_id,
                    text_content=merged_text,
                    analysis_text=merged_text, # 暂定，后续可做规范化
                    block_type=current_block.block_type, # 合并后通常保持原类型，或变为更通用的 TEXT
                    order_in_document=current_block.order_in_document, # 保留第一个块的顺序
                    page_number=current_block.page_number, # 保留第一个块的页码
                    # text_hash_md5 和 simhash_value 需要在后续阶段重新计算，或清除
                    metadata=final_metadata
                )
                processed_blocks.append(new_merged_block)
                logger.trace(f"P04 - Merged {last_merged_original_block_index - i + 1} blocks of type {current_block.block_type} into new block {new_merged_block.block_id} for file {file_id_for_log}.")
                i = last_merged_original_block_index + 1 # 更新主循环的索引
            else: # 没有发生合并
                processed_blocks.append(current_block)
                i += 1
        
        # 重新计算所有块的 order_in_document (非常重要)
        for new_order, block in enumerate(processed_blocks):
            block.order_in_document = new_order
            
        return processed_blocks

```