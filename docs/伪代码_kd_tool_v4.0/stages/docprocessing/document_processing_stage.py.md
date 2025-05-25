```python
# ------------------------------------------------------------------------------
# 文件名: kd_tool/stages/docprocessing/document_processing_stage.py
# 模块: P03 - 文档处理阶段 (DocumentProcessingStage) - 原始提取
# 描述:
#   此阶段的核心职责是调用底层文档解析库（如 unstructured），
#   将输入文件解析为初步的、可能较细碎的内容块 (ContentBlockDTO)。
#   它为后续的 P04 BlockMergingStage 提供原始素材。
# 架构约束:
#   - 必须实现 StageInterface。
#   - 必须通过构造函数接收 Logger, StorageInterface (占位), DocumentProcessingStageSettings。
#   - 专注于原始解析和初步DTO转换，复杂的合并逻辑移至 BlockMergingStage。
#   - 错误处理必须清晰，区分文件读取、解析、DTO转换等错误。
# ------------------------------------------------------------------------------

from typing import List, Any, Dict # <-- 确保 Dict 已导入
from pathlib import Path
from loguru import Logger
import datetime # 导入datetime
import uuid # 导入uuid

# 核心接口和 DTOs
from kd_tool.core.interfaces import StageInterface
from kd_tool.schemas.dtos import (
    PipelineContextDTO,
    ContentBlockDTO, # 使用 v4.0 定义的 DTO
    FileRecordDTO    # 使用 v4.0 定义的 DTO
)
# 架构约束: DocumentProcessingStageSettings 从 schemas.settings_models 导入
from kd_tool.schemas.settings_models import DocumentProcessingStageSettings
from kd_tool.schemas.enums import BlockType, ProcessingStatus # 导入所需枚举

# 自定义错误 (从同目录的 errors.py 导入)
# 架构约束: 必须使用在 stages/docprocessing/errors.py.md 中定义的错误
from .errors import (
    DocumentProcessingError,
    FileReadError,
    ParsingError,
    DTOConversionError,
    UnsupportedFileTypeError
)

# 架构设想: 内部解析器封装
class _InternalParserWrapper:
    """
    对底层解析库（如 unstructured）的内部封装。
    架构说明:
        - 这是一个概念性的封装，隔离 DocumentProcessingStage 与具体解析库的直接耦合。
        - 初始化时接收 Logger 和解析策略。
        - parse_file_to_raw_elements 方法负责实际的文件解析，并返回结构化的原始元素列表。
        - 它应该处理底层库可能抛出的异常，并将其包装为本模块定义的 ParsingError 或 FileReadError。
    """
    def __init__(self, logger: Logger, strategy: str):
        self._logger = logger.bind(parser_wrapper=self.__class__.__name__)
        self._strategy = strategy
        self._logger.info(f"InternalParserWrapper initialized with strategy: '{self._strategy}'.")

    def parse_file_to_raw_elements(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        解析文件，返回原始元素字典列表。
        每个字典应包含 'type' (原始类型字符串) 和 'text' (文本内容) 字段，
        以及可选的 'metadata' 字典 (例如包含页码、原始元素ID等)。
        """
        self._logger.debug(f"Attempting to parse file: {file_path} using strategy: {self._strategy}")
        
        # --- 伪代码: 调用底层解析库 (例如 unstructured) ---
        # try:
        #     # 示例: from unstructured.partition.auto import partition
        #     # elements = partition(filename=str(file_path), strategy=self._strategy, **parser_options)
        #     # raw_data_list = []
        #     # for el in elements:
        #     #     raw_data_list.append({
        #     #         "type": type(el).__name__, # 或者 el.category
        #     #         "text": el.text,
        #     #         "metadata": el.metadata.to_dict() # 假设有 to_dict 方法
        #     #     })
        #     # return raw_data_list
        # except FileNotFoundError as e:
        #     # 编码要求: 必须捕获 FileNotFoundError 并抛出 FileReadError
        #     raise FileReadError(file_path, original_exception=e)
        # except Exception as e: # 捕获其他所有来自解析库的异常
        #     # 编码要求: 必须捕获并抛出 ParsingError
        #     raise ParsingError(file_path, "unstructured", original_exception=e)
        # --- 伪代码占位符 (开始) ---
        if "nonexistent_file.txt" in str(file_path):
            raise FileReadError(file_path, original_exception=FileNotFoundError("Simulated file not found"))
        if "fail_parse.docx" in str(file_path):
            raise ParsingError(file_path, "unstructured", original_exception=ValueError("Simulated unstructured parsing failure"))
        
        # 模拟成功解析的输出
        return [
            {"type": "Title", "text": f"Document Title for {file_path.name}", "metadata": {"page_number": 1, "source_id": "elem_001"}},
            {"type": "NarrativeText", "text": "This is the first paragraph.", "metadata": {"page_number": 1, "source_id": "elem_002"}},
            {"type": "ListItem", "text": "First item in a list.", "metadata": {"page_number": 1, "source_id": "elem_003"}},
            {"type": "Code", "text": "print('Hello')", "metadata": {"page_number": 2, "language": "python", "source_id": "elem_004"}}
        ]
        # --- 伪代码占位符 (结束) ---


class DocumentProcessingStage(StageInterface):
    """
    P03 - 文档处理阶段（原始提取）实现。
    负责将文件解析为初步的 ContentBlockDTO 列表。
    """

    def __init__(
        self,
        logger: Logger,
        storage: StageInterface, # 存储接口，通常此阶段不直接写入主要数据，但可用于读取辅助信息或记录状态
        settings: DocumentProcessingStageSettings # <-- 使用已定义的 Settings DTO
    ) -> None:
        self._logger = logger.bind(stage="DocumentProcessingStageP03")
        self._storage = storage
        self._settings = settings
        # 架构约束: 初始化内部的解析器封装
        self._parser = _InternalParserWrapper(logger=self._logger, strategy=self._settings.parsing_strategy)
        self._logger.info(
            f"DocumentProcessingStage (P03 - Raw Extraction) initialized. "
            f"Supported extensions: {self._settings.supported_extensions}. "
            f"Parsing strategy: '{self._settings.parsing_strategy}'."
        )

    def process(self, context: PipelineContextDTO) -> PipelineContextDTO:
        task_id = context.get_task_id()
        logger = self._logger.bind(task_id=task_id)

        logger.info("P03 - Document Processing (Raw Extraction) starting...")

        if not self._settings.enabled:
            logger.warning("P03 - DocumentProcessingStage is disabled. Skipping.")
            return context

        # 架构约束: 从 Context 中获取状态为 PENDING 的 FileRecordDTO
        files_to_process = [
            record for record in context.file_records.values()
            if record.processing_status == ProcessingStatus.PENDING
        ]

        if not files_to_process:
            logger.info("P03 - No pending files found for document processing. Skipping.")
            return context

        logger.info(f"P03 - Found {len(files_to_process)} pending files to process.")
        processed_file_count = 0
        total_blocks_generated_in_run = 0

        for file_record in files_to_process:
            file_path = file_record.original_path
            file_id = file_record.file_id
            current_file_block_count = 0
            logger.debug(f"P03 - Processing file: {file_path} (ID: {file_id})")

            try:
                # 1. 检查文件类型是否支持
                # 架构约束: 必须使用 settings.supported_extensions 进行检查
                # 编码要求: 必须抛出 UnsupportedFileTypeError 如果文件类型不受支持
                if file_path.suffix.lower() not in self._settings.supported_extensions:
                    raise UnsupportedFileTypeError(file_path, detected_file_type=file_path.suffix)

                # 2. 调用内部解析器获取原始元素
                # 架构约束: _InternalParserWrapper 内部处理 FileReadError 和 ParsingError
                raw_elements = self._parser.parse_file_to_raw_elements(file_path)

                if not raw_elements:
                    logger.warning(f"P03 - No raw elements extracted from {file_path}. File processed as empty.")
                    context.file_records[file_id].processing_status = ProcessingStatus.PROCESSED_EMPTY
                    context.file_records[file_id].metadata['p03_notes'] = "No elements extracted by parser."
                    # 更新处理历史
                    file_record.processing_history.append({
                        "stage": "P03", "status": ProcessingStatus.PROCESSED_EMPTY.value, 
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "details": "No raw elements extracted."
                    })
                    processed_file_count +=1
                    continue

                # 3. 将原始元素转换为初步的 ContentBlockDTO
                # 架构约束: 必须处理 DTOConversionError
                preliminary_blocks = self._convert_raw_elements_to_dtos(
                    raw_elements, file_id=file_id, file_path_for_error_reporting=file_path
                )

                for block_dto in preliminary_blocks:
                    context.add_content_block(block_dto)
                    current_file_block_count += 1
                
                total_blocks_generated_in_run += current_file_block_count
                
                # 4. 更新 Context 中文件的状态为待合并
                # 架构约束: FileRecordDTO 的状态应更新，以表明P03已完成，P04可介入
                context.file_records[file_id].processing_status = ProcessingStatus.BLOCKS_EXTRACTED 
                context.file_records[file_id].metadata['p03_extracted_block_count'] = current_file_block_count
                file_record.processing_history.append({
                    "stage": "P03", "status": ProcessingStatus.BLOCKS_EXTRACTED.value, 
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), 
                    "details": f"Extracted {current_file_block_count} preliminary blocks."
                })
                processed_file_count += 1
                logger.success(f"P03 - Successfully extracted {current_file_block_count} preliminary blocks from: {file_path}")

            except DocumentProcessingError as e: # 首先捕获我们定义的更具体的错误
                logger.error(f"P03 - Controlled error during processing of {file_path}: {e.message}", context_info=e.context_info)
                context.add_error(e) # 将我们自定义的错误实例添加到context
                if file_id in context.file_records: # 确保 file_id 有效
                    context.file_records[file_id].processing_status = ProcessingStatus.FAILED
                    context.file_records[file_id].metadata['p03_error'] = e.to_dict() # 存储错误的字典表示
                    file_record.processing_history.append({
                        "stage": "P03", "status": ProcessingStatus.FAILED.value, 
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "error": e.to_dict()
                    })
            except Exception as e: # 捕获其他所有意外错误
                logger.exception(f"P03 - Unexpected critical error processing file {file_path}.")
                # 将未知错误包装成我们的 DocumentProcessingError
                wrapped_error = DocumentProcessingError(
                    message=f"Unexpected critical error: {str(e)}", 
                    original_exception=e, 
                    file_path=file_path
                )
                context.add_error(wrapped_error)
                if file_id in context.file_records:
                    context.file_records[file_id].processing_status = ProcessingStatus.FAILED
                    context.file_records[file_id].metadata['p03_error'] = wrapped_error.to_dict()
                    file_record.processing_history.append({
                        "stage": "P03", "status": ProcessingStatus.FAILED.value, 
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "error": wrapped_error.to_dict()
                    })
        
        logger.info(
            f"P03 - Document Processing (Raw Extraction) finished. "
            f"Successfully processed files: {processed_file_count}/{len(files_to_process)}. "
            f"Total preliminary blocks generated in this run: {total_blocks_generated_in_run}."
        )
        return context

    def _convert_raw_elements_to_dtos(
        self,
        raw_elements: List[Dict[str, Any]],
        file_id: str,
        file_path_for_error_reporting: Path
    ) -> List[ContentBlockDTO]:
        """
        将原始解析元素列表转换为初步的 ContentBlockDTO 列表。
        此阶段的转换应尽量直接，主要做类型映射和基本信息填充。
        """
        dtos: List[ContentBlockDTO] = []
        for index, raw_element in enumerate(raw_elements):
            element_info_for_error = f"Raw element at index {index} (type: {raw_element.get('type', 'N/A')})"
            try:
                # 编码要求: 必须从 raw_element 安全地提取文本和类型
                element_text = str(raw_element.get("text", "")) # 确保是字符串
                raw_element_type_str = str(raw_element.get("type", "UnknownElement")) 
                raw_element_metadata = raw_element.get("metadata", {})
                if not isinstance(raw_element_metadata, dict): raw_element_metadata = {}


                # --- 架构约束: BlockType 映射逻辑 ---
                # 这里的映射规则需要根据底层解析库 (如 unstructured) 实际输出的类型名来精确定义。
                # 以下是一个示例性的、需要细化的映射。
                block_type_mapped = BlockType.UNKNOWN # 默认值
                # 示例映射 (需要根据 unstructured 的具体输出调整)
                if "Title" in raw_element_type_str or "Heading" in raw_element_type_str:
                    block_type_mapped = BlockType.HEADING
                elif "NarrativeText" in raw_element_type_str: # unstructured 常见类型
                    block_type_mapped = BlockType.TEXT
                elif "ListItem" in raw_element_type_str: # unstructured 常见类型
                    block_type_mapped = BlockType.LIST_ITEM
                elif "Code" in raw_element_type_str or "CodeSnippet" in raw_element_type_str: # "CodeSnippet" 可能来自某些解析器
                    block_type_mapped = BlockType.CODE
                elif "Table" in raw_element_type_str: # unstructured 常见类型
                    block_type_mapped = BlockType.TABLE
                # ... 可根据 unstructured 输出添加更多精确或启发式的映射规则 ...
                
                # 编码要求: 必须正确填充 ContentBlockDTO 的字段
                dto = ContentBlockDTO(
                    # block_id 会由 DTO 自身 default_factory 生成
                    file_id=file_id,
                    text_content=element_text,
                    analysis_text=element_text, # 初步阶段，analysis_text 等同于 text_content
                    block_type=block_type_mapped,
                    order_in_document=index,    # 记录原始顺序，对P04合并阶段可能有用
                    page_number=raw_element_metadata.get("page_number"), # 尝试从元数据获取
                    metadata={ # 存储P03阶段的原始解析信息，供P04或其他阶段参考
                        "p03_source_element_type": raw_element_type_str,
                        "p03_source_parser_metadata": raw_element_metadata, # 例如 unstructured 的元素元数据
                    }
                )
                dtos.append(dto)
            except Exception as e:
                # 编码要求: 转换单个元素时发生的任何错误都必须包装为 DTOConversionError
                raise DTOConversionError(
                    file_path=file_path_for_error_reporting,
                    element_info=element_info_for_error,
                    original_exception=e
                )
        return dtos
```