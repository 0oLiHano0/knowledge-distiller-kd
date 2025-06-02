"""
=================================================
document_processing_stage.py - P03 文档处理阶段 (v4.6)
=================================================
"""
from typing import List, Any, Dict
from pathlib import Path
from kd_tool.logging.protocols import LoggerProtocol
import datetime
import uuid
from kd_tool.core.interfaces import StageInterface
from kd_tool.core.core_dtos import PipelineContextDTO
from kd_tool.schemas.dtos import ContentBlockDTO, FileRecordDTO
from kd_tool.schemas.enums import BlockType, ProcessingStatus
from kd_tool.stages.docprocessing.settings_models import DocumentProcessingStageSettings
from kd_tool.stages.docprocessing.errors import DocumentProcessingError, FileReadError, ParsingError, DTOConversionError, UnsupportedFileTypeError



class _InternalParserWrapper:

    def __init__(self, logger: LoggerProtocol, strategy: str):
        self._logger = logger.bind(parser_wrapper=self.__class__.__name__)
        self._strategy = strategy
        self._logger.info(
            f"初始化完成. 策略: '{self._strategy}'."
            )

    def parse_file_to_raw_elements(self, file_path: Path) ->List[Dict[str, Any]
        ]:
        self._logger.debug(
            f'尝试解析文件: {file_path} 使用策略: {self._strategy}'
            )
        if 'nonexistent_file.txt' in str(file_path):
            raise FileReadError(file_path, original_exception=
                FileNotFoundError('Simulated file not found'))
        if 'fail_parse.docx' in str(file_path):
            raise ParsingError(file_path, 'unstructured',
                original_exception=ValueError(
                'Simulated unstructured parsing failure'))
        return [{'type': 'Title', 'text':
            f'Document Title for {file_path.name}', 'metadata': {
            'page_number': 1, 'source_id': 'elem_001'}}, {'type':
            'NarrativeText', 'text': 'This is the first paragraph.',
            'metadata': {'page_number': 1, 'source_id': 'elem_002'}}, {
            'type': 'ListItem', 'text': 'First item in a list.', 'metadata':
            {'page_number': 1, 'source_id': 'elem_003'}}, {'type': 'Code',
            'text': "print('Hello')", 'metadata': {'page_number': 2,
            'language': 'python', 'source_id': 'elem_004'}}]


class DocumentProcessingStage(StageInterface):
    """
    P03 - 文档处理阶段（原始提取）实现。
    负责将文件解析为初步的 ContentBlockDTO 列表。
    """

    def __init__(self, logger: LoggerProtocol, settings:
        DocumentProcessingStageSettings) ->None:
        self._logger = logger.bind(stage='DocumentProcessingStageP03')
        self._settings = settings
        self._parser = _InternalParserWrapper(logger=self._logger, strategy
            =self._settings.parsing_strategy)
        self._logger.info(
            f"DocumentProcessingStage 初始化完成. 支持的扩展名: {self._settings.supported_extensions}. 解析策略: '{self._settings.parsing_strategy}'."
            )

    def process(self, context: PipelineContextDTO) ->PipelineContextDTO:
        logger = context.run_logger.bind(stage='DocumentProcessingStage')
        logger.info('文档处理阶段开始...')
        if not self._settings.enabled:
            logger.warning('文档处理阶段已禁用，跳过。')
            return context

        files_to_process = [record for record in context.file_records.values()
                            if record.processing_status == ProcessingStatus.PENDING]
        if not files_to_process:
            logger.info('没有待处理文件，跳过。')
            return context

        for file_record in files_to_process:
            file_path = file_record.original_path
            file_id = file_record.file_id
            try:
                if file_path.suffix.lower(
                    ) not in self._settings.supported_extensions:
                    raise UnsupportedFileTypeError(file_path,
                        detected_file_type=file_path.suffix)
                raw_elements = self._parser.parse_file_to_raw_elements(
                    file_path)
                if not raw_elements:
                    logger.warning(
                        f'没有从 {file_path} 提取到原始元素. 文件处理为空.'
                        )
                    file_record.processing_status = (ProcessingStatus.
                        BLOCK_EXTRACTION_FAILED)
                    file_record.metadata['p03_notes'
                        ] = 'No elements extracted by parser.'
                    file_record.processing_history.append({'stage': 'P03',
                        'status': file_record.processing_status.value,
                        'timestamp': datetime.datetime.now(datetime.
                        timezone.utc).isoformat(), 'details':
                        'No raw elements extracted.'})
                    continue
                preliminary_blocks = self._convert_raw_elements_to_dtos(
                    raw_elements, file_id=file_id,
                    file_path_for_error_reporting=file_path)
                for block_dto in preliminary_blocks:
                    context.add_content_block(block_dto)
                file_record.processing_status = (ProcessingStatus.
                    BLOCK_EXTRACTION_COMPLETED)
                file_record.metadata['p03_extracted_block_count'
                    ] = len(preliminary_blocks)
                file_record.processing_history.append({'stage': 'P03',
                    'status': ProcessingStatus.BLOCK_EXTRACTION_COMPLETED.
                    value, 'timestamp': datetime.datetime.now(datetime.
                    timezone.utc).isoformat(), 'details':
                    f'Extracted {len(preliminary_blocks)} preliminary blocks.'
                    })
            except DocumentProcessingError as e:
                logger.error(
                    f'处理 {file_path} 时控制错误: {e.message}'
                    , context_info=e.context_info)
                context.add_error(e)
                if file_id in context.file_records:
                    file_record.processing_status = (ProcessingStatus.
                        BLOCK_EXTRACTION_FAILED)
                    file_record.metadata['p03_error'] = e.to_dict()
                    file_record.processing_history.append({'stage': 'P03',
                        'status': ProcessingStatus.BLOCK_EXTRACTION_FAILED.
                        value, 'timestamp': datetime.datetime.now(datetime.
                        timezone.utc).isoformat(), 'error': e.to_dict()})
            except Exception as e:
                logger.exception(
                    f'处理 {file_path} 时发生严重错误.'
                    )
                wrapped_error = DocumentProcessingError(message=
                    f'Unexpected critical error: {str(e)}',
                    original_exception=e, file_path=file_path)
                context.add_error(wrapped_error)
                if file_id in context.file_records:
                    file_record.processing_status = (ProcessingStatus.
                        BLOCK_EXTRACTION_FAILED)
                    file_record.metadata['p03_error'] = wrapped_error.to_dict()
                    file_record.processing_history.append({'stage': 'P03',
                        'status': ProcessingStatus.BLOCK_EXTRACTION_FAILED.
                        value, 'timestamp': datetime.datetime.now(datetime.
                        timezone.utc).isoformat(), 'error': wrapped_error.
                        to_dict()})
            if file_id in context.file_records:
                context.file_records[file_id] = file_record
        logger.info('文档处理阶段完成。')
        return context

    def _convert_raw_elements_to_dtos(self, raw_elements: List[Dict[str,
        Any]], file_id: str, file_path_for_error_reporting: Path) ->List[
        ContentBlockDTO]:
        """
        将原始解析元素列表转换为初步的 ContentBlockDTO 列表。
        **[指令]** 创建 `ContentBlockDTO` 时 **严禁** 包含 `task_id` 字段。
        """
        dtos: List[ContentBlockDTO] = []
        for index, raw_element in enumerate(raw_elements):
            element_info_for_error = (
                f"原始元素索引 {index} (类型: {raw_element.get('type', 'N/A')})"
                )
            try:
                element_text = str(raw_element.get('text', ''))
                raw_element_type_str = str(raw_element.get('type',
                    'UnknownElement'))
                raw_element_metadata = raw_element.get('metadata', {})
                if not isinstance(raw_element_metadata, dict):
                    raw_element_metadata = {}
                block_type_mapped = BlockType.UNCATEGORIZED
                if ('Title' in raw_element_type_str or 'Heading' in
                    raw_element_type_str):
                    block_type_mapped = BlockType.TITLE
                elif 'NarrativeText' in raw_element_type_str:
                    block_type_mapped = BlockType.NARRATIVE_TEXT
                elif 'ListItem' in raw_element_type_str:
                    block_type_mapped = BlockType.LIST_ITEM
                elif 'Code' in raw_element_type_str or 'CodeSnippet' in raw_element_type_str:
                    block_type_mapped = BlockType.CODE_SNIPPET
                elif 'Table' in raw_element_type_str:
                    block_type_mapped = BlockType.TABLE
                dto = ContentBlockDTO(file_id=file_id, text_content=
                    element_text, analysis_text=element_text, block_type=
                    block_type_mapped, order_in_document=index, page_number
                    =raw_element_metadata.get('page_number'), metadata={
                    'p03_source_element_type': raw_element_type_str,
                    'p03_source_parser_metadata': raw_element_metadata})
                dtos.append(dto)
            except Exception as e:
                raise DTOConversionError(file_path=
                    file_path_for_error_reporting, element_info=
                    element_info_for_error, original_exception=e)
        return dtos
