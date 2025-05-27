"""
=================================================
document_processing_stage.py - P03 文档处理阶段 (v4.6)
=================================================
... (模块注释保持不变) ...
---
"""
from typing import List, Any, Dict
from pathlib import Path
from loguru import Logger
import datetime
import uuid
from ....core.interfaces import StageInterface, StorageInterface
from ....core.dtos import PipelineContextDTO
from ....schemas.dtos import ContentBlockDTO, FileRecordDTO
from ....schemas.enums import BlockType, ProcessingStatus
from kd_tool.stages.docprocessing.settings_models import DocumentProcessingStageSettings
from kd_tool.stages.docprocessing.errors import DocumentProcessingError, FileReadError, ParsingError, DTOConversionError, UnsupportedFileTypeError


class _InternalParserWrapper:

    def __init__(self, logger: Logger, strategy: str):
        self._logger = logger.bind(parser_wrapper=self.__class__.__name__)
        self._strategy = strategy
        self._logger.info(
            f"InternalParserWrapper initialized with strategy: '{self._strategy}'."
            )

    def parse_file_to_raw_elements(self, file_path: Path) ->List[Dict[str, Any]
        ]:
        self._logger.debug(
            f'Attempting to parse file: {file_path} using strategy: {self._strategy}'
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

    def __init__(self, logger: Logger, storage: StorageInterface, settings:
        DocumentProcessingStageSettings) ->None:
        self._logger = logger.bind(stage='DocumentProcessingStageP03')
        self._storage = storage
        self._settings = settings
        self._parser = _InternalParserWrapper(logger=self._logger, strategy
            =self._settings.parsing_strategy)
        self._logger.info(
            f"DocumentProcessingStage (P03 - Raw Extraction) initialized. Supported extensions: {self._settings.supported_extensions}. Parsing strategy: '{self._settings.parsing_strategy}'."
            )

    def process(self, context: PipelineContextDTO) ->PipelineContextDTO:
        task_id = context.get_task_id_str()
        logger = context.run_logger.bind(stage='DocumentProcessingStageP03')
        logger.info('P03 - Document Processing (Raw Extraction) starting...')
        if not self._settings.enabled:
            logger.warning(
                'P03 - DocumentProcessingStage is disabled. Skipping.')
            return context
        files_to_process = [record for record in context.file_records.
            values() if record.processing_status == ProcessingStatus.PENDING]
        if not files_to_process:
            logger.info(
                'P03 - No pending files found for document processing. Skipping.'
                )
            return context
        logger.info(
            f'P03 - Found {len(files_to_process)} pending files to process.')
        processed_file_count = 0
        total_blocks_generated_in_run = 0
        for file_record in files_to_process:
            file_path = file_record.original_path
            file_id = file_record.file_id
            current_file_block_count = 0
            logger.debug(f'P03 - Processing file: {file_path} (ID: {file_id})')
            try:
                if file_path.suffix.lower(
                    ) not in self._settings.supported_extensions:
                    raise UnsupportedFileTypeError(file_path,
                        detected_file_type=file_path.suffix)
                raw_elements = self._parser.parse_file_to_raw_elements(
                    file_path)
                if not raw_elements:
                    logger.warning(
                        f'P03 - No raw elements extracted from {file_path}. File processed as empty.'
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
                    processed_file_count += 1
                    continue
                preliminary_blocks = self._convert_raw_elements_to_dtos(
                    raw_elements, file_id=file_id,
                    file_path_for_error_reporting=file_path)
                for block_dto in preliminary_blocks:
                    context.add_content_block(block_dto)
                    current_file_block_count += 1
                total_blocks_generated_in_run += current_file_block_count
                file_record.processing_status = (ProcessingStatus.
                    BLOCK_EXTRACTION_COMPLETED)
                file_record.metadata['p03_extracted_block_count'
                    ] = current_file_block_count
                file_record.processing_history.append({'stage': 'P03',
                    'status': ProcessingStatus.BLOCK_EXTRACTION_COMPLETED.
                    value, 'timestamp': datetime.datetime.now(datetime.
                    timezone.utc).isoformat(), 'details':
                    f'Extracted {current_file_block_count} preliminary blocks.'
                    })
                processed_file_count += 1
                logger.success(
                    f'P03 - Successfully extracted {current_file_block_count} preliminary blocks from: {file_path}'
                    )
            except DocumentProcessingError as e:
                logger.error(
                    f'P03 - Controlled error during processing of {file_path}: {e.message}'
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
                    f'P03 - Unexpected critical error processing file {file_path}.'
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
        logger.info(
            f'P03 - Document Processing (Raw Extraction) finished. Successfully processed files: {processed_file_count}/{len(files_to_process)}. Total preliminary blocks generated in this run: {total_blocks_generated_in_run}.'
            )
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
                f"Raw element at index {index} (type: {raw_element.get('type', 'N/A')})"
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
