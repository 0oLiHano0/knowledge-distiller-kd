# knowledge_distiller_kd/storage/file_storage.py
"""
Implements the StorageInterface using the local file system,
persisting data as JSON files. Uses pathlib for path manipulation
and core.models DTOs for data exchange.
(Version fixing initialization TypeError)
"""

import datetime
import json
import logging
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Callable

# Import DTOs and Interface definition
from knowledge_distiller_kd.core.models import (
    AnalysisResult, AnalysisType, BlockType, ContentBlock, DecisionType,
    FileRecord, UserDecision
)
from knowledge_distiller_kd.storage.storage_interface import StorageInterface
# Import potential custom errors if defined
# from knowledge_distiller_kd.core.exceptions import StorageError, ConfigurationError

# Setup logger for this module
logger = logging.getLogger(__name__)

# Generic TypeVar for DTOs used in helper functions
T_DTO = TypeVar('T_DTO', ContentBlock, AnalysisResult, UserDecision, FileRecord)

# --- Helper Functions for JSON Serialization/Deserialization ---

def _default_serializer(obj: Any) -> Union[str, Dict[str, Any], Any]:
    """
    Default JSON serializer for custom objects like datetime, Enum, Path, and DTOs.
    Uses the DTO's to_dict() method if available.
    """
    if isinstance(obj, datetime.datetime):
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=datetime.timezone.utc)
        return obj.isoformat()
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, 'to_dict') and callable(obj.to_dict):
        return obj.to_dict()
    # Let default JSON encoder handle standard types or raise TypeError
    try:
        # Check if obj is directly serializable first
        json.dumps(obj)
        return obj
    except TypeError:
         # If standard dump fails, raise TypeError to be caught by caller
         raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _save_json_file(path: Path, data: Any) -> None:
    """
    Saves data to a JSON file using the custom serializer (_default_serializer).
    Ensures the parent directory exists.

    Args:
        path (Path): The path to the JSON file.
        data (Any): The data to serialize and save.

    Raises:
        IOError: If file writing fails.
        TypeError: If data contains non-serializable types not handled by _default_serializer.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4, default=_default_serializer)
    except (IOError, TypeError) as e:
        logger.exception(f"Failed to save JSON data to {path}: {e}")
        raise e


def _load_json_file(path: Path, default_factory: Callable[[], Any]) -> Any:
    """
    Loads JSON data from a file.
    Handles file not found, empty file, and JSON decode errors gracefully.
    Returns the result of default_factory() in case of errors or if the file is missing/empty.

    Args:
        path (Path): The path to the JSON file.
        default_factory (Callable[[], Any]): A function that returns the default
                                             structure (e.g., dict, list).

    Returns:
        Any: The loaded JSON data or the default structure.
    """
    default_data = None # Placeholder
    if not path.exists():
        logger.warning(f"JSON file not found: {path}. Initializing with default structure.")
        try:
            default_data = default_factory() # Call factory to get default
            _save_json_file(path, default_data) # Create with default
            return default_data
        except Exception as e_save:
            logger.error(f"Failed to create default file {path}: {e_save}")
            # If we couldn't even save the default, return the factory result directly
            # or raise an error depending on desired robustness
            return default_factory() if default_data is None else default_data


    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                 logger.warning(f"JSON file is empty: {path}. Returning default structure.")
                 return default_factory() # Return default for empty file
            return json.loads(content)
    except json.JSONDecodeError:
        logger.exception(f"Failed to decode JSON from {path}. Replacing with default structure.")
        try:
            corrupt_path = path.with_suffix(f".corrupt.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
            path.rename(corrupt_path)
            logger.info(f"Backed up corrupted file to: {corrupt_path}")
        except OSError as backup_err:
            logger.error(f"Could not back up corrupted file {path}: {backup_err}")

        try:
            default_data = default_factory() # Get default
            _save_json_file(path, default_data) # Overwrite with default
            return default_data
        except Exception as e_save:
             logger.error(f"Failed to save default file after JSON decode error {path}: {e_save}")
             # Return factory result or raise
             return default_factory() if default_data is None else default_data

    except IOError as e:
        logger.exception(f"IOError reading file {path}: {e}. Returning default structure.")
        return default_factory()
    except Exception as e:
        logger.exception(f"Unexpected error reading file {path}: {e}. Returning default structure.")
        return default_factory()


def _deserialize_dto(data: Dict[str, Any], dto_class: Type[T_DTO]) -> Optional[T_DTO]:
    """
    Helper to safely deserialize a dictionary into a specific DTO class using its from_dict method.
    """
    if not isinstance(data, dict): # Add check if data is actually a dict
        logger.warning(f"Cannot deserialize non-dict data into {dto_class.__name__}: {type(data)}")
        return None
    if not hasattr(dto_class, 'from_dict'):
        logger.warning(f"Cannot deserialize - {dto_class.__name__} lacks from_dict method.")
        return None
    try:
        return dto_class.from_dict(data)
    except Exception as e:
        logger.exception(f"Failed to deserialize dict into {dto_class.__name__}: {data}. Error: {e}")
        return None

# --- Default Structure Factories ---
# Define these as simple functions to avoid potential lambda issues

def _default_metadata_factory() -> Dict[str, Dict]:
    """Returns the default structure for the metadata file."""
    return {"files": {}, "path_to_id": {}}

def _default_blocks_factory() -> Dict:
    """Returns the default structure for the blocks file."""
    return {}

def _default_results_factory() -> Dict:
    """Returns the default structure for the results file."""
    return {}

def _default_decisions_factory() -> Dict:
    """Returns the default structure for the decisions file."""
    return {}


# --- FileStorage Implementation ---

class FileStorage(StorageInterface):
    """
    File system based implementation of the StorageInterface.
    Stores data in JSON files within a specified base directory.
    """
    _metadata_filename = "metadata.json"
    _blocks_filename = "blocks.json"
    _results_filename = "results.json"
    _decisions_filename = "decisions.json"

    # *** No longer using lambda defaults here ***

    def __init__(self, base_path: Path, *args, **kwargs):
        """Initializes the FileStorage."""
        if not isinstance(base_path, Path):
            raise TypeError("base_path must be a pathlib.Path object")

        self.base_path = base_path.resolve()
        self._metadata_path = self.base_path / self._metadata_filename
        self._blocks_path = self.base_path / self._blocks_filename
        self._results_path = self.base_path / self._results_filename
        self._decisions_path = self.base_path / self._decisions_filename

        self._metadata: Dict[str, Any] = {}
        self._blocks: Dict[str, Dict[str, Any]] = {}
        self._results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._decisions: Dict[str, Dict[str, Any]] = {}

        self._initialized = False
        logger.debug(f"FileStorage initialized with base path: {self.base_path}")


    def initialize(self) -> None:
        """Sets up the storage directory and loads existing data into memory caches."""
        if self._initialized:
            logger.debug("FileStorage already initialized.")
            return

        logger.info(f"Initializing FileStorage at: {self.base_path}")
        initialization_successful = False
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)

            # *** Use the dedicated factory functions ***
            self._metadata = _load_json_file(self._metadata_path, _default_metadata_factory)
            self._blocks = _load_json_file(self._blocks_path, _default_blocks_factory)
            self._results = _load_json_file(self._results_path, _default_results_factory)
            self._decisions = _load_json_file(self._decisions_path, _default_decisions_factory)

            # Basic type validation after loading
            if not isinstance(self._metadata, dict): self._metadata = _default_metadata_factory()
            if not isinstance(self._blocks, dict): self._blocks = _default_blocks_factory()
            if not isinstance(self._results, dict): self._results = _default_results_factory()
            if not isinstance(self._decisions, dict): self._decisions = _default_decisions_factory()

            initialization_successful = True # Mark success only if all loads are okay
            logger.info("FileStorage initialization complete. Data loaded/initialized.")

        except Exception as e:
            logger.exception(f"Critical error during FileStorage initialization: {e}")
            # Leave in uninitialized state
        finally:
             # Set _initialized flag based on success
             self._initialized = initialization_successful
             if not self._initialized:
                  logger.error("FileStorage initialization failed.")


    def _ensure_initialized(self):
        """Helper to check if storage is initialized before operations."""
        if not self._initialized:
            logger.warning("Storage accessed before explicit initialization or initialization failed. Attempting to initialize now.")
            self.initialize()
            if not self._initialized: # Check again if init failed
                 raise RuntimeError("Storage could not be initialized. Check logs for details.")


    # --- File Registration ---
    def register_file(self, filepath: str) -> str:
        """Registers a file path, returning a unique file ID."""
        self._ensure_initialized()
        normalized_path_str = str(Path(filepath).resolve())

        # Ensure nested dicts exist before access
        if "path_to_id" not in self._metadata: self._metadata["path_to_id"] = {}
        if "files" not in self._metadata: self._metadata["files"] = {}

        if normalized_path_str in self._metadata["path_to_id"]:
            file_id = self._metadata["path_to_id"][normalized_path_str]
            logger.debug(f"File path '{filepath}' already registered with ID: {file_id}")
            return file_id
        else:
            file_id = str(uuid.uuid4())
            record = FileRecord(file_id=file_id, original_path=filepath)
            record_dict = record.to_dict()

            self._metadata["files"][file_id] = record_dict
            self._metadata["path_to_id"][normalized_path_str] = file_id

            try:
                _save_json_file(self._metadata_path, self._metadata)
                logger.info(f"Registered new file '{filepath}' with ID: {file_id}")
                return file_id
            except Exception as e:
                self._metadata["files"].pop(file_id, None)
                self._metadata["path_to_id"].pop(normalized_path_str, None)
                logger.error(f"Failed to save metadata after registering file '{filepath}': {e}")
                raise e


    def get_file_record(self, file_id: str) -> Optional[FileRecord]:
        """Retrieves the FileRecord for a given file ID."""
        self._ensure_initialized()
        record_dict = self._metadata.get("files", {}).get(file_id)
        if record_dict:
            return _deserialize_dto(record_dict, FileRecord)
        return None

    def list_files(self) -> List[FileRecord]:
        """Lists all registered files."""
        self._ensure_initialized()
        files_dict = self._metadata.get("files", {})
        records = []
        for record_dict in files_dict.values():
            record = _deserialize_dto(record_dict, FileRecord)
            if record:
                records.append(record)
            else:
                logger.warning(f"Could not deserialize file record: {record_dict}")
        return records

    # --- Content Block Operations ---
    def save_blocks(self, file_id: str, blocks: List[ContentBlock]) -> None:
        """Saves or updates a list of content blocks."""
        self._ensure_initialized()
        if not blocks:
            logger.debug("No blocks provided to save.")
            return

        updated = False
        for block in blocks:
            if not isinstance(block, ContentBlock):
                logger.warning(f"Skipping non-ContentBlock item in save_blocks: {block}")
                continue
            if not block.block_id:
                 logger.error(f"ContentBlock is missing block_id: {block}")
                 continue

            block_dict = block.to_dict()
            if block.block_id not in self._blocks or self._blocks[block.block_id] != block_dict:
                self._blocks[block.block_id] = block_dict
                updated = True
        if updated:
            try:
                _save_json_file(self._blocks_path, self._blocks)
                logger.info(f"Saved/Updated {len(blocks)} blocks. Total blocks: {len(self._blocks)}")
            except Exception as e:
                logger.error(f"Failed to save blocks file after updates: {e}")
                raise e


    def get_block(self, block_id: str) -> Optional[ContentBlock]:
        """Retrieves a specific content block by its ID."""
        self._ensure_initialized()
        block_dict = self._blocks.get(block_id)
        if block_dict:
            return _deserialize_dto(block_dict, ContentBlock)
        return None

    def get_blocks_by_file(self, file_id: str) -> List[ContentBlock]:
        """Retrieves all content blocks associated with a specific file ID."""
        self._ensure_initialized()
        matching_blocks = []
        for block_id, block_dict in self._blocks.items(): # Iterate items for debugging clarity
            if block_dict.get("file_id") == file_id:
                block = _deserialize_dto(block_dict, ContentBlock)
                if block:
                    matching_blocks.append(block)
                else:
                     logger.warning(f"Could not deserialize block {block_id} with file_id {file_id}: {block_dict}")
        return matching_blocks


    def get_blocks_for_analysis(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[ContentBlock]:
        """Retrieves content blocks based on filter criteria."""
        self._ensure_initialized()
        all_blocks = list(self._blocks.values()) # Get a list of block dicts
        if not filter_criteria:
            # No filter, return all deserialized blocks
            return [dto for b in all_blocks if (dto := _deserialize_dto(b, ContentBlock))]

        matching_blocks = []
        for block_dict in all_blocks:
            match = True
            # Apply filters
            if "file_id" in filter_criteria and block_dict.get("file_id") != filter_criteria["file_id"]:
                match = False
            if "block_type" in filter_criteria:
                filter_type = filter_criteria["block_type"]
                if isinstance(filter_type, BlockType):
                    if block_dict.get("block_type") != filter_type.value: match = False
                elif isinstance(filter_type, str):
                     if block_dict.get("block_type") != filter_type: match = False
                else:
                    logger.warning(f"Unsupported block_type filter value: {filter_type}")
                    match = False

            if match:
                block = _deserialize_dto(block_dict, ContentBlock)
                if block:
                    matching_blocks.append(block)
                else:
                    logger.warning(f"Could not deserialize block matching criteria: {block_dict}")

        return matching_blocks

    # --- Analysis Result Operations ---
    def save_analysis_result(self, analysis_type: AnalysisType, result_data: List[AnalysisResult]) -> None:
        """Saves or updates a list of analysis results for a specific type."""
        self._ensure_initialized()
        if not result_data:
            logger.debug("No analysis results provided to save.")
            return

        analysis_type_str = analysis_type.value
        updated = False

        if analysis_type_str not in self._results:
            self._results[analysis_type_str] = {}
            updated = True

        for result in result_data:
            if not isinstance(result, AnalysisResult):
                logger.warning(f"Skipping non-AnalysisResult item: {result}")
                continue
            if result.analysis_type != analysis_type:
                 logger.warning(f"Mismatch: Result type {result.analysis_type} != expected {analysis_type}. Skipping.")
                 continue
            if not result.result_id:
                 logger.error(f"AnalysisResult is missing result_id: {result}")
                 continue

            result_dict = result.to_dict()
            if result.result_id not in self._results[analysis_type_str] or \
               self._results[analysis_type_str][result.result_id] != result_dict:
                self._results[analysis_type_str][result.result_id] = result_dict
                updated = True

        if updated:
            try:
                _save_json_file(self._results_path, self._results)
                logger.info(f"Saved/Updated {len(result_data)} results for type '{analysis_type_str}'.")
            except Exception as e:
                logger.error(f"Failed to save results file after updates for type '{analysis_type_str}': {e}")
                raise e


    def get_analysis_results(self, analysis_type: AnalysisType, filter_criteria: Optional[Dict[str, Any]] = None) -> List[AnalysisResult]:
        """Retrieves analysis results based on type and filters."""
        self._ensure_initialized()
        analysis_type_str = analysis_type.value
        results_for_type = self._results.get(analysis_type_str, {})
        matching_results = []

        if not results_for_type:
            return []

        filter_criteria = filter_criteria or {}
        min_score = filter_criteria.get("min_score")
        block_id_filter = filter_criteria.get("block_id")

        for result_id, result_dict in results_for_type.items(): # Iterate items for debugging
            match = True
            # Apply filters
            if min_score is not None:
                score = result_dict.get("score")
                if score is None or score < min_score: match = False
            if block_id_filter is not None:
                if result_dict.get("block_id_1") != block_id_filter and \
                   result_dict.get("block_id_2") != block_id_filter: match = False

            if match:
                result = _deserialize_dto(result_dict, AnalysisResult)
                if result:
                    if result.analysis_type == analysis_type:
                        matching_results.append(result)
                    else:
                         logger.warning(f"Mismatch: Loaded result {result_id} type {result.analysis_type} != requested {analysis_type}. Skipping.")
                else:
                    logger.warning(f"Could not deserialize result {result_id} matching criteria: {result_dict}")

        return matching_results

    # --- User Decision Operations ---
    def save_user_decision(self, decision_data: UserDecision) -> None:
        """Saves or updates a user's decision."""
        self._ensure_initialized()
        if not isinstance(decision_data, UserDecision):
            logger.error(f"Invalid data type passed to save_user_decision: {type(decision_data)}")
            return

        if not decision_data.decision_id:
             logger.error(f"UserDecision is missing decision_id: {decision_data}")
             return

        decision_dict = decision_data.to_dict()
        updated = False

        if decision_data.decision_id not in self._decisions or \
           self._decisions[decision_data.decision_id] != decision_dict:
            self._decisions[decision_data.decision_id] = decision_dict
            updated = True

        if updated:
            try:
                _save_json_file(self._decisions_path, self._decisions)
                logger.info(f"Saved/Updated decision ID: {decision_data.decision_id}")
            except Exception as e:
                logger.error(f"Failed to save decisions file after update for ID {decision_data.decision_id}: {e}")
                raise e


    def get_user_decisions(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[UserDecision]:
        """Retrieves user decisions based on filter criteria."""
        self._ensure_initialized()
        if not self._decisions:
            return []

        matching_decisions = []
        filter_criteria = filter_criteria or {}

        analysis_type_filter = filter_criteria.get("analysis_type")
        if isinstance(analysis_type_filter, AnalysisType): analysis_type_filter = analysis_type_filter.value
        decision_filter = filter_criteria.get("decision")
        if isinstance(decision_filter, DecisionType): decision_filter = decision_filter.value
        block_id_filter = filter_criteria.get("block_id")
        decision_id_filter = filter_criteria.get("decision_id")

        for decision_id, decision_dict in self._decisions.items(): # Iterate items
            match = True
            # Apply filters
            if decision_id_filter is not None and decision_dict.get("decision_id", decision_id) != decision_id_filter: match = False # Use key if dict missing id
            if analysis_type_filter is not None and decision_dict.get("analysis_type") != analysis_type_filter: match = False
            if decision_filter is not None and decision_dict.get("decision") != decision_filter: match = False
            if block_id_filter is not None:
                 if decision_dict.get("block_id_1") != block_id_filter and \
                    decision_dict.get("block_id_2") != block_id_filter: match = False

            if match:
                decision = _deserialize_dto(decision_dict, UserDecision)
                if decision:
                    matching_decisions.append(decision)
                else:
                     logger.warning(f"Could not deserialize decision {decision_id} matching criteria: {decision_dict}")

        return matching_decisions


    def get_undecided_pairs(self, analysis_type: AnalysisType) -> List[AnalysisResult]:
        """Retrieves analysis results for a type that lack a definitive user decision."""
        self._ensure_initialized()
        analysis_type_str = analysis_type.value
        results_for_type = self._results.get(analysis_type_str, {})
        undecided_results = []

        if not results_for_type:
            return []

        for result_id, result_dict in results_for_type.items():
            decision = self._decisions.get(result_id)
            is_decided = False
            if decision:
                decision_enum_val = decision.get("decision")
                if decision_enum_val and decision_enum_val != DecisionType.UNDECIDED.value:
                    is_decided = True

            if not is_decided:
                result = _deserialize_dto(result_dict, AnalysisResult)
                if result:
                     if result.analysis_type == analysis_type:
                         undecided_results.append(result)
                     else:
                          logger.warning(f"Mismatch: Undecided result {result.result_id} has type {result.analysis_type}, expected {analysis_type}. Skipping.")
                else:
                    logger.warning(f"Could not deserialize undecided result {result_id}: {result_dict}")

        return undecided_results


    def finalize(self) -> None:
        """Performs cleanup tasks."""
        logger.info("Finalizing FileStorage (no specific actions required).")
        self._initialized = False

