# -*- coding: utf-8 -*-
"""
Defines the abstract interface for the storage layer using ABC, ensuring consistency
across different storage implementations (e.g., file system, database).

Version 3:
- Removed the @runtime_checkable decorator as it's incompatible with abc.ABC.
- Kept ABC approach with @abstractmethod.
- Kept added methods: get_file_record, list_files, get_blocks_by_file.
- Kept filter_criteria optional in relevant get methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
# Removed runtime_checkable import
from typing import List, Optional, Dict, Any

# Use relative import assuming this file is in knowledge_distiller_kd/storage/
# and models.py is in knowledge_distiller_kd/core/
# Ensure core.models uses the latest version (kd_tool_core_models_v1_fixed_v3)
from ..core.models import (
    ContentBlock,
    AnalysisResult,
    UserDecision,
    FileRecord,
    AnalysisType # Import Enum for type hinting if needed directly
)

# Removed @runtime_checkable decorator
class StorageInterface(ABC):
    """
    Abstract Base Class defining the contract for storage operations
    within the Knowledge Distiller.
    Concrete implementations MUST inherit from this class and implement all abstract methods.
    """

    # __init__ should be defined by concrete implementations, not the interface.

    @abstractmethod
    def initialize(self) -> None:
        """
        Sets up the storage backend (e.g., creates directories, tables, connects).
        Should be idempotent (safe to call multiple times).
        """
        raise NotImplementedError

    @abstractmethod
    def register_file(self, filepath: str) -> str:
        """
        Registers a file path with the storage system and returns a unique file ID.
        If the file path has already been registered, it should return the existing ID.

        Args:
            filepath (str): The path to the file to be registered.

        Returns:
            str: A unique identifier (file_id) for the registered file.
        """
        raise NotImplementedError

    @abstractmethod
    def get_file_record(self, file_id: str) -> Optional[FileRecord]:
        """
        Retrieves the FileRecord associated with a given file ID.

        Args:
            file_id (str): The unique identifier of the file.

        Returns:
            Optional[FileRecord]: The FileRecord if found, otherwise None.
        """
        raise NotImplementedError

    @abstractmethod
    def list_files(self) -> List[FileRecord]:
        """
        Lists all files currently registered in the storage system.

        Returns:
            List[FileRecord]: A list of all registered file records.
        """
        raise NotImplementedError

    @abstractmethod
    def save_blocks(self, file_id: str, blocks: List[ContentBlock]) -> None:
        """
        Saves or updates a list of content blocks associated with a specific file ID.
        If a block with the same block_id already exists, it should be updated.

        Args:
            file_id (str): The file ID these blocks belong to (should match block.file_id).
                           Provided for potential optimization or validation.
            blocks (List[ContentBlock]): The list of content blocks to save.
        """
        raise NotImplementedError

    @abstractmethod
    def get_block(self, block_id: str) -> Optional[ContentBlock]:
        """
        Retrieves a specific content block by its unique ID.

        Args:
            block_id (str): The unique identifier of the content block.

        Returns:
            Optional[ContentBlock]: The ContentBlock if found, otherwise None.
        """
        raise NotImplementedError

    @abstractmethod
    def get_blocks_by_file(self, file_id: str) -> List[ContentBlock]:
        """
        Retrieves all content blocks associated with a specific file ID.

        Args:
            file_id (str): The unique identifier of the file.

        Returns:
            List[ContentBlock]: A list of content blocks for the given file.
        """
        raise NotImplementedError

    @abstractmethod
    def get_blocks_for_analysis(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[ContentBlock]:
        """
        Retrieves content blocks based on specified filter criteria,
        often used as input for analysis tasks.

        Args:
            filter_criteria (Optional[Dict[str, Any]]): A dictionary defining filters
                (e.g., {'file_id': '...', 'block_type': BlockType.TEXT}). If None or empty,
                potentially returns all blocks (implementation-defined).

        Returns:
            List[ContentBlock]: A list of content blocks matching the criteria.
        """
        raise NotImplementedError

    @abstractmethod
    def save_analysis_result(self, analysis_type: AnalysisType, result_data: List[AnalysisResult]) -> None:
        """
        Saves a list of analysis results for a specific analysis type.
        If a result with the same result_id already exists, it should be updated.

        Args:
            analysis_type (AnalysisType): The type of analysis these results belong to.
                                          Provided for potential optimization or validation.
            result_data (List[AnalysisResult]): The list of analysis results to save.
                                                 Each result's analysis_type should match.
        """
        raise NotImplementedError

    @abstractmethod
    def get_analysis_results(self, analysis_type: AnalysisType, filter_criteria: Optional[Dict[str, Any]] = None) -> List[AnalysisResult]:
        """
        Retrieves analysis results based on analysis type and filter criteria.

        Args:
            analysis_type (AnalysisType): The type of analysis results to retrieve.
            filter_criteria (Optional[Dict[str, Any]]): A dictionary defining filters
                (e.g., {'block_id_1': '...', 'min_score': 0.8}).

        Returns:
            List[AnalysisResult]: A list of analysis results matching the criteria.
        """
        raise NotImplementedError

    @abstractmethod
    def save_user_decision(self, decision_data: UserDecision) -> None:
        """
        Saves or updates a user's decision regarding an analysis result pair.
        Identified by the combination of block_id_1, block_id_2, and analysis_type
        (implicitly via decision_data.decision_id).

        Args:
            decision_data (UserDecision): The user decision object to save.
        """
        raise NotImplementedError

    @abstractmethod
    def get_user_decisions(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[UserDecision]:
        """
        Retrieves user decisions based on specified filter criteria.

        Args:
            filter_criteria (Optional[Dict[str, Any]]): A dictionary defining filters
                (e.g., {'analysis_type': AnalysisType.SEMANTIC_SIMILARITY, 'decision': DecisionType.KEEP_BOTH}).

        Returns:
            List[UserDecision]: A list of user decisions matching the criteria.
        """
        raise NotImplementedError

    @abstractmethod
    def get_undecided_pairs(self, analysis_type: AnalysisType) -> List[AnalysisResult]:
        """
        Retrieves analysis results for a specific type that do not yet have a
        corresponding user decision (other than UNDECIDED).

        Args:
            analysis_type (AnalysisType): The type of analysis to check for undecided pairs.

        Returns:
            List[AnalysisResult]: A list of analysis results awaiting a definitive user decision.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> None:
        """
        Performs any necessary cleanup or finalization tasks (e.g., closing connections,
        flushing buffers). Called when the storage instance is no longer needed.
        """
        raise NotImplementedError

    # Optional: Add methods for deleting items if necessary in the future
    # @abstractmethod
    # def delete_block(self, block_id: str) -> bool: ...
    # @abstractmethod
    # def delete_file(self, file_id: str) -> bool: ... # Consider cascading deletes

