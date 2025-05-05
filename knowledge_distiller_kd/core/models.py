# -*- coding: utf-8 -*-
# knowledge_distiller_kd/core/models.py
"""
Defines the core Data Transfer Objects (DTOs) and Enumerations used throughout the application,
particularly for data exchange between layers (e.g., storage, analysis, UI).
(Version confirmed to pass FileStorage tests - incorporates AI feedback)
"""

import datetime
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# --- Enumerations (Reflecting version that passed tests) ---

class BlockType(Enum):
    """Enumeration for different types of content blocks."""
    UNKNOWN = "unknown"
    TEXT = "text"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    CODE = "code" # Note: Consider if CODE_SNIPPET was more accurate previously
    TABLE = "table"
    # Add more types as needed based on 'unstructured' or specific needs

class AnalysisType(Enum):
    """Enumeration for different types of analysis performed."""
    UNKNOWN = "unknown"
    MD5_DUPLICATE = "md5_duplicate"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    # Add more types as needed

class DecisionType(Enum):
    """Enumeration for user decisions on analysis results."""
    UNDECIDED = "undecided"
    MERGE = "merge" # Note: Revisit if this accurately reflects needed actions vs. KEEP_*
    IGNORE = "ignore" # Note: Revisit if this accurately reflects needed actions vs. SKIP
    MARK_DUPLICATE = "mark_duplicate"
    MARK_SIMILAR = "mark_similar" # Note: Revisit if this accurately reflects needed actions
    # Add more types as needed

# --- Data Transfer Objects (DTOs) ---

@dataclass
class ContentBlock:
    """Represents a distinct block of content extracted from a file."""
    # Non-Default Fields first
    file_id: str
    text: str
    block_type: BlockType # Uses the Enum defined above

    # Default Fields after
    block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)


    def to_dict(self) -> Dict[str, Any]:
        """Serializes the ContentBlock to a dictionary."""
        return {
            "block_id": self.block_id,
            "file_id": self.file_id,
            "text": self.text,
            "block_type": self.block_type.value, # Store enum value
            "metadata": self.metadata,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentBlock':
        """Deserializes a dictionary into a ContentBlock object."""
        if "file_id" not in data or "text" not in data:
            raise ValueError("Missing required fields 'file_id' or 'text' in ContentBlock data")
        try:
            # Use UNKNOWN as the default if block_type is missing or invalid
            block_type_value = data.get("block_type", BlockType.UNKNOWN.value)
            block_type = BlockType(block_type_value)
        except ValueError:
            logger.warning(f"Invalid BlockType value '{block_type_value}' found for block {data.get('block_id')}. Defaulting to UNKNOWN.")
            block_type = BlockType.UNKNOWN

        return cls(
            block_id=data.get("block_id", str(uuid.uuid4())),
            file_id=data["file_id"],
            text=data["text"],
            block_type=block_type,
            metadata=data.get("metadata", {}),
            )

@dataclass
class AnalysisResult:
    """
    Represents the result of an analysis comparing two content blocks.
    Includes a deterministically generated result_id based on block IDs and type.
    """
    # Non-Default Fields first
    block_id_1: str
    block_id_2: str
    analysis_type: AnalysisType

    # Default Fields after
    score: Optional[float] = None

    # Generated Fields (using UUIDv5 for deterministic ID)
    result_id: str = field(init=False)

    def __post_init__(self):
        """Generate a deterministic result_id after initialization."""
        sorted_ids = sorted([self.block_id_1, self.block_id_2])
        id_string = f"{sorted_ids[0]}_{sorted_ids[1]}_{self.analysis_type.value}"
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8') # Example DNS namespace
        self.result_id = str(uuid.uuid5(namespace, id_string))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the AnalysisResult to a dictionary."""
        return {
            "result_id": self.result_id,
            "block_id_1": self.block_id_1,
            "block_id_2": self.block_id_2,
            "analysis_type": self.analysis_type.value, # Store enum value
            "score": self.score,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Deserializes a dictionary into an AnalysisResult object."""
        if "block_id_1" not in data or "block_id_2" not in data or "analysis_type" not in data:
            raise ValueError("Missing required fields 'block_id_1', 'block_id_2', or 'analysis_type' in AnalysisResult data")
        try:
            # Use UNKNOWN as default if analysis_type is missing or invalid
            analysis_type_value = data.get("analysis_type", AnalysisType.UNKNOWN.value)
            analysis_type = AnalysisType(analysis_type_value)
        except ValueError:
             logger.warning(f"Invalid AnalysisType value '{analysis_type_value}' found for result. Defaulting to UNKNOWN.")
             analysis_type = AnalysisType.UNKNOWN

        return cls(
            block_id_1=data["block_id_1"],
            block_id_2=data["block_id_2"],
            analysis_type=analysis_type,
            score=data.get("score"),
            )

@dataclass
class UserDecision:
    """
    Represents a user's decision regarding a pair of content blocks (AnalysisResult).
    Includes a deterministically generated decision_id based on block IDs and type.
    """
    # Non-Default Fields first
    block_id_1: str
    block_id_2: str
    analysis_type: AnalysisType

    # Default Fields after
    decision: DecisionType = DecisionType.UNDECIDED
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    notes: Optional[str] = None

    # Generated Fields (using UUIDv5 for deterministic ID, same as AnalysisResult's result_id)
    decision_id: str = field(init=False)

    def __post_init__(self):
        """Generate a deterministic decision_id after initialization."""
        sorted_ids = sorted([self.block_id_1, self.block_id_2])
        id_string = f"{sorted_ids[0]}_{sorted_ids[1]}_{self.analysis_type.value}"
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8') # Example DNS namespace
        self.decision_id = str(uuid.uuid5(namespace, id_string))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the UserDecision to a dictionary."""
        return {
            "decision_id": self.decision_id,
            "block_id_1": self.block_id_1,
            "block_id_2": self.block_id_2,
            "analysis_type": self.analysis_type.value, # Store enum value
            "decision": self.decision.value, # Store enum value
            "timestamp": self.timestamp.isoformat(), # Use ISO format for timestamps
            "notes": self.notes,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserDecision':
        """Deserializes a dictionary into a UserDecision object."""
        if "block_id_1" not in data or "block_id_2" not in data or "analysis_type" not in data:
            raise ValueError("Missing required fields 'block_id_1', 'block_id_2', or 'analysis_type' in UserDecision data")

        try:
            # Use UNKNOWN as default if analysis_type is missing or invalid
            analysis_type_value = data.get("analysis_type", AnalysisType.UNKNOWN.value)
            analysis_type = AnalysisType(analysis_type_value)
        except ValueError:
             logger.warning(f"Invalid AnalysisType value '{analysis_type_value}' found for decision. Defaulting to UNKNOWN.")
             analysis_type = AnalysisType.UNKNOWN

        try:
            # Use UNDECIDED as default if decision is missing or invalid
            decision_value = data.get("decision", DecisionType.UNDECIDED.value)
            decision = DecisionType(decision_value)
        except ValueError:
            logger.warning(f"Invalid DecisionType value '{decision_value}' found for decision {data.get('decision_id')}. Defaulting to UNDECIDED.")
            decision = DecisionType.UNDECIDED

        # Handle timestamp deserialization carefully
        timestamp_str = data.get("timestamp")
        timestamp = datetime.datetime.now(datetime.timezone.utc) # Default
        if timestamp_str:
            try:
                if timestamp_str.endswith('Z'):
                    timestamp_str = timestamp_str[:-1] + '+00:00'
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
                if timestamp.tzinfo is None:
                     timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)
            except ValueError:
                 logger.warning(f"Invalid timestamp format '{timestamp_str}'. Using current UTC time.")

        return cls(
            block_id_1=data["block_id_1"],
            block_id_2=data["block_id_2"],
            analysis_type=analysis_type,
            decision=decision,
            timestamp=timestamp,
            notes=data.get("notes"),
            )

@dataclass
class FileRecord:
    """
    Represents metadata associated with a registered file.
    Fields without defaults MUST come before fields with defaults.
    """
    # --- Non-Default Fields ---
    file_id: str          # Unique ID assigned by the storage system
    original_path: str    # The original path provided by the user (Corrected)

    # --- Default Fields ---
    registration_time: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict) # Keep metadata flexible


    def to_dict(self) -> Dict[str, Any]:
        """Serializes the FileRecord to a dictionary."""
        return {
            "file_id": self.file_id,
            "original_path": self.original_path,
            "registration_time": self.registration_time.isoformat(),
            "metadata": self.metadata,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileRecord':
        """Deserializes a dictionary into a FileRecord object."""
        if "file_id" not in data or "original_path" not in data:
            raise ValueError("Missing required fields 'file_id' or 'original_path' in FileRecord data")

        timestamp_str = data.get("registration_time")
        registration_time = datetime.datetime.now(datetime.timezone.utc) # Default
        if timestamp_str:
            try:
                if timestamp_str.endswith('Z'):
                    timestamp_str = timestamp_str[:-1] + '+00:00'
                registration_time = datetime.datetime.fromisoformat(timestamp_str)
                if registration_time.tzinfo is None:
                     registration_time = registration_time.replace(tzinfo=datetime.timezone.utc)
            except ValueError:
                 logger.warning(f"Invalid registration_time format '{timestamp_str}'. Using current UTC time.")

        return cls(
            file_id=data["file_id"],
            original_path=data["original_path"],
            registration_time=registration_time,
            metadata=data.get("metadata", {}),
            )
