# -*- coding: utf-8 -*-
# knowledge_distiller_kd/core/models.py
"""
Defines the core Data Transfer Objects (DTOs) and Enumerations used throughout the application,
particularly for data exchange between layers (e.g., storage, analysis, UI).
"""

import datetime
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# --- Enumerations ---

class BlockType(Enum):
    UNKNOWN = "unknown"
    TEXT = "text"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    CODE = "code"
    TABLE = "table"


class AnalysisType(Enum):
    UNKNOWN = "unknown"
    MD5_DUPLICATE = "md5_duplicate"
    SEMANTIC_SIMILARITY = "semantic_similarity"


class DecisionType(Enum):
    UNDECIDED = "undecided"
    MERGE = "merge"
    IGNORE = "ignore"
    MARK_DUPLICATE = "mark_duplicate"
    MARK_SIMILAR = "mark_similar"
    DELETE = "delete"


# --- Dataclasses / DTOs ---

@dataclass
class ContentBlock:
    file_id: str
    text: str
    block_type: BlockType

    block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "file_id": self.file_id,
            "text": self.text,
            "block_type": self.block_type.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentBlock':
        if "file_id" not in data or "text" not in data:
            raise ValueError("Missing required fields 'file_id' or 'text'")
        try:
            block_type = BlockType(data.get("block_type", BlockType.UNKNOWN.value))
        except ValueError:
            logger.warning(f"Invalid BlockType '{data.get('block_type')}', defaulting to UNKNOWN.")
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
    block_id_1: str
    block_id_2: str
    analysis_type: AnalysisType

    score: Optional[float] = None
    result_id: str = field(init=False)

    def __post_init__(self):
        sorted_ids = sorted([self.block_id_1, self.block_id_2])
        id_string = f"{sorted_ids[0]}_{sorted_ids[1]}_{self.analysis_type.value}"
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        self.result_id = str(uuid.uuid5(namespace, id_string))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "block_id_1": self.block_id_1,
            "block_id_2": self.block_id_2,
            "analysis_type": self.analysis_type.value,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        if "block_id_1" not in data or "block_id_2" not in data or "analysis_type" not in data:
            raise ValueError("Missing required fields in AnalysisResult data")
        try:
            analysis_type = AnalysisType(data.get("analysis_type", AnalysisType.UNKNOWN.value))
        except ValueError:
            logger.warning(f"Invalid AnalysisType '{data.get('analysis_type')}', defaulting to UNKNOWN.")
            analysis_type = AnalysisType.UNKNOWN

        return cls(
            block_id_1=data["block_id_1"],
            block_id_2=data["block_id_2"],
            analysis_type=analysis_type,
            score=data.get("score"),
        )


@dataclass
class UserDecision:
    block_id_1: str
    block_id_2: str
    analysis_type: AnalysisType

    decision: DecisionType = DecisionType.UNDECIDED
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    notes: Optional[str] = None
    decision_id: str = field(init=False)

    def __post_init__(self):
        sorted_ids = sorted([self.block_id_1, self.block_id_2])
        id_string = f"{sorted_ids[0]}_{sorted_ids[1]}_{self.analysis_type.value}"
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        self.decision_id = str(uuid.uuid5(namespace, id_string))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "block_id_1": self.block_id_1,
            "block_id_2": self.block_id_2,
            "analysis_type": self.analysis_type.value,
            "decision": self.decision.value,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserDecision':
        if "block_id_1" not in data or "block_id_2" not in data or "analysis_type" not in data:
            raise ValueError("Missing required fields in UserDecision data")
        try:
            analysis_type = AnalysisType(data.get("analysis_type", AnalysisType.UNKNOWN.value))
        except ValueError:
            logger.warning(f"Invalid AnalysisType '{data.get('analysis_type')}', defaulting to UNKNOWN.")
            analysis_type = AnalysisType.UNKNOWN

        try:
            decision = DecisionType(data.get("decision", DecisionType.UNDECIDED.value))
        except ValueError:
            logger.warning(f"Invalid DecisionType '{data.get('decision')}', defaulting to UNDECIDED.")
            decision = DecisionType.UNDECIDED

        timestamp = datetime.datetime.now(datetime.timezone.utc)
        ts = data.get("timestamp")
        if ts:
            try:
                if ts.endswith('Z'):
                    ts = ts[:-1] + '+00:00'
                timestamp = datetime.datetime.fromisoformat(ts)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)
            except ValueError:
                logger.warning(f"Invalid timestamp '{data.get('timestamp')}', using now().")

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
    file_id: str
    original_path: str

    registration_time: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "original_path": self.original_path,
            "registration_time": self.registration_time.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileRecord':
        if "file_id" not in data or "original_path" not in data:
            raise ValueError("Missing required fields in FileRecord data")

        registration_time = datetime.datetime.now(datetime.timezone.utc)
        ts = data.get("registration_time")
        if ts:
            try:
                if ts.endswith('Z'):
                    ts = ts[:-1] + '+00:00'
                registration_time = datetime.datetime.fromisoformat(ts)
                if registration_time.tzinfo is None:
                    registration_time = registration_time.replace(tzinfo=datetime.timezone.utc)
            except ValueError:
                logger.warning(f"Invalid registration_time '{data.get('registration_time')}', using now().")

        return cls(
            file_id=data["file_id"],
            original_path=data["original_path"],
            registration_time=registration_time,
            metadata=data.get("metadata", {}),
        )


# --- Pydantic DTO with new validator style ---

class BlockDTO(BaseModel):
    block_id: str
    file_id: str
    block_type: BlockType
    text_content: str
    analysis_text: str
    char_count: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    kd_processing_status: DecisionType = DecisionType.UNDECIDED
    duplicate_of_block_id: Optional[str] = None

    @model_validator(mode="before")
    def ensure_analysis_text(cls, values: dict) -> dict:
        """
        如果没有传入 analysis_text，就自动用 text_content 填充，
        避免后续流程因缺失而报错。
        """
        if values.get("analysis_text") is None:
            values["analysis_text"] = values.get("text_content", "")
        return values


def normalize_text_for_analysis(text: str) -> str:
    """
    文本规范化占位函数，目前直接原样返回。
    如果需要更复杂的清洗，可以后续再改。
    """
    return text
