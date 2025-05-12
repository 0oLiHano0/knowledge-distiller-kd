# -*- coding: utf-8 -*-
# knowledge_distiller_kd/core/models.py
"""
Defines the core Data Transfer Objects (DTOs) and Enumerations used throughout the application,
particularly for data exchange between layers (e.g., storage, analysis, UI).
"""

import datetime
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field, model_validator

# SQLAlchemy ORM 基础组件
from sqlalchemy import Column, String, Float, Boolean, Integer, Text, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship, declarative_base

# 创建SQLAlchemy Base类
Base = declarative_base()

# --- Enumerations ---

class BlockType(Enum):
    UNKNOWN = "unknown"
    TEXT = "text"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    CODE = "code"
    CODE_MERGED = "code_merged"
    TABLE = "table"


class AnalysisType(Enum):
    UNKNOWN = "unknown"
    MD5_DUPLICATE = "md5_duplicate"
    SEMANTIC_SIMILARITY = "semantic_similarity"


class DecisionType(Enum):
    UNDECIDED = "undecided"
    KEEP = "keep"  # 0508新增：合并后保留状态
    MERGE = "merge"
    IGNORE = "ignore"
    MARK_DUPLICATE = "mark_duplicate"
    MARK_SIMILAR = "mark_similar"
    DELETE = "delete"


# --- Dataclasses / DTOs ---

@dataclass
class ContentBlock:
    """
    表示文档中的一个内容块。
    包含块的基本信息、元数据和状态信息。
    """
    # 基本字段
    file_id: str
    text: str
    block_type: BlockType

    # 可选字段
    block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_path: str = ""
    analysis_text: str = ""
    original_text: str = field(init=False)
    content_hash: str = field(init=False)  # 新增字段：内容的MD5哈希值
    
    # 新增字段
    created_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    status: str = "pending"  # pending, processed, deleted
    parent_block_id: Optional[str] = None
    version: int = 1
    tags: List[str] = field(default_factory=list)
    char_count: int = 0
    token_count: int = 0
    processing_status: DecisionType = DecisionType.UNDECIDED
    duplicate_of_block_id: Optional[str] = None

    def __post_init__(self):
        """初始化后处理：设置默认值和计算派生值"""
        # 设置默认的 analysis_text
        if not self.analysis_text:
            self.analysis_text = self.text
            
        # 设置默认的 file_path
        if 'original_path' in self.metadata and not self.file_path:
            self.file_path = self.metadata['original_path']
            
        # 设置 original_text
        self.original_text = self.text
        
        # 计算字符数和词数
        self.char_count = len(self.text)
        self.token_count = len(self.text.split())
        
        # 计算内容哈希值
        import hashlib
        self.content_hash = hashlib.md5(self.text.encode('utf-8')).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """将 ContentBlock 序列化为字典"""
        return {
            "block_id": self.block_id,
            "file_id": self.file_id,
            "text": self.text,
            "block_type": self.block_type.value,
            "metadata": self.metadata,
            "file_path": self.file_path,
            "analysis_text": self.analysis_text,
            "original_text": self.original_text,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status,
            "parent_block_id": self.parent_block_id,
            "version": self.version,
            "tags": self.tags,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "processing_status": self.processing_status.value,
            "duplicate_of_block_id": self.duplicate_of_block_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentBlock':
        """从字典反序列化为 ContentBlock"""
        if "file_id" not in data or "text" not in data:
            raise ValueError("Missing required fields 'file_id' or 'text'")
            
        try:
            block_type = BlockType(data.get("block_type", BlockType.UNKNOWN.value))
        except ValueError:
            logger.warning(f"Invalid BlockType '{data.get('block_type')}', defaulting to UNKNOWN.")
            block_type = BlockType.UNKNOWN
            
        try:
            processing_status = DecisionType(data.get("processing_status", DecisionType.UNDECIDED.value))
        except ValueError:
            logger.warning(f"Invalid DecisionType '{data.get('processing_status')}', defaulting to UNDECIDED.")
            processing_status = DecisionType.UNDECIDED

        # 处理时间字段
        created_at = datetime.datetime.now(datetime.timezone.utc)
        if "created_at" in data:
            try:
                created_at = datetime.datetime.fromisoformat(data["created_at"])
            except ValueError:
                logger.warning(f"Invalid created_at format. Using current time.")

        updated_at = datetime.datetime.now(datetime.timezone.utc)
        if "updated_at" in data:
            try:
                updated_at = datetime.datetime.fromisoformat(data["updated_at"])
            except ValueError:
                logger.warning(f"Invalid updated_at format. Using current time.")

        instance = cls(
            block_id=data.get("block_id", str(uuid.uuid4())),
            file_id=data["file_id"],
            text=data["text"],
            block_type=block_type,
            metadata=data.get("metadata", {}),
            file_path=data.get("file_path", ""),
            analysis_text=data.get("analysis_text", data["text"]),
            created_at=created_at,
            updated_at=updated_at,
            status=data.get("status", "pending"),
            parent_block_id=data.get("parent_block_id"),
            version=data.get("version", 1),
            tags=data.get("tags", []),
            char_count=data.get("char_count", 0),
            token_count=data.get("token_count", 0),
            processing_status=processing_status,
            duplicate_of_block_id=data.get("duplicate_of_block_id")
        )
        
        # 如果字典中有original_text，使用它；否则默认使用text
        if "original_text" in data:
            instance.original_text = data["original_text"]
            
        # 如果字典中有content_hash，使用它；否则会在__post_init__中计算
        if "content_hash" in data:
            instance.content_hash = data["content_hash"]
            
        return instance


@dataclass
class AnalysisResult:
    """
    表示两个内容块之间的分析结果。
    包含分析的基本信息、分数和详细信息。
    """
    # 基本字段
    block_id_1: str
    block_id_2: str
    analysis_type: AnalysisType
    score: Optional[float] = None
    
    # 新增字段
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    created_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    analysis_version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, reviewed, archived
    review_count: int = 0
    last_reviewed_at: Optional[datetime.datetime] = None
    last_reviewer_id: Optional[str] = None
    
    # 生成字段
    result_id: str = field(init=False)

    def __post_init__(self):
        """生成确定性的 result_id"""
        sorted_ids = sorted([self.block_id_1, self.block_id_2])
        id_string = f"{sorted_ids[0]}_{sorted_ids[1]}_{self.analysis_type.value}"
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        self.result_id = str(uuid.uuid5(namespace, id_string))

    def to_dict(self) -> Dict[str, Any]:
        """将 AnalysisResult 序列化为字典"""
        return {
            "result_id": self.result_id,
            "block_id_1": self.block_id_1,
            "block_id_2": self.block_id_2,
            "analysis_type": self.analysis_type.value,
            "score": self.score,
            "details": self.details,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "analysis_version": self.analysis_version,
            "metadata": self.metadata,
            "status": self.status,
            "review_count": self.review_count,
            "last_reviewed_at": self.last_reviewed_at.isoformat() if self.last_reviewed_at else None,
            "last_reviewer_id": self.last_reviewer_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """从字典反序列化为 AnalysisResult"""
        if "block_id_1" not in data or "block_id_2" not in data or "analysis_type" not in data:
            raise ValueError("Missing required fields in AnalysisResult data")
            
        try:
            analysis_type = AnalysisType(data.get("analysis_type", AnalysisType.UNKNOWN.value))
        except ValueError:
            logger.warning(f"Invalid AnalysisType '{data.get('analysis_type')}', defaulting to UNKNOWN.")
            analysis_type = AnalysisType.UNKNOWN

        # 处理时间字段
        created_at = datetime.datetime.now(datetime.timezone.utc)
        if "created_at" in data:
            try:
                created_at = datetime.datetime.fromisoformat(data["created_at"])
            except ValueError:
                logger.warning(f"Invalid created_at format. Using current time.")

        last_reviewed_at = None
        if "last_reviewed_at" in data and data["last_reviewed_at"]:
            try:
                last_reviewed_at = datetime.datetime.fromisoformat(data["last_reviewed_at"])
            except ValueError:
                logger.warning(f"Invalid last_reviewed_at format. Setting to None.")

        return cls(
            block_id_1=data["block_id_1"],
            block_id_2=data["block_id_2"],
            analysis_type=analysis_type,
            score=data.get("score"),
            details=data.get("details", {}),
            confidence=data.get("confidence", 0.0),
            created_at=created_at,
            analysis_version=data.get("analysis_version", "1.0.0"),
            metadata=data.get("metadata", {}),
            status=data.get("status", "pending"),
            review_count=data.get("review_count", 0),
            last_reviewed_at=last_reviewed_at,
            last_reviewer_id=data.get("last_reviewer_id")
        )


@dataclass
class UserDecision:
    """
    表示用户对分析结果的决策。
    包含决策的基本信息、状态和审核信息。
    """
    # 基本字段
    block_id_1: str
    block_id_2: str
    analysis_type: AnalysisType
    decision: DecisionType = DecisionType.UNDECIDED
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    notes: Optional[str] = None
    
    # 新增字段
    user_id: Optional[str] = None
    reviewed_at: Optional[datetime.datetime] = None
    reviewer_id: Optional[str] = None
    decision_reason: Optional[str] = None
    priority: int = 0  # 0: 普通, 1: 高, 2: 紧急
    status: str = "pending"  # pending, reviewed, executed
    execution_time: Optional[datetime.datetime] = None
    execution_status: Optional[str] = None
    execution_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 生成字段
    decision_id: str = field(init=False)

    def __post_init__(self):
        """生成确定性的 decision_id"""
        sorted_ids = sorted([self.block_id_1, self.block_id_2])
        id_string = f"{sorted_ids[0]}_{sorted_ids[1]}_{self.analysis_type.value}"
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        self.decision_id = str(uuid.uuid5(namespace, id_string))

    def to_dict(self) -> Dict[str, Any]:
        """将 UserDecision 序列化为字典"""
        return {
            "decision_id": self.decision_id,
            "block_id_1": self.block_id_1,
            "block_id_2": self.block_id_2,
            "analysis_type": self.analysis_type.value,
            "decision": self.decision.value,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
            "user_id": self.user_id,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewer_id": self.reviewer_id,
            "decision_reason": self.decision_reason,
            "priority": self.priority,
            "status": self.status,
            "execution_time": self.execution_time.isoformat() if self.execution_time else None,
            "execution_status": self.execution_status,
            "execution_notes": self.execution_notes,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserDecision':
        """从字典反序列化为 UserDecision"""
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

        # 处理时间字段
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        if "timestamp" in data:
            try:
                timestamp = datetime.datetime.fromisoformat(data["timestamp"])
            except ValueError:
                logger.warning(f"Invalid timestamp format. Using current time.")

        reviewed_at = None
        if "reviewed_at" in data and data["reviewed_at"]:
            try:
                reviewed_at = datetime.datetime.fromisoformat(data["reviewed_at"])
            except ValueError:
                logger.warning(f"Invalid reviewed_at format. Setting to None.")

        execution_time = None
        if "execution_time" in data and data["execution_time"]:
            try:
                execution_time = datetime.datetime.fromisoformat(data["execution_time"])
            except ValueError:
                logger.warning(f"Invalid execution_time format. Setting to None.")

        return cls(
            block_id_1=data["block_id_1"],
            block_id_2=data["block_id_2"],
            analysis_type=analysis_type,
            decision=decision,
            timestamp=timestamp,
            notes=data.get("notes"),
            user_id=data.get("user_id"),
            reviewed_at=reviewed_at,
            reviewer_id=data.get("reviewer_id"),
            decision_reason=data.get("decision_reason"),
            priority=data.get("priority", 0),
            status=data.get("status", "pending"),
            execution_time=execution_time,
            execution_status=data.get("execution_status"),
            execution_notes=data.get("execution_notes"),
            metadata=data.get("metadata", {})
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

# --- Czkawka 适配器相关 DTO ---

class DuplicateFileInfoDTO(BaseModel):
    path: str                   # 原始文件路径字符串
    size: int                   # 文件大小 (bytes)
    modified: Optional[int] = None  # 修改时间戳 (optional)

class DuplicateFileGroupDTO(BaseModel):
    files: List[DuplicateFileInfoDTO]  # 一组精确重复的文件列表
    header: Optional[str] = None       # （可选）原始 header 信息

class CzkawkaConfigDTO(BaseModel):
    czkawka_cli_path: str = "czkawka_cli"
    default_args: List[str] = ["duplicates", "--json", "-d"]

# --- SQLAlchemy ORM Models ---

class FileEntity(Base):
    """文件记录的SQLAlchemy ORM实体"""
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String(36), unique=True, nullable=False)
    original_path = Column(String(512), nullable=False)
    registration_time = Column(DateTime, nullable=False)
    
    # 关系
    blocks = relationship("BlockEntity", back_populates="file")
    
    def __repr__(self) -> str:
        return f"<FileEntity file_id={self.file_id}, path={self.original_path}>"


class BlockEntity(Base):
    """内容块的SQLAlchemy ORM实体"""
    __tablename__ = "blocks"
    
    id = Column(Integer, primary_key=True, index=True)
    block_id = Column(String(36), unique=True, nullable=False)
    file_id = Column(String(36), ForeignKey("files.file_id"), nullable=False)
    text = Column(Text, nullable=False)
    block_type = Column(String(50), nullable=False)
    
    # 关系
    file = relationship("FileEntity", back_populates="blocks")
    analysis_results_1 = relationship(
        "AnalysisResultEntity", 
        foreign_keys="AnalysisResultEntity.block_id_1",
        back_populates="block_1"
    )
    analysis_results_2 = relationship(
        "AnalysisResultEntity", 
        foreign_keys="AnalysisResultEntity.block_id_2",
        back_populates="block_2"
    )
    
    def __repr__(self) -> str:
        return f"<BlockEntity block_id={self.block_id}, type={self.block_type}>"


class AnalysisResultEntity(Base):
    """分析结果的SQLAlchemy ORM实体"""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(String(36), unique=True, nullable=False)
    block_id_1 = Column(String(36), ForeignKey("blocks.block_id"), nullable=False)
    block_id_2 = Column(String(36), ForeignKey("blocks.block_id"), nullable=False)
    analysis_type = Column(String(50), nullable=False)
    score = Column(Float, nullable=True)
    
    # 关系
    block_1 = relationship("BlockEntity", foreign_keys=[block_id_1], back_populates="analysis_results_1")
    block_2 = relationship("BlockEntity", foreign_keys=[block_id_2], back_populates="analysis_results_2")
    decision = relationship("UserDecisionEntity", back_populates="analysis_result", uselist=False)
    
    def __repr__(self) -> str:
        return f"<AnalysisResultEntity result_id={self.result_id}, type={self.analysis_type}>"


class UserDecisionEntity(Base):
    """用户决策的SQLAlchemy ORM实体"""
    __tablename__ = "user_decisions"
    
    id = Column(Integer, primary_key=True, index=True)
    decision_id = Column(String(36), unique=True, nullable=False)
    result_id = Column(String(36), ForeignKey("analysis_results.result_id"), nullable=False, unique=True)
    decision = Column(String(50), nullable=False, default="undecided")  
    timestamp = Column(DateTime, nullable=False)
    notes = Column(Text, nullable=True)
    
    # 关系
    analysis_result = relationship("AnalysisResultEntity", back_populates="decision")
    
    def __repr__(self) -> str:
        return f"<UserDecisionEntity decision_id={self.decision_id}, decision={self.decision}>"