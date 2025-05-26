"""
此模块定义了项目使用的 SQLAlchemy ORM 模型，
这些模型直接映射到数据库中的表结构。
"""
from sqlalchemy import Column, String, Integer, DateTime, Text, Float, ForeignKey, Enum as SAEnum, JSON
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from datetime import datetime, timezone
from pathlib import Path
import uuid
from kd_tool.schemas.enums import BlockType, AnalysisType, DecisionType, ProcessingStatus
Base = declarative_base()


class FileOrmModel(Base):
    __tablename__ = 'files'
    file_id: str = Column(String, primary_key=True, default=lambda :
        f'file_{uuid.uuid4()}')
    original_path: str = Column(String, nullable=False, unique=True)
    file_hash_md5: str = Column(String(32), nullable=True, index=True)
    size_bytes: int = Column(Integer, nullable=True)
    last_modified_at: datetime = Column(DateTime(timezone=True), nullable=True)
    registered_at: datetime = Column(DateTime(timezone=True), nullable=
        False, default=lambda : datetime.now(timezone.utc))
    processing_status: ProcessingStatus = Column(SAEnum(ProcessingStatus,
        name='processing_status_enum', native_enum=False), nullable=False,
        default=ProcessingStatus.PENDING)
    processing_history = Column(SQLiteJSON, nullable=True, default=list)
    metadata_ = Column('metadata', SQLiteJSON, nullable=True, default=dict)
    content_blocks = relationship('ContentBlockOrmModel', back_populates=
        'file', cascade='all, delete-orphan')

    def __repr__(self):
        return (
            f"<FileOrmModel(file_id='{self.file_id}', path='{self.original_path}')>"
            )


class ContentBlockOrmModel(Base):
    __tablename__ = 'content_blocks'
    block_id: str = Column(String, primary_key=True, default=lambda :
        f'block_{uuid.uuid4()}')
    file_id: str = Column(String, ForeignKey('files.file_id'), nullable=
        False, index=True)
    text_content: str = Column(Text, nullable=False)
    analysis_text: str = Column(Text, nullable=True)
    block_type: BlockType = Column(SAEnum(BlockType, name='block_type_enum',
        native_enum=False), nullable=False)
    order_in_document: int = Column(Integer, nullable=True)
    page_number: int = Column(Integer, nullable=True)
    text_hash_md5: str = Column(String(32), nullable=True, index=True)
    simhash_value: str = Column(String(16), nullable=True, index=True)
    metadata_ = Column('metadata', SQLiteJSON, nullable=True, default=dict)
    file = relationship('FileOrmModel', back_populates='content_blocks')
    analysis_results_as_block1 = relationship('AnalysisResultOrmModel',
        foreign_keys='[AnalysisResultOrmModel.block_id_1]', back_populates=
        'block1', cascade='all, delete-orphan')
    analysis_results_as_block2 = relationship('AnalysisResultOrmModel',
        foreign_keys='[AnalysisResultOrmModel.block_id_2]', back_populates=
        'block2', cascade='all, delete-orphan')

    def __repr__(self):
        return (
            f"<ContentBlockOrmModel(block_id='{self.block_id}', type='{self.block_type.value if self.block_type else None}')>"
            )


class AnalysisResultOrmModel(Base):
    __tablename__ = 'analysis_results'
    pair_analysis_id: str = Column(String(32), primary_key=True)
    block_id_1: str = Column(String, ForeignKey('content_blocks.block_id'),
        nullable=False, index=True)
    block_id_2: str = Column(String, ForeignKey('content_blocks.block_id'),
        nullable=False, index=True)
    analysis_type: AnalysisType = Column(SAEnum(AnalysisType, name=
        'analysis_type_enum', native_enum=False), nullable=False)
    score: float = Column(Float, nullable=True)
    details: Dict[str, Any] = Column(SQLiteJSON, nullable=True, default=dict)
    block1 = relationship('ContentBlockOrmModel', foreign_keys=[block_id_1],
        back_populates='analysis_results_as_block1')
    block2 = relationship('ContentBlockOrmModel', foreign_keys=[block_id_2],
        back_populates='analysis_results_as_block2')
    user_decision = relationship('UserDecisionOrmModel', back_populates=
        'analysis_result', uselist=False, cascade='all, delete-orphan')

    def __repr__(self):
        return (
            f"<AnalysisResultOrmModel(pair_analysis_id='{self.pair_analysis_id}', type='{self.analysis_type.value if self.analysis_type else None}')>"
            )


class UserDecisionOrmModel(Base):
    __tablename__ = 'user_decisions'
    pair_analysis_id: str = Column(String(32), ForeignKey(
        'analysis_results.pair_analysis_id'), primary_key=True)
    decision: DecisionType = Column(SAEnum(DecisionType, name=
        'decision_type_enum', native_enum=False), nullable=False, default=
        DecisionType.UNDECIDED)
    decided_at: datetime = Column(DateTime(timezone=True), nullable=False,
        default=lambda : datetime.now(timezone.utc))
    decided_by: str = Column(String, nullable=True)
    notes: str = Column(Text, nullable=True)
    analysis_result = relationship('AnalysisResultOrmModel', back_populates
        ='user_decision')

    def __repr__(self):
        return (
            f"<UserDecisionOrmModel(pair_analysis_id='{self.pair_analysis_id}', decision='{self.decision.value if self.decision else None}')>"
            )
