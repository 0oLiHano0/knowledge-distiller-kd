"""
SQLAlchemy ORM模型定义，用于知识库的数据结构。
定义了Document（文档）、Block（文本块）、Analysis（分析结果）和Decision（决策）四个主要实体。
"""

from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, func, Index
import json
from datetime import datetime
from typing import Dict, Any, Optional

# 声明基类
Base = declarative_base()


class Document(Base):
    """
    文档模型，代表一个被处理的文件。
    存储文件路径、哈希值、文件类型、大小等基本信息。
    """
    __tablename__ = 'files'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(String, unique=True, nullable=False)  # 业务唯一UUID，不允许为空
    path = Column(String, unique=True, nullable=False)
    file_hash = Column(String, nullable=False)
    type = Column(String)
    size = Column(Integer)
    ctime = Column(DateTime)
    mtime = Column(DateTime)
    ingest_time = Column(DateTime, server_default=func.now())
    status = Column(String)
    
    # 添加唯一索引，使用 IGNORE 策略
    __table_args__ = (
        Index('ix_files_path', 'path', unique=True, sqlite_where=None),
    )
    
    # 关系：一个Document可以有多个Block
    blocks = relationship('Block', back_populates='document', cascade='all, delete-orphan')
    
    def __repr__(self) -> str:
        return f"Document(id={self.id}, file_id='{self.file_id}', path='{self.path}', status='{self.status}')"


class Block(Base):
    """
    文本块模型，代表从文档中提取的内容块。
    存储块内容、哈希值、处理状态等信息。
    """
    __tablename__ = 'blocks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(Integer, ForeignKey('files.id', ondelete='CASCADE'), nullable=False)
    block_id = Column(String(64), nullable=False)  # 使用content_hash作为block_id
    content_hash = Column(String(64), nullable=False)  # 内容哈希值，不允许为空
    simhash = Column(String(64))
    text = Column(Text, nullable=False)
    block_type = Column(String(32), nullable=False)
    processing_status = Column(String(32), default='processed')
    meta_data = Column(JSON)
    
    # 关系
    document = relationship('Document', back_populates='blocks')
    analysis_results = relationship('Analysis', back_populates='block', cascade='all, delete-orphan')
    decisions = relationship('Decision', back_populates='block', cascade='all, delete-orphan', foreign_keys='Decision.block_id')
    duplicate_decisions = relationship('Decision', back_populates='duplicate_of', cascade='all, delete-orphan', foreign_keys='Decision.duplicate_of_block_id')
    
    def __repr__(self):
        return f"<Block(id={self.id}, block_id='{self.block_id}', type='{self.block_type}')>"


class Analysis(Base):
    """
    分析结果模型，存储对两个Block之间的分析数据。
    可以包含相似度评分、重复检测结果等。
    """
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True)
    result_id = Column(String(64), unique=True, nullable=False)  # 唯一标识符
    block_id_1 = Column(String(64), nullable=False)  # 第一个块的ID
    block_id_2 = Column(String(64), nullable=False)  # 第二个块的ID
    block_id = Column(Integer, ForeignKey('blocks.id', ondelete='CASCADE'), nullable=False)  # 历史兼容字段
    analysis_type = Column(String(32), nullable=False)
    score = Column(JSON, nullable=False)
    details = Column(JSON)
    
    # 关系
    block = relationship('Block', back_populates='analysis_results', foreign_keys=[block_id])
    decisions = relationship('Decision', back_populates='analysis_result', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Analysis(id={self.id}, type='{self.analysis_type}', result_id='{self.result_id}')>"


class Decision(Base):
    """
    决策模型，记录对分析结果的处理决策。
    例如标记为重复、删除、保留等决策。
    """
    __tablename__ = 'user_decisions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    decision_id = Column(String(64), unique=True, nullable=False)  # 唯一标识符
    result_id = Column(String(64), ForeignKey('analysis_results.result_id'), nullable=False)  # 关联到分析结果
    block_id = Column(Integer, ForeignKey('blocks.id', ondelete='CASCADE'), nullable=True)  # 历史兼容字段
    decision_type = Column(String, nullable=False)
    duplicate_of_block_id = Column(Integer, ForeignKey('blocks.id'), nullable=True)  # 历史兼容字段
    timestamp = Column(DateTime, server_default=func.now())
    comment = Column(Text)
    
    # 关系：Decision关联到一个Analysis
    analysis_result = relationship('Analysis', back_populates='decisions', foreign_keys=[result_id])
    # 保留以下关系用于历史兼容
    block = relationship('Block', back_populates='decisions', foreign_keys=[block_id])
    duplicate_of = relationship('Block', back_populates='duplicate_decisions', foreign_keys=[duplicate_of_block_id])
    
    def __repr__(self) -> str:
        return f"Decision(id={self.id}, decision_id='{self.decision_id}', type='{self.decision_type}')" 