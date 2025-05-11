"""
SQLAlchemy ORM模型定义，用于知识库的数据结构。
定义了Document（文档）、Block（文本块）、Analysis（分析结果）和Decision（决策）四个主要实体。
"""

from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, func, Index

# 声明基类
Base = declarative_base()


class Document(Base):
    """
    文档模型，代表一个被处理的文件。
    存储文件路径、哈希值、文件类型、大小等基本信息。
    """
    __tablename__ = 'files'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
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
        return f"Document(id={self.id}, path='{self.path}', status='{self.status}')"


class Block(Base):
    """
    文本块模型，代表从文档中提取的内容块。
    存储块内容、哈希值、处理状态等信息。
    """
    __tablename__ = 'blocks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    block_id = Column(String, nullable=False)
    file_id = Column(Integer, ForeignKey('files.id', ondelete='CASCADE'), nullable=False)
    content_hash = Column(String)
    simhash = Column(String)
    text = Column(Text)
    block_type = Column(String)
    processing_status = Column(String)
    meta_data = Column(JSON)  # 改名为meta_data，避免与SQLAlchemy的metadata冲突
    
    # 关系：Block属于一个Document
    document = relationship('Document', back_populates='blocks')
    # Block可以有多个Analysis
    analyses = relationship('Analysis', back_populates='block', cascade='all, delete-orphan')
    
    def __repr__(self) -> str:
        return f"Block(id={self.id}, file_id={self.file_id}, type='{self.block_type}')"


class Analysis(Base):
    """
    分析结果模型，存储对Block的各种分析数据。
    可以包含相似度评分、重复检测结果等。
    """
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    block_id = Column(Integer, ForeignKey('blocks.id', ondelete='CASCADE'), nullable=False)
    analysis_type = Column(String, nullable=False)
    score = Column(JSON)
    details = Column(JSON)
    
    # 关系：Analysis关联到一个Block
    block = relationship('Block', back_populates='analyses')
    
    def __repr__(self) -> str:
        return f"Analysis(id={self.id}, block_id={self.block_id}, type='{self.analysis_type}')"


class Decision(Base):
    """
    决策模型，记录对Block的处理决策。
    例如标记为重复、删除、保留等决策。
    """
    __tablename__ = 'user_decisions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    block_id = Column(Integer, ForeignKey('blocks.id', ondelete='CASCADE'), nullable=False)
    decision_type = Column(String, nullable=False)
    duplicate_of_block_id = Column(Integer, ForeignKey('blocks.id'))
    timestamp = Column(DateTime, server_default=func.now())
    comment = Column(Text)
    
    # 关系：Decision关联到一个Block
    block = relationship('Block', foreign_keys=[block_id])
    duplicate_of = relationship('Block', foreign_keys=[duplicate_of_block_id])
    
    def __repr__(self) -> str:
        return f"Decision(id={self.id}, block_id={self.block_id}, type='{self.decision_type}')" 