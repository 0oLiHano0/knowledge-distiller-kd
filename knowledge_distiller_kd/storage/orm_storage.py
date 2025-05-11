"""
SQLite 数据库存储实现，使用 SQLAlchemy ORM。
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import hashlib

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

from knowledge_distiller_kd.core.models import (
    AnalysisResult, AnalysisType, BlockType, ContentBlock, DecisionType,
    FileRecord, UserDecision
)
from knowledge_distiller_kd.storage.storage_interface import StorageInterface
from knowledge_distiller_kd.storage.sqlite_storage import SessionLocal
from knowledge_distiller_kd.storage.models_sqlalchemy import (
    Document, Block, Analysis, Decision
)

logger = logging.getLogger(__name__)

class ORMStorage(StorageInterface):
    """
    SQLite 数据库存储实现，使用 SQLAlchemy ORM。
    实现了 StorageInterface 的所有方法。
    """

    def __init__(self, *args, **kwargs):
        """初始化 ORMStorage。"""
        self._initialized = False
        logger.debug("ORMStorage initialized")

    def initialize(self) -> None:
        """初始化数据库连接。"""
        if self._initialized:
            logger.debug("ORMStorage already initialized")
            return

        try:
            # 测试数据库连接
            with SessionLocal() as session:
                session.execute(text("SELECT 1"))
            self._initialized = True
            logger.info("ORMStorage initialized successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize ORMStorage: {e}")
            raise

    def _ensure_initialized(self):
        """确保存储已初始化。"""
        if not self._initialized:
            logger.warning("Storage accessed before initialization")
            self.initialize()
            if not self._initialized:
                raise RuntimeError("Storage could not be initialized")

    def register_file(self, filepath: str) -> str:
        """注册文件并返回文件ID。"""
        self._ensure_initialized()
        normalized_path = str(Path(filepath).resolve())
        file_hash = hashlib.md5(normalized_path.encode()).hexdigest()
        file_id = str(uuid.uuid4())  # 生成唯一的 file_id

        with SessionLocal() as session:
            # 检查文件是否已注册
            existing_doc = session.query(Document).filter_by(path=normalized_path).first()
            if existing_doc:
                return existing_doc.file_id

            # 创建新文档记录
            doc = Document(
                file_id=file_id,  # 使用生成的 file_id
                path=normalized_path,
                file_hash=file_hash,
                status="registered",
                type="",  # 设置默认值
                size=0,   # 设置默认值
                ctime=None,  # 设置默认值
                mtime=None   # 设置默认值
            )
            session.add(doc)
            session.commit()
            return doc.file_id

    def get_file_record(self, file_id: str) -> Optional[FileRecord]:
        """获取文件记录。"""
        self._ensure_initialized()
        with SessionLocal() as session:
            doc = session.query(Document).filter_by(file_id=file_id).first()
            if doc:
                return FileRecord(
                    file_id=doc.file_id,
                    original_path=doc.path
                )
            return None

    def list_files(self) -> List[FileRecord]:
        """列出所有注册的文件。"""
        self._ensure_initialized()
        with SessionLocal() as session:
            docs = session.query(Document).all()
            return [
                FileRecord(
                    file_id=doc.file_id,
                    original_path=doc.path
                )
                for doc in docs
            ]

    def save_blocks(self, file_id: str, blocks: List[ContentBlock]) -> None:
        """保存内容块。"""
        self._ensure_initialized()
        with SessionLocal() as session:
            # 首先获取 Document 记录
            doc = session.query(Document).filter_by(file_id=file_id).first()
            if not doc:
                logger.error(f"Document with file_id {file_id} not found")
                return

            for block in blocks:
                # 检查块是否已存在
                existing_block = session.query(Block).filter_by(block_id=block.block_id).first()
                if existing_block:
                    # 更新现有块
                    existing_block.text = block.text
                    existing_block.block_type = block.block_type.value
                    existing_block.meta_data = block.metadata
                else:
                    # 创建新块
                    new_block = Block(
                        block_id=block.block_id,
                        file_id=doc.id,  # 使用 Document 的 id 作为外键
                        content_hash=block.content_hash if hasattr(block, 'content_hash') else None,
                        text=block.text,
                        block_type=block.block_type.value,
                        meta_data=block.metadata,
                        processing_status="processed"
                    )
                    session.add(new_block)
            session.commit()

    def get_block(self, block_id: str) -> Optional[ContentBlock]:
        """获取单个内容块。"""
        self._ensure_initialized()
        with SessionLocal() as session:
            block = session.query(Block).filter_by(block_id=block_id).first()
            if block:
                return ContentBlock(
                    block_id=block.block_id,
                    file_id=block.file_id,
                    text=block.text,
                    block_type=BlockType(block.block_type),
                    metadata=block.meta_data
                )
            return None

    def get_blocks_by_file(self, file_id: str) -> List[ContentBlock]:
        """获取文件的所有内容块。"""
        self._ensure_initialized()
        with SessionLocal() as session:
            blocks = session.query(Block).filter_by(file_id=file_id).all()
            return [
                ContentBlock(
                    block_id=block.block_id,
                    file_id=block.file_id,
                    text=block.text,
                    block_type=BlockType(block.block_type),
                    metadata=block.meta_data
                )
                for block in blocks
            ]

    def get_blocks_for_analysis(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[ContentBlock]:
        """获取用于分析的内容块。"""
        self._ensure_initialized()
        with SessionLocal() as session:
            query = session.query(Block)
            
            if filter_criteria:
                if "file_id" in filter_criteria:
                    query = query.filter_by(file_id=filter_criteria["file_id"])
                if "block_type" in filter_criteria:
                    query = query.filter_by(block_type=filter_criteria["block_type"].value)
            
            blocks = query.all()
            return [
                ContentBlock(
                    block_id=block.block_id,
                    file_id=block.file_id,
                    text=block.text,
                    block_type=BlockType(block.block_type),
                    metadata=block.meta_data
                )
                for block in blocks
            ]

    def save_analysis_result(self, analysis_type: AnalysisType, result_data: List[AnalysisResult]) -> None:
        """保存分析结果。"""
        self._ensure_initialized()
        with SessionLocal() as session:
            for result in result_data:
                # 查找 Block.id
                block = session.query(Block).filter_by(block_id=result.block_id_1).first()
                if not block:
                    continue  # 跳过未找到的块
                analysis = Analysis(
                    block_id=block.id,
                    analysis_type=analysis_type.value,
                    score=result.score,
                    details=getattr(result, 'details', None)
                )
                session.add(analysis)
            session.commit()

    def get_analysis_results(self, analysis_type: AnalysisType, filter_criteria: Optional[Dict[str, Any]] = None) -> List[AnalysisResult]:
        """获取分析结果。"""
        self._ensure_initialized()
        with SessionLocal() as session:
            query = session.query(Analysis).filter_by(analysis_type=analysis_type.value)
            
            if filter_criteria:
                if "min_score" in filter_criteria:
                    query = query.filter(Analysis.score >= filter_criteria["min_score"])
                if "block_id" in filter_criteria:
                    query = query.filter(Analysis.block_id == filter_criteria["block_id"])
            
            results = query.all()
            return [
                AnalysisResult(
                    analysis_type=AnalysisType(result.analysis_type),
                    block_id_1=str(result.block_id),
                    block_id_2=str(result.block_id),  # 占位
                    score=result.score
                )
                for result in results
            ]

    def save_user_decision(self, decision_data: UserDecision) -> None:
        """保存用户决策。"""
        self._ensure_initialized()
        with SessionLocal() as session:
            # 查找 Block.id
            block = session.query(Block).filter_by(block_id=decision_data.block_id_1).first()
            if not block:
                return  # 跳过未找到的块
            decision = Decision(
                block_id=block.id,
                decision_type=decision_data.decision.value,
                comment=decision_data.notes
            )
            session.add(decision)
            session.commit()

    def get_user_decisions(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[UserDecision]:
        """获取用户决策。"""
        self._ensure_initialized()
        with SessionLocal() as session:
            query = session.query(Decision)
            
            if filter_criteria:
                if "decision" in filter_criteria:
                    query = query.filter_by(decision_type=filter_criteria["decision"].value)
                if "block_id" in filter_criteria:
                    # 反查 Block.id
                    block = session.query(Block).filter_by(block_id=filter_criteria["block_id"]).first()
                    if block:
                        query = query.filter_by(block_id=block.id)
                    else:
                        return []
            decisions = query.all()
            result = []
            for decision in decisions:
                block = session.query(Block).filter_by(id=decision.block_id).first()
                block_id_str = block.block_id if block else str(decision.block_id)
                result.append(UserDecision(
                    analysis_type=AnalysisType.SEMANTIC_SIMILARITY,  # 占位
                    block_id_1=block_id_str,
                    block_id_2=block_id_str,  # 占位
                    decision=DecisionType(decision.decision_type),
                    notes=decision.comment
                ))
            return result

    def get_undecided_pairs(self, analysis_type: AnalysisType) -> List[AnalysisResult]:
        """获取未决策的分析结果对。"""
        self._ensure_initialized()
        with SessionLocal() as session:
            analyses = session.query(Analysis).filter_by(analysis_type=analysis_type.value).all()
            decisions = session.query(Decision).all()
            decided_block_ids = {d.block_id for d in decisions}
            undecided = [a for a in analyses if a.block_id not in decided_block_ids]
            return [
                AnalysisResult(
                    analysis_type=AnalysisType(a.analysis_type),
                    block_id_1=str(a.block_id),
                    block_id_2=str(a.block_id),  # 占位
                    score=a.score
                )
                for a in undecided
            ]

    def finalize(self) -> None:
        """清理资源。"""
        logger.info("Finalizing ORMStorage")
        self._initialized = False 