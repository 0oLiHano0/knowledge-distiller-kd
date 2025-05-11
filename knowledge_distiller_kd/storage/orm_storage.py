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
        """
        保存内容块。
        
        Args:
            file_id (str): 文件业务ID
            blocks (List[ContentBlock]): 要保存的内容块列表
            
        Raises:
            ValueError: 如果文档不存在或blocks中的file_id与参数file_id不匹配
            SQLAlchemyError: 数据库操作异常
            RuntimeError: 如果存储未初始化
        """
        self._ensure_initialized()
        
        if not blocks:
            logger.debug(f"没有提供要保存的块")
            return
            
        with SessionLocal() as session:
            # 查找文档记录获取数据库ID
            doc = session.query(Document).filter_by(file_id=file_id).first()
            if not doc:
                logger.error(f"未找到file_id为 {file_id} 的文档记录")
                raise ValueError(f"未找到file_id为 {file_id} 的文档记录")
            
            doc_db_id = doc.id  # 获取文档的数据库ID，用作外键
            
            # 批量保存所有块，使用事务确保一致性
            try:
                for block in blocks:
                    # 验证块的file_id与指定的file_id匹配
                    if block.file_id != file_id:
                        logger.warning(f"块 {block.block_id} 的file_id ({block.file_id}) 与指定的file_id ({file_id}) 不匹配，已更正")
                        block.file_id = file_id
                    
                    # 确保content_hash字段有值
                    content_hash = block.content_hash
                    if not content_hash:
                        # 如果未提供content_hash，使用block_id作为默认值
                        content_hash = block.block_id
                        logger.warning(f"块 {block.block_id} 缺少content_hash，使用block_id作为替代")
                    
                    # 查询现有块
                    existing_block = session.query(Block).filter_by(block_id=block.block_id).first()
                    
                    if existing_block:
                        # 更新现有块
                        existing_block.text = block.text
                        existing_block.block_type = block.block_type.value
                        existing_block.content_hash = content_hash
                        existing_block.processing_status = "processed"
                        
                        # 更新元数据 (合并而非替换)
                        meta = existing_block.meta_data or {}
                        meta.update(block.metadata or {})
                        existing_block.meta_data = meta
                        
                        # 确保file_id关联正确
                        if existing_block.file_id != doc_db_id:
                            logger.info(f"更新块 {block.block_id} 的文档关联，从 {existing_block.file_id} 到 {doc_db_id}")
                            existing_block.file_id = doc_db_id
                    else:
                        # 创建新块
                        new_block = Block(
                            block_id=block.block_id,
                            file_id=doc_db_id,  # 使用文档的数据库ID作为外键
                            content_hash=content_hash,
                            text=block.text,
                            block_type=block.block_type.value,
                            meta_data=block.metadata or {},
                            processing_status="processed"
                        )
                        session.add(new_block)
                
                # 提交所有更改
                session.commit()
                logger.info(f"成功保存 {len(blocks)} 个块到文档 {file_id}")
            
            except Exception as e:
                session.rollback()
                logger.error(f"保存块时出错: {e}")
                raise

    def get_block(self, block_id: str) -> Optional[ContentBlock]:
        """
        获取单个内容块。
        
        Args:
            block_id (str): 块ID
            
        Returns:
            Optional[ContentBlock]: 内容块对象，如果未找到则返回None
            
        Raises:
            SQLAlchemyError: 数据库操作异常
            RuntimeError: 存储未初始化
        """
        self._ensure_initialized()
        logger.debug(f"获取块ID为 {block_id} 的内容")
        
        try:
            with SessionLocal() as session:
                # 查询块
                block = session.query(Block).filter_by(block_id=block_id).first()
                if not block:
                    logger.debug(f"块 {block_id} 未找到")
                    return None
                
                # 获取原始file_id (业务ID)
                doc = session.query(Document).get(block.file_id)
                if not doc:
                    logger.warning(f"块 {block_id} 关联的文档ID {block.file_id} 未找到，使用'unknown'作为file_id")
                    orig_file_id = "unknown"
                else:
                    orig_file_id = doc.file_id
                
                # 构建内容块对象
                content_block = ContentBlock(
                    block_id=block.block_id,
                    file_id=orig_file_id,  # 使用原始file_id而非数据库ID
                    text=block.text,
                    block_type=BlockType(block.block_type),
                    metadata=block.meta_data or {}
                )
                
                logger.debug(f"成功获取块 {block_id}，关联文档ID: {orig_file_id}")
                return content_block
                
        except SQLAlchemyError as e:
            logger.error(f"获取块 {block_id} 时发生数据库错误: {e}")
            raise
        except Exception as e:
            logger.error(f"获取块 {block_id} 时发生未预期错误: {e}")
            raise

    def get_blocks_by_file(self, file_id: str) -> List[ContentBlock]:
        """
        获取文件的所有内容块。
        
        Args:
            file_id (str): 文件ID
            
        Returns:
            List[ContentBlock]: 内容块列表
        """
        self._ensure_initialized()
        with SessionLocal() as session:
            # 首先查找文档记录获取数据库ID
            doc = session.query(Document).filter_by(file_id=file_id).first()
            if not doc:
                logger.error(f"未找到file_id为 {file_id} 的文档记录")
                return []
            
            # 使用文档的数据库ID查询相关块
            blocks = session.query(Block).filter_by(file_id=doc.id).all()
            
            if not blocks:
                logger.debug(f"文档 {file_id} 没有关联的内容块")
            
            # 转换为DTO对象
            return [
                ContentBlock(
                    block_id=block.block_id,
                    file_id=file_id,  # 返回原始file_id而非数据库ID
                    text=block.text,
                    block_type=BlockType(block.block_type),
                    metadata=block.meta_data or {}
                )
                for block in blocks
            ]

    def get_blocks_for_analysis(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[ContentBlock]:
        """
        获取用于分析的内容块。
        
        Args:
            filter_criteria (Optional[Dict[str, Any]]): 过滤条件，如file_id、block_type等
            
        Returns:
            List[ContentBlock]: 内容块列表
        """
        self._ensure_initialized()
        with SessionLocal() as session:
            # 构建基本查询
            query = session.query(Block)
            
            # 应用过滤条件
            if filter_criteria:
                if "file_id" in filter_criteria:
                    file_id = filter_criteria["file_id"]
                    # 获取文档的数据库ID
                    doc = session.query(Document).filter_by(file_id=file_id).first()
                    if doc:
                        query = query.filter_by(file_id=doc.id)
                    else:
                        logger.warning(f"未找到file_id为 {file_id} 的文档记录，过滤条件可能无效")
                        return []  # 如果找不到文档，直接返回空列表
                
                if "block_type" in filter_criteria:
                    block_type = filter_criteria["block_type"]
                    if isinstance(block_type, BlockType):
                        query = query.filter_by(block_type=block_type.value)
                    else:
                        query = query.filter_by(block_type=str(block_type))
                
                # 其他可能的过滤条件
                if "content_hash" in filter_criteria:
                    query = query.filter_by(content_hash=filter_criteria["content_hash"])
            
            # 执行查询获取结果
            blocks = query.all()
            
            # 创建结果列表
            result_blocks = []
            for block in blocks:
                # 获取原始file_id (业务ID)
                doc = session.query(Document).get(block.file_id)
                orig_file_id = doc.file_id if doc else "unknown"
                
                # 构建ContentBlock对象
                content_block = ContentBlock(
                    block_id=block.block_id,
                    file_id=orig_file_id,  # 使用原始file_id
                    text=block.text,
                    block_type=BlockType(block.block_type),
                    metadata=block.meta_data or {}
                )
                result_blocks.append(content_block)
            
            return result_blocks

    def save_analysis_result(self, analysis_type: AnalysisType, result_data: List[AnalysisResult]) -> None:
        """
        保存分析结果。
        
        Args:
            analysis_type (AnalysisType): 分析类型
            result_data (List[AnalysisResult]): 分析结果列表
            
        Raises:
            ValueError: 如果引用的块不存在
            SQLAlchemyError: 数据库操作异常
            RuntimeError: 如果存储未初始化
        """
        self._ensure_initialized()
        
        if not result_data:
            logger.debug("没有提供要保存的分析结果")
            return
            
        with SessionLocal() as session:
            try:
                for result in result_data:
                    # 查找相关的块
                    block1 = session.query(Block).filter_by(block_id=result.block_id_1).first()
                    if not block1:
                        error_msg = f"引用的块 {result.block_id_1} 不存在，无法保存分析结果"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    block2 = None
                    if result.block_id_2 and result.block_id_2 != result.block_id_1:
                        block2 = session.query(Block).filter_by(block_id=result.block_id_2).first()
                        if not block2:
                            error_msg = f"引用的块 {result.block_id_2} 不存在，无法保存分析结果"
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                    
                    # 使用block_id_2，如果不存在则使用block_id_1
                    block_id_2 = result.block_id_2 if result.block_id_2 and block2 else result.block_id_1
                    
                    # 检查是否已存在相同的分析结果
                    existing_analysis = session.query(Analysis).filter_by(result_id=result.result_id).first()
                    
                    if existing_analysis:
                        # 更新现有分析结果
                        existing_analysis.score = str(result.score) if result.score is not None else '0'
                        existing_analysis.details = result.details or {}
                    else:
                        # 创建新的分析结果
                        analysis = Analysis(
                            result_id=result.result_id,
                            block_id_1=result.block_id_1,
                            block_id_2=block_id_2,
                            block_id=block1.id,  # 为了兼容旧代码
                            analysis_type=analysis_type.value,
                            score=str(result.score) if result.score is not None else '0',
                            details=result.details or {}
                        )
                        session.add(analysis)
                
                # 提交所有更改
                session.commit()
                logger.info(f"成功保存 {len(result_data)} 个分析结果")
            except Exception as e:
                session.rollback()
                logger.error(f"保存分析结果时出错: {e}")
                raise

    def get_analysis_results(self, analysis_type: AnalysisType, filter_criteria: Optional[Dict[str, Any]] = None) -> List[AnalysisResult]:
        """获取分析结果。"""
        self._ensure_initialized()
        with SessionLocal() as session:
            query = session.query(Analysis).filter_by(analysis_type=analysis_type.value)
            
            if filter_criteria:
                if "min_score" in filter_criteria:
                    # 对于JSON格式的score字段，需要特殊处理
                    try:
                        min_score = float(filter_criteria["min_score"])
                        # 这里使用简单的字符串比较，实际情况可能需要更复杂的JSON查询
                        query = query.filter(Analysis.score >= str(min_score))
                    except ValueError:
                        logger.warning(f"Invalid min_score: {filter_criteria['min_score']}")
                
                if "block_id" in filter_criteria:
                    block_id = filter_criteria["block_id"]
                    query = query.filter(
                        (Analysis.block_id_1 == block_id) | 
                        (Analysis.block_id_2 == block_id)
                    )
            
            results = query.all()
            return [
                AnalysisResult(
                    analysis_type=AnalysisType(result.analysis_type),
                    block_id_1=result.block_id_1,
                    block_id_2=result.block_id_2,
                    score=float(result.score) if result.score else None,
                    details=result.details or {}
                )
                for result in results
            ]

    def save_user_decision(self, decision_data: UserDecision) -> None:
        """
        保存用户决策。
        
        Args:
            decision_data (UserDecision): 用户决策数据
            
        Raises:
            ValueError: 如果引用的块不存在
            SQLAlchemyError: 数据库操作异常
            RuntimeError: 如果存储未初始化
        """
        self._ensure_initialized()
        
        with SessionLocal() as session:
            try:
                # 查找关联的分析结果
                # 生成与AnalysisResult相同的result_id
                sorted_ids = sorted([decision_data.block_id_1, decision_data.block_id_2])
                id_string = f"{sorted_ids[0]}_{sorted_ids[1]}_{decision_data.analysis_type.value}"
                namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
                result_id = str(uuid.uuid5(namespace, id_string))
                
                # 查找分析结果
                analysis_result = session.query(Analysis).filter_by(result_id=result_id).first()
                
                # 如果找不到分析结果，尝试创建一个
                if not analysis_result:
                    # 查找块
                    block1 = session.query(Block).filter_by(block_id=decision_data.block_id_1).first()
                    block2 = session.query(Block).filter_by(block_id=decision_data.block_id_2).first()
                    
                    if not block1 or not block2:
                        error_msg = f"无法找到决策引用的块: {decision_data.block_id_1}, {decision_data.block_id_2}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    # 创建一个新的分析结果
                    analysis_result = Analysis(
                        result_id=result_id,
                        block_id_1=decision_data.block_id_1,
                        block_id_2=decision_data.block_id_2,
                        block_id=block1.id,  # 为了兼容旧代码
                        analysis_type=decision_data.analysis_type.value,
                        score=0.0,  # 默认得分
                        details={}
                    )
                    session.add(analysis_result)
                    session.flush()  # 确保analysis_result有id
                
                # 创建决策
                existing_decision = session.query(Decision).filter_by(result_id=result_id).first()
                if existing_decision:
                    # 更新现有决策
                    existing_decision.decision_type = decision_data.decision.value
                    existing_decision.comment = decision_data.notes
                    existing_decision.timestamp = decision_data.timestamp
                else:
                    # 创建新决策
                    decision = Decision(
                        decision_id=decision_data.decision_id,
                        result_id=result_id,
                        decision_type=decision_data.decision.value,
                        timestamp=decision_data.timestamp,
                        comment=decision_data.notes
                    )
                    session.add(decision)
                
                session.commit()
                logger.info(f"成功保存决策 {decision_data.decision_id}，类型: {decision_data.decision.value}")
            except Exception as e:
                session.rollback()
                logger.error(f"保存用户决策时出错: {e}")
                raise

    def get_user_decisions(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[UserDecision]:
        """获取用户决策。"""
        self._ensure_initialized()
        with SessionLocal() as session:
            # 联合查询决策和分析结果
            query = session.query(Decision, Analysis).join(
                Analysis, Decision.result_id == Analysis.result_id
            )
            
            if filter_criteria:
                if "decision" in filter_criteria:
                    query = query.filter(Decision.decision_type == filter_criteria["decision"].value)
                if "block_id" in filter_criteria:
                    query = query.filter(
                        (Analysis.block_id_1 == filter_criteria["block_id"]) | 
                        (Analysis.block_id_2 == filter_criteria["block_id"])
                    )
                if "analysis_type" in filter_criteria:
                    query = query.filter(Analysis.analysis_type == filter_criteria["analysis_type"].value)
            
            results = query.all()
            user_decisions = []
            
            for decision, analysis in results:
                user_decision = UserDecision(
                    analysis_type=AnalysisType(analysis.analysis_type),
                    block_id_1=analysis.block_id_1,
                    block_id_2=analysis.block_id_2,
                    decision=DecisionType(decision.decision_type),
                    notes=decision.comment,
                    timestamp=decision.timestamp
                )
                user_decisions.append(user_decision)
                
            return user_decisions

    def get_undecided_pairs(self, analysis_type: AnalysisType) -> List[AnalysisResult]:
        """
        获取指定分析类型下未有决策的分析结果对。
        
        Args:
            analysis_type (AnalysisType): 分析类型
            
        Returns:
            List[AnalysisResult]: 未决策的分析结果列表
        """
        self._ensure_initialized()
        with SessionLocal() as session:
            # 1. 查询所有已有决策的result_id
            decided_result_ids = set()
            decisions = session.query(Decision).all()
            for decision in decisions:
                if decision.result_id:
                    decided_result_ids.add(decision.result_id)
            
            # 2. 查询指定分析类型的所有分析结果
            analysis_results = session.query(Analysis).filter_by(analysis_type=analysis_type.value).all()
            
            # 3. 过滤出未决策的分析结果
            undecided_results = []
            for analysis in analysis_results:
                if analysis.result_id not in decided_result_ids:
                    # 创建分析结果对象
                    result = AnalysisResult(
                        analysis_type=AnalysisType(analysis.analysis_type),
                        block_id_1=analysis.block_id_1,
                        block_id_2=analysis.block_id_2,
                        score=float(analysis.score) if analysis.score else None,
                        details=analysis.details or {}
                    )
                    undecided_results.append(result)
            
            return undecided_results

    def finalize(self) -> None:
        """清理资源。"""
        logger.info("Finalizing ORMStorage")
        self._initialized = False 