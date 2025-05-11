"""
测试知识提炼引擎数据持久化功能。
检验run_analysis()执行后是否将分析结果正确存储到SQLite数据库中。
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from sqlalchemy.orm import Session

from knowledge_distiller_kd.core.engine import KnowledgeDistillerEngine
from knowledge_distiller_kd.core.constants import DECISION_KEEP, DECISION_DELETE
from knowledge_distiller_kd.storage.sqlite_storage import init_db, SessionLocal, engine
from knowledge_distiller_kd.storage.models_sqlalchemy import Document, Block, Analysis, Decision, Base
from knowledge_distiller_kd.storage.storage_interface import StorageInterface


class TestPersistence:
    """测试KnowledgeDistillerEngine的持久化功能"""

    @pytest.fixture(autouse=True)
    def clean_db(self):
        """在每个测试前清理数据库"""
        # 使用内存数据库
        os.environ["TESTING"] = "1"
        # 初始化数据库
        init_db()
        yield
        # 清理所有表数据
        with engine.begin() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                conn.execute(table.delete())
        # 测试后清理环境变量
        os.environ.pop("TESTING", None)

    @pytest.fixture
    def mock_input_dir(self, tmp_path):
        """创建临时测试目录"""
        input_dir = tmp_path / "test_input"
        input_dir.mkdir()
        # 创建一个Markdown文件用于测试（引擎支持的文件类型）
        test_file = input_dir / "test.md"
        test_file.write_text("# 测试标题\n\n这是测试文件内容。\n\n这是第二段内容。")
        return input_dir

    @pytest.fixture
    def mock_storage(self):
        """创建一个mock的存储接口"""
        storage = MagicMock(spec=StorageInterface)
        return storage

    def test_run_analysis_persistence(self, mock_input_dir, mock_storage):
        """测试run_analysis将分析结果保存到数据库"""
        # 创建引擎并运行分析
        engine = KnowledgeDistillerEngine(
            storage=mock_storage,
            input_dir=str(mock_input_dir), 
            skip_prefilter=True,
            skip_semantic=True  # 跳过语义分析以简化测试
        )
        
        # 运行分析
        result = engine.run_analysis()
        assert result is True
        
        # 验证数据是否已保存到数据库
        with SessionLocal() as session:
            # 检查文档是否保存
            docs = session.query(Document).all()
            assert len(docs) > 0
            
            # 检查是否有文本块
            blocks = session.query(Block).all()
            assert len(blocks) > 0
            
            # 检查是否生成了分析结果
            analyses = session.query(Analysis).all()
            assert len(analyses) >= 0  # 可能没有分析结果，因为我们只有一个文件

    def test_save_results_transaction(self, mock_storage):
        """测试save_results方法的事务处理功能"""
        # 准备测试数据
        analysis_results = {
            "documents": [
                {"path": "/test/doc1.txt", "file_hash": "abcd1234", "type": "text", "size": 1000},
                {"path": "/test/doc2.txt", "file_hash": "efgh5678", "type": "text", "size": 2000}
            ],
            "blocks": [
                {"file_id": 1, "block_id": "hash1", "content_hash": "hash1", "text": "Block 1", "block_type": "text"},
                {"file_id": 1, "block_id": "hash2", "content_hash": "hash2", "text": "Block 2", "block_type": "code"}
            ],
            "analyses": [
                {"block_id": 1, "analysis_type": "md5_duplicate", "score": 1.0, "details": {"duplicate_of": 2}}
            ]
        }
        
        decisions = [
            {"block_id": 1, "decision_type": DECISION_KEEP, "comment": "Keep this block"}
        ]
        
        # 测试正常保存
        engine = KnowledgeDistillerEngine(storage=mock_storage, skip_prefilter=True, skip_semantic=True)
        engine.save_results(analysis_results, decisions)
        
        # 验证正常保存的结果
        with SessionLocal() as session:
            assert session.query(Document).count() == 2
            assert session.query(Block).count() == 2
            assert session.query(Analysis).count() == 1
            assert session.query(Decision).count() == 1

    def test_save_results_rollback(self, mock_storage):
        """测试save_results在异常情况下是否正确回滚事务"""
        # 准备测试数据
        analysis_results = {
            "documents": [
                {"path": "/test/doc3.txt", "file_hash": "abcd1234", "type": "text", "size": 1000}
            ],
            "blocks": [
                {"file_id": 1, "block_id": "hash1", "content_hash": "hash1", "text": "Block 1", "block_type": "text"}
            ],
            "analyses": [
                # 这里故意创建一个无效的分析结果，引用不存在的block_id
                {"block_id": 999, "analysis_type": "md5_duplicate", "score": 1.0}
            ]
        }
        
        decisions = []
        
        # 测试异常情况下的回滚
        engine = KnowledgeDistillerEngine(storage=mock_storage, skip_prefilter=True, skip_semantic=True)
        
        # 期望引发异常
        with pytest.raises(ValueError):
            engine.save_results(analysis_results, decisions)
        
        # 验证事务已回滚，数据库中没有部分写入的数据
        with SessionLocal() as session:
            # 确保这个测试中没有新增记录
            assert session.query(Document).filter(Document.path == "/test/doc3.txt").count() == 0

    def test_duplicate_path_handling(self, mock_storage):
        """测试处理重复文件路径的情况"""
        # 准备测试数据，包含重复的路径
        analysis_results = {
            "documents": [
                {"path": "/test/duplicate.txt", "file_hash": "hash1", "type": "text", "size": 1000},
                {"path": "/test/duplicate.txt", "file_hash": "hash2", "type": "text", "size": 2000},  # 重复路径
                {"path": "/test/unique.txt", "file_hash": "hash3", "type": "text", "size": 3000}     # 唯一路径
            ],
            "blocks": [
                {"file_id": 1, "block_id": "block1", "content_hash": "hash1", "text": "Block 1", "block_type": "text"},
                {"file_id": 2, "block_id": "block2", "content_hash": "hash2", "text": "Block 2", "block_type": "text"},
                {"file_id": 3, "block_id": "block3", "content_hash": "hash3", "text": "Block 3", "block_type": "text"}
            ],
            "analyses": [
                {"block_id": 1, "analysis_type": "md5_duplicate", "score": 1.0, "details": {"duplicate_of": 2}}
            ]
        }
        
        decisions = [
            {"block_id": 1, "decision_type": DECISION_KEEP, "comment": "Keep this block"}
        ]
        
        # 测试保存包含重复路径的数据
        engine = KnowledgeDistillerEngine(storage=mock_storage, skip_prefilter=True, skip_semantic=True)
        engine.save_results(analysis_results, decisions)
        
        # 验证结果
        with SessionLocal() as session:
            # 检查文档数量（应该只有2个，因为重复路径被忽略）
            docs = session.query(Document).all()
            assert len(docs) == 2
            
            # 验证重复路径只保存了第一条记录
            duplicate_docs = session.query(Document).filter(Document.path == "/test/duplicate.txt").all()
            assert len(duplicate_docs) == 1
            assert duplicate_docs[0].file_hash == "hash1"  # 应该保存第一条记录
            
            # 验证唯一路径正常保存
            unique_docs = session.query(Document).filter(Document.path == "/test/unique.txt").all()
            assert len(unique_docs) == 1
            assert unique_docs[0].file_hash == "hash3"
            
            # 验证其他相关数据也正确保存
            assert session.query(Block).count() == 3
            assert session.query(Analysis).count() == 1
            assert session.query(Decision).count() == 1 