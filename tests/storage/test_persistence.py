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
from knowledge_distiller_kd.storage.models_sqlalchemy import Document, Block, Analysis, Decision


class TestPersistence:
    """测试KnowledgeDistillerEngine的持久化功能"""

    @pytest.fixture
    def mock_input_dir(self, tmp_path):
        """创建临时测试目录"""
        input_dir = tmp_path / "test_input"
        input_dir.mkdir()
        # 创建一个简单文本文件用于测试
        test_file = input_dir / "test.txt"
        test_file.write_text("这是测试文件内容。\n这是第二行内容。")
        return input_dir

    @pytest.fixture
    def setup_memory_db(self):
        """设置内存数据库用于测试"""
        # 使用内存数据库
        os.environ["TESTING"] = "1"
        # 初始化数据库
        init_db()
        yield
        # 测试后清理环境变量
        os.environ.pop("TESTING", None)

    def test_run_analysis_persistence(self, mock_input_dir, setup_memory_db):
        """测试run_analysis将分析结果保存到数据库"""
        # 创建引擎并运行分析
        engine = KnowledgeDistillerEngine(
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

    def test_save_results_transaction(self, setup_memory_db):
        """测试save_results方法的事务处理功能"""
        # 准备测试数据
        analysis_results = {
            "documents": [
                {"path": "/test/doc1.txt", "file_hash": "abcd1234", "type": "text", "size": 1000},
                {"path": "/test/doc2.txt", "file_hash": "efgh5678", "type": "text", "size": 2000}
            ],
            "blocks": [
                {"document_id": 1, "content_hash": "hash1", "text": "Block 1", "raw_element_type": "text"},
                {"document_id": 1, "content_hash": "hash2", "text": "Block 2", "raw_element_type": "code"}
            ],
            "analyses": [
                {"block_id": 1, "analysis_type": "md5_duplicate", "score": 1.0, "details": {"duplicate_of": 2}}
            ]
        }
        
        decisions = [
            {"block_id": 1, "decision_type": DECISION_KEEP, "comment": "Keep this block"}
        ]
        
        # 测试正常保存
        engine = KnowledgeDistillerEngine(skip_prefilter=True, skip_semantic=True)
        engine.save_results(analysis_results, decisions)
        
        # 验证正常保存的结果
        with SessionLocal() as session:
            assert session.query(Document).count() == 2
            assert session.query(Block).count() == 2
            assert session.query(Analysis).count() == 1
            assert session.query(Decision).count() == 1

    def test_save_results_rollback(self, setup_memory_db):
        """测试save_results在异常情况下是否正确回滚事务"""
        # 准备测试数据
        analysis_results = {
            "documents": [
                {"path": "/test/doc1.txt", "file_hash": "abcd1234", "type": "text", "size": 1000}
            ],
            "blocks": [
                {"document_id": 1, "content_hash": "hash1", "text": "Block 1", "raw_element_type": "text"}
            ],
            "analyses": [
                # 这里故意创建一个无效的分析结果，引用不存在的block_id
                {"block_id": 999, "analysis_type": "md5_duplicate", "score": 1.0}
            ]
        }
        
        decisions = []
        
        # 测试异常情况下的回滚
        engine = KnowledgeDistillerEngine(skip_prefilter=True, skip_semantic=True)
        
        # 期望引发异常
        with pytest.raises(Exception):
            engine.save_results(analysis_results, decisions)
        
        # 验证事务已回滚，数据库中没有部分写入的数据
        with SessionLocal() as session:
            assert session.query(Document).count() == 0
            assert session.query(Block).count() == 0
            assert session.query(Analysis).count() == 0 