"""
测试 ORM 模型的注册和基本操作。
"""

import pytest
from datetime import datetime
from uuid import uuid4
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from knowledge_distiller_kd.storage.models_sqlalchemy import Base, Document, Block, Analysis, Decision

# 测试数据库 URL
TEST_DB_URL = "sqlite:///:memory:"

@pytest.fixture
def engine():
    """创建测试数据库引擎"""
    engine = create_engine(TEST_DB_URL)
    Base.metadata.create_all(engine)
    return engine

@pytest.fixture
def session(engine):
    """创建测试会话"""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_document_model_crud(session):
    """测试文档模型的 CRUD 操作"""
    # 创建文档
    file_id = str(uuid4())
    doc = Document(
        file_id=file_id,
        path="/test/path/doc.md",
        file_hash="abc123",
        type="markdown",
        size=1024,
        ctime=datetime.utcnow(),
        mtime=datetime.utcnow()
    )
    session.add(doc)
    session.commit()

    # 查询文档
    saved_doc = session.query(Document).filter_by(file_id=file_id).first()
    assert saved_doc is not None
    assert saved_doc.path == "/test/path/doc.md"
    assert saved_doc.file_hash == "abc123"

    # 更新文档
    saved_doc.status = "processed"
    session.commit()
    updated_doc = session.query(Document).filter_by(file_id=file_id).first()
    assert updated_doc.status == "processed"

    # 删除文档
    session.delete(updated_doc)
    session.commit()
    deleted_doc = session.query(Document).filter_by(file_id=file_id).first()
    assert deleted_doc is None

def test_block_model_crud(session):
    """测试内容块模型的 CRUD 操作"""
    # 创建文档
    file_id = str(uuid4())
    doc = Document(
        file_id=file_id,
        path="/test/path/doc.md",
        file_hash="abc123"
    )
    session.add(doc)
    session.commit()

    # 创建内容块
    block_id = str(uuid4())
    block = Block(
        block_id=block_id,
        file_id=file_id,
        content_hash=f"hash_{block_id}",
        text="Test content",
        block_type="text"
    )
    session.add(block)
    session.commit()

    # 查询内容块
    saved_block = session.query(Block).filter_by(block_id=block_id).first()
    assert saved_block is not None
    assert saved_block.text == "Test content"
    assert saved_block.block_type == "text"

    # 更新内容块
    saved_block.processing_status = "processed"
    session.commit()
    updated_block = session.query(Block).filter_by(block_id=block_id).first()
    assert updated_block.processing_status == "processed"

    # 删除内容块
    session.delete(updated_block)
    session.commit()
    deleted_block = session.query(Block).filter_by(block_id=block_id).first()
    assert deleted_block is None

def test_analysis_model_crud(session):
    """测试分析结果模型的 CRUD 操作"""
    # 创建文档和内容块
    file_id = str(uuid4())
    doc = Document(
        file_id=file_id,
        path="/test/path/doc.md",
        file_hash="abc123"
    )
    session.add(doc)
    session.commit()

    block_id = str(uuid4())
    block = Block(
        block_id=block_id,
        file_id=file_id,
        content_hash=f"hash_{block_id}",
        text="Test content",
        block_type="text"
    )
    session.add(block)
    session.commit()

    # 创建分析结果
    result_id = str(uuid4())
    analysis = Analysis(
        result_id=result_id,
        block_id_1=block.block_id,
        block_id_2=block.block_id,  # 自己与自己比较
        block_id=block.id,  # 兼容旧代码
        analysis_type="semantic_similarity",
        score=0.95,
        details={"method": "test"}
    )
    session.add(analysis)
    session.commit()

    # 查询分析结果
    retrieved = session.query(Analysis).filter_by(result_id=result_id).first()
    assert retrieved is not None
    assert retrieved.score == 0.95
    assert retrieved.block_id == block.id

    # 修改分析结果
    retrieved.score = 0.85
    session.commit()

    # 验证更新
    updated = session.query(Analysis).filter_by(result_id=result_id).first()
    assert updated.score == 0.85

    # 删除分析结果
    session.delete(updated)
    session.commit()
    assert session.query(Analysis).filter_by(result_id=result_id).first() is None

def test_decision_model_crud(session):
    """测试用户决策模型的 CRUD 操作"""
    # 创建文档和内容块
    file_id = str(uuid4())
    doc = Document(
        file_id=file_id,
        path="/test/path/doc.md",
        file_hash="abc123"
    )
    session.add(doc)
    session.commit()

    block_id = str(uuid4())
    block = Block(
        block_id=block_id,
        file_id=file_id,
        content_hash=f"hash_{block_id}",
        text="Test content",
        block_type="text"
    )
    session.add(block)
    session.commit()

    # 创建分析结果
    result_id = str(uuid4())
    analysis = Analysis(
        result_id=result_id,
        block_id_1=block.block_id,
        block_id_2=block.block_id,  # 自己与自己比较
        block_id=block.id,
        analysis_type="semantic_similarity",
        score=0.9,
        details={}
    )
    session.add(analysis)
    session.commit()

    # 创建用户决策
    decision_id = str(uuid4())
    decision = Decision(
        decision_id=decision_id,
        result_id=result_id,
        block_id=block.id,
        decision_type="keep",
        comment="This is a test decision"
    )
    session.add(decision)
    session.commit()

    # 查询决策
    retrieved = session.query(Decision).filter_by(decision_id=decision_id).first()
    assert retrieved is not None
    assert retrieved.decision_type == "keep"
    assert retrieved.result_id == result_id

    # 修改决策
    retrieved.decision_type = "merge"
    session.commit()

    # 验证更新
    updated = session.query(Decision).filter_by(decision_id=decision_id).first()
    assert updated.decision_type == "merge"

    # 删除决策
    session.delete(updated)
    session.commit()
    assert session.query(Decision).filter_by(decision_id=decision_id).first() is None

def test_relationships(session):
    """测试模型之间的关系"""
    # 创建文档
    file_id = str(uuid4())
    doc = Document(
        file_id=file_id,
        path="/test/path/doc.md",
        file_hash="hash123",
        type="markdown",
        size=1000,
        status="processed"
    )
    session.add(doc)
    session.flush()

    # 创建块
    block = Block(
        file_id=doc.id,
        block_id="block123",
        content_hash="hash456",
        text="Test content",
        block_type="text",
        processing_status="processed"
    )
    session.add(block)
    session.flush()

    # 创建分析结果
    result_id = str(uuid4())
    analysis = Analysis(
        result_id=result_id,
        block_id_1=block.block_id,
        block_id_2=block.block_id,
        block_id=block.id,
        analysis_type="md5_duplicate",
        score=1.0,
        details={"duplicate_of": 2}
    )
    session.add(analysis)
    session.flush()

    # 创建决策
    decision_id = str(uuid4())
    decision = Decision(
        decision_id=decision_id,
        result_id=result_id,
        block_id=block.id,
        decision_type="merge",
        comment="Test decision"
    )
    session.add(decision)
    session.flush()

    # 测试关系
    # 文档 -> 块
    assert len(doc.blocks) == 1
    assert doc.blocks[0].id == block.id

    # 块 -> 文档
    assert block.document.id == doc.id

    # 块 -> 分析结果
    assert len(block.analysis_results) == 1
    assert block.analysis_results[0].id == analysis.id

    # 分析结果 -> 块
    assert analysis.block.id == block.id

    # 分析结果 -> 决策
    assert len(analysis.decisions) == 1
    assert analysis.decisions[0].id == decision.id

    # 决策 -> 分析结果
    assert decision.analysis_result.id == analysis.id

    # 决策 -> 块
    assert decision.block.id == block.id

    # 块 -> 决策
    assert len(block.decisions) == 1
    assert block.decisions[0].id == decision.id

    session.commit()

def test_table_names(engine):
    """测试表名是否正确"""
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "files" in tables
    assert "blocks" in tables
    assert "analysis_results" in tables
    assert "user_decisions" in tables 