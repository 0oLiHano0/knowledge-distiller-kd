import os
import pytest
from sqlalchemy.orm import Session

# 导入将要创建的模型
# 现在导入这些模型会导致导入错误，因为文件尚未创建
from knowledge_distiller_kd.storage.models_sqlalchemy import Base, Document, Block, Analysis, Decision
from knowledge_distiller_kd.storage.sqlite_storage import engine, SessionLocal, init_db


@pytest.fixture(scope="function")
def setup_test_db():
    """
    设置测试数据库，每个测试函数执行前都会重置数据库
    """
    # 设置测试环境变量
    os.environ["TESTING"] = "1"
    
    # 创建所有表
    Base.metadata.create_all(bind=engine)
    
    # 提供会话给测试
    db = SessionLocal()
    try:
        yield db
    finally:
        # 清理：删除所有表
        Base.metadata.drop_all(bind=engine)
        db.close()


def test_document_model_crud(setup_test_db):
    """
    测试Document模型的CRUD操作
    """
    session = setup_test_db
    
    # 创建测试Document对象
    doc = Document(
        path="/test/path/document.md",
        file_hash="abc123",
        type="markdown",
        size=1024,
        status="processed"
    )
    
    # 添加到数据库
    session.add(doc)
    session.commit()
    
    # 查询并验证
    queried_doc = session.query(Document).filter(Document.path == "/test/path/document.md").first()
    assert queried_doc is not None
    assert queried_doc.file_hash == "abc123"
    assert queried_doc.type == "markdown"
    assert queried_doc.size == 1024
    assert queried_doc.status == "processed"


def test_block_model_crud(setup_test_db):
    """
    测试Block模型的CRUD操作
    """
    session = setup_test_db
    
    # 首先创建一个Document
    doc = Document(
        path="/test/path/document.md",
        file_hash="abc123"
    )
    session.add(doc)
    session.commit()
    
    # 创建关联的Block
    block = Block(
        file_id=doc.id,
        block_id="def456",
        content_hash="def456",
        text="这是测试文本块",
        block_type="paragraph",
        processing_status="processed",
        meta_data={"source": "test", "position": 1}
    )
    
    # 添加到数据库
    session.add(block)
    session.commit()
    
    # 查询并验证
    queried_block = session.query(Block).filter(Block.content_hash == "def456").first()
    assert queried_block is not None
    assert queried_block.text == "这是测试文本块"
    assert queried_block.block_type == "paragraph"
    assert queried_block.file_id == doc.id
    assert queried_block.meta_data["source"] == "test"


def test_analysis_model_crud(setup_test_db):
    """
    测试Analysis模型的CRUD操作
    """
    session = setup_test_db
    
    # 创建Document和Block
    doc = Document(path="/test/doc.md", file_hash="abc")
    session.add(doc)
    session.commit()
    
    block = Block(file_id=doc.id, block_id="def", content_hash="def", text="测试文本")
    session.add(block)
    session.commit()
    
    # 创建Analysis
    analysis = Analysis(
        block_id=block.id,
        analysis_type="semantic_similarity",
        score={"similarity": 0.95},
        details={"method": "cosine", "vector": [0.1, 0.2, 0.3]}
    )
    
    # 添加到数据库
    session.add(analysis)
    session.commit()
    
    # 查询并验证
    queried_analysis = session.query(Analysis).filter(
        Analysis.block_id == block.id, 
        Analysis.analysis_type == "semantic_similarity"
    ).first()
    
    assert queried_analysis is not None
    assert queried_analysis.score == {"similarity": 0.95}
    assert queried_analysis.details["method"] == "cosine"


def test_decision_model_crud(setup_test_db):
    """
    测试Decision模型的CRUD操作
    """
    session = setup_test_db
    
    # 创建Document和两个Block
    doc = Document(path="/test/doc.md", file_hash="abc")
    session.add(doc)
    session.commit()
    
    block1 = Block(file_id=doc.id, block_id="def1", content_hash="def1", text="文本1")
    block2 = Block(file_id=doc.id, block_id="def2", content_hash="def2", text="文本2")
    session.add_all([block1, block2])
    session.commit()
    
    # 创建Decision
    decision = Decision(
        block_id=block1.id,
        decision_type="duplicate",
        duplicate_of_block_id=block2.id,
        comment="这是一个重复块"
    )
    
    # 添加到数据库
    session.add(decision)
    session.commit()
    
    # 查询并验证
    queried_decision = session.query(Decision).filter(
        Decision.block_id == block1.id
    ).first()
    
    assert queried_decision is not None
    assert queried_decision.decision_type == "duplicate"
    assert queried_decision.duplicate_of_block_id == block2.id
    assert queried_decision.comment == "这是一个重复块"


def test_relationships(setup_test_db):
    """
    测试模型间的关系
    """
    session = setup_test_db
    
    # 创建Document
    doc = Document(path="/test/doc.md", file_hash="abc")
    session.add(doc)
    session.commit()
    
    # 创建Block
    block = Block(file_id=doc.id, block_id="def", content_hash="def", text="测试文本")
    session.add(block)
    session.commit()
    
    # 创建Analysis
    analysis = Analysis(
        block_id=block.id,
        analysis_type="semantic_similarity",
        score={"similarity": 0.95}
    )
    session.add(analysis)
    session.commit()
    
    # 验证关系
    # 通过Document获取其Blocks
    assert len(doc.blocks) == 1
    assert doc.blocks[0].content_hash == "def"
    
    # 通过Block获取其Document和Analyses
    assert block.document.path == "/test/doc.md"
    assert len(block.analyses) == 1
    assert block.analyses[0].analysis_type == "semantic_similarity"
    
    # 通过Analysis获取其Block
    assert analysis.block.text == "测试文本" 