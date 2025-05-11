"""
存储接口测试模块。

测试 StorageInterface 的基本功能，确保所有存储实现都满足接口要求。
"""

import pytest
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from knowledge_distiller_kd.core.models import (
    AnalysisResult, AnalysisType, BlockType, ContentBlock, DecisionType,
    FileRecord, UserDecision
)
from knowledge_distiller_kd.storage.storage_interface import StorageInterface
from knowledge_distiller_kd.storage.orm_storage import ORMStorage
from knowledge_distiller_kd.storage.sqlite_storage import init_db, SessionLocal
from knowledge_distiller_kd.storage.models_sqlalchemy import Document, Block, Analysis, Decision

@pytest.fixture
def storage():
    """提供存储实例。"""
    # 初始化数据库
    init_db()
    
    # 创建存储实例
    storage = ORMStorage()
    storage.initialize()
    
    yield storage
    
    # 清理：删除测试数据
    with SessionLocal() as session:
        session.query(Decision).delete()
        session.query(Analysis).delete()
        session.query(Block).delete()
        session.query(Document).delete()
        session.commit()

def test_storage_initialization(storage):
    """测试存储初始化。"""
    assert isinstance(storage, StorageInterface)
    assert storage._initialized

def test_register_file(storage):
    """测试文件注册功能。"""
    filepath = "test.md"
    file_id = storage.register_file(filepath)
    assert file_id is not None
    
    # 验证文件记录
    record = storage.get_file_record(file_id)
    assert record is not None
    assert record.original_path == str(Path(filepath).resolve())

def test_save_and_get_blocks(storage):
    """测试内容块的保存和获取。"""
    # 注册文件
    file_id = storage.register_file("test.md")
    
    # 创建测试块
    blocks = [
        ContentBlock(
            block_id="block1",
            file_id=file_id,
            text="Test content 1",
            block_type=BlockType.TEXT,
            metadata={"key": "value"}
        ),
        ContentBlock(
            block_id="block2",
            file_id=file_id,
            text="Test content 2",
            block_type=BlockType.CODE,
            metadata={"key": "value2"}
        )
    ]
    
    # 保存块
    storage.save_blocks(file_id, blocks)
    
    # 获取块
    saved_blocks = storage.get_blocks_by_file(file_id)
    assert len(saved_blocks) == 2
    assert saved_blocks[0].block_id == "block1"
    assert saved_blocks[1].block_id == "block2"

def test_analysis_results(storage):
    """测试分析结果的保存和获取。"""
    # 注册文件和块
    file_id = storage.register_file("test.md")
    block1 = ContentBlock(
        block_id="block1",
        file_id=file_id,
        text="Test content 1",
        block_type=BlockType.TEXT
    )
    block2 = ContentBlock(
        block_id="block2",
        file_id=file_id,
        text="Test content 2",
        block_type=BlockType.TEXT
    )
    storage.save_blocks(file_id, [block1, block2])
    
    # 创建分析结果
    results = [
        AnalysisResult(
            analysis_type=AnalysisType.SEMANTIC_SIMILARITY,
            block_id_1="block1",
            block_id_2="block2",
            score=0.8
        )
    ]
    results[0].details = {"method": "test"}
    
    # 保存分析结果
    storage.save_analysis_result(AnalysisType.SEMANTIC_SIMILARITY, results)
    
    # 获取分析结果
    saved_results = storage.get_analysis_results(
        AnalysisType.SEMANTIC_SIMILARITY,
        {"min_score": 0.7}
    )
    assert len(saved_results) == 1
    assert saved_results[0].score == 0.8

def test_user_decisions(storage):
    """测试用户决策的保存和获取。"""
    # 先注册文件
    file_id = storage.register_file("test.md")
    
    # 创建并保存测试块
    block1 = ContentBlock(
        block_id="block1",
        file_id=file_id,
        text="Test content 1",
        block_type=BlockType.TEXT
    )
    block2 = ContentBlock(
        block_id="block2",
        file_id=file_id,
        text="Test content 2",
        block_type=BlockType.TEXT
    )
    storage.save_blocks(file_id, [block1, block2])
    
    # 创建分析结果
    results = [
        AnalysisResult(
            analysis_type=AnalysisType.SEMANTIC_SIMILARITY,
            block_id_1="block1",
            block_id_2="block2",
            score=0.8
        )
    ]
    storage.save_analysis_result(AnalysisType.SEMANTIC_SIMILARITY, results)
    
    # 创建决策
    decision = UserDecision(
        analysis_type=AnalysisType.SEMANTIC_SIMILARITY,
        block_id_1="block1",
        block_id_2="block2",
        decision=DecisionType.KEEP,
        notes="Test decision"
    )
    
    # 保存决策
    storage.save_user_decision(decision)
    
    # 获取决策
    decisions = storage.get_user_decisions()
    assert len(decisions) == 1
    assert decisions[0].decision == DecisionType.KEEP

def test_undecided_pairs(storage):
    """测试获取未决策的分析结果对。"""
    # 注册文件和块
    file_id = storage.register_file("test.md")
    block1 = ContentBlock(
        block_id="block1",
        file_id=file_id,
        text="Test content 1",
        block_type=BlockType.TEXT
    )
    block2 = ContentBlock(
        block_id="block2",
        file_id=file_id,
        text="Test content 2",
        block_type=BlockType.TEXT
    )
    storage.save_blocks(file_id, [block1, block2])
    
    # 创建分析结果
    results = [
        AnalysisResult(
            analysis_type=AnalysisType.SEMANTIC_SIMILARITY,
            block_id_1="block1",
            block_id_2="block2",
            score=0.8
        )
    ]
    results[0].details = {"method": "test"}
    storage.save_analysis_result(AnalysisType.SEMANTIC_SIMILARITY, results)
    
    # 获取未决策对
    undecided = storage.get_undecided_pairs(AnalysisType.SEMANTIC_SIMILARITY)
    assert len(undecided) == 1
    
    # 添加决策
    decision = UserDecision(
        analysis_type=AnalysisType.SEMANTIC_SIMILARITY,
        block_id_1="block1",
        block_id_2="block2",
        decision=DecisionType.KEEP
    )
    storage.save_user_decision(decision)
    
    # 再次获取未决策对
    undecided = storage.get_undecided_pairs(AnalysisType.SEMANTIC_SIMILARITY)
    assert len(undecided) == 0 