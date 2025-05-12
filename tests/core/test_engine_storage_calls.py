"""
测试引擎对存储接口的调用。

验证 KnowledgeDistillerEngine 正确地使用 StorageInterface 的方法进行数据持久化操作，
而非直接使用文件 IO 或数据库访问。
"""

import pytest
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from unittest.mock import MagicMock, patch, call

from knowledge_distiller_kd.core.config import AppConfig
from knowledge_distiller_kd.core.engine import KnowledgeDistillerEngine
from knowledge_distiller_kd.storage.storage_interface import StorageInterface
from knowledge_distiller_kd.processing.document_processor import ContentBlock as OldContentBlock
from knowledge_distiller_kd.core.models import (
    ContentBlock as ContentBlockDTO,
    UserDecision as UserDecisionDTO,
    AnalysisResult as AnalysisResultDTO,
    FileRecord as FileRecordDTO,
    AnalysisType, DecisionType, BlockType
)


@pytest.fixture
def mock_logger() -> MagicMock:
    """创建logger的模拟对象"""
    logger = MagicMock()
    return logger


@pytest.fixture
def mock_config() -> MagicMock:
    """创建配置的模拟对象，模拟AppConfig的行为"""
    config = MagicMock(spec=AppConfig)
    
    # 配置engine属性
    engine_config = MagicMock()
    engine_config.similarity_threshold = 0.85
    engine_config.semantic_model = "test-model"
    engine_config.batch_size = 32
    engine_config.cache_dir = "cache"
    engine_config.cache_base_dir = ".kd_cache"
    config.engine = engine_config
    
    return config


@pytest.fixture
def mock_storage() -> MagicMock:
    """创建存储接口的模拟对象"""
    storage = MagicMock(spec=StorageInterface)
    return storage


@pytest.fixture
def sample_content_blocks() -> List[MagicMock]:
    """创建示例的OldContentBlock对象列表"""
    blocks = []
    for i in range(3):
        block = MagicMock(spec=OldContentBlock)
        block.block_id = f"block_{i}"
        block.analysis_text = f"Sample text for block {i}"
        block.metadata = {'original_path': f"/path/to/file_{i}.md"}
        
        # 为element创建mock以支持__class__属性访问
        element_mock = MagicMock()
        element_mock.__class__.__name__ = "NarrativeText"
        block.element = element_mock
        
        blocks.append(block)
    return blocks


@pytest.fixture
def test_engine(mock_storage: MagicMock, mock_config: MagicMock, mock_logger: MagicMock, tmp_path: Path) -> KnowledgeDistillerEngine:
    """创建测试用的引擎实例"""
    # 创建输入目录
    input_dir = tmp_path / "input_dir"
    input_dir.mkdir()
    
    with patch('knowledge_distiller_kd.core.engine.MD5Analyzer'), \
         patch('knowledge_distiller_kd.core.engine.SemanticAnalyzer'):
        
        # 初始化引擎
        engine = KnowledgeDistillerEngine(
            storage=mock_storage,
            config=mock_config,
            logger=mock_logger,
            input_dir=input_dir
        )
        
        return engine


def test_process_documents_uses_storage_calls(test_engine: KnowledgeDistillerEngine, 
                                           mock_storage: MagicMock, 
                                           sample_content_blocks: List[MagicMock],
                                           tmp_path: Path):
    """测试_process_documents方法使用存储接口保存处理后的内容块"""
    # 创建测试文件
    test_file = test_engine.input_dir / "test.md"
    test_file.touch()
    
    # 模拟process_directory返回
    with patch('knowledge_distiller_kd.core.engine.process_directory') as mock_process_dir:
        mock_process_dir.return_value = {str(test_file): sample_content_blocks}
        
        # 模拟storage.register_file返回
        mock_storage.register_file.return_value = "test_file_id"
        
        # 调用处理方法
        result = test_engine._process_documents()
        
        # 验证存储接口的调用
        assert mock_storage.register_file.call_count == 1
        mock_storage.register_file.assert_called_with(str(test_file.resolve()))
        
        # 验证save_blocks被调用，并且传递了正确的参数类型
        assert mock_storage.save_blocks.call_count == 1
        # 验证第一个参数是file_id
        assert mock_storage.save_blocks.call_args[1]['file_id'] == "test_file_id"
        # 验证第二个参数是ContentBlockDTO对象列表
        blocks_arg = mock_storage.save_blocks.call_args[1]['blocks']
        assert isinstance(blocks_arg, list)
        assert len(blocks_arg) == len(sample_content_blocks)
        assert all(isinstance(block, ContentBlockDTO) for block in blocks_arg)
        
        # 验证方法返回True表示成功
        assert result is True


def test_load_decisions_uses_storage_calls(test_engine: KnowledgeDistillerEngine, mock_storage: MagicMock):
    """测试load_decisions方法使用存储接口加载决策"""
    # 准备测试数据
    test_blocks = [
        ContentBlockDTO(
            block_id="block_1",
            file_id="file_1",
            block_type=BlockType.TEXT,
            text="Test content 1",
            metadata={'original_path': '/path/to/file.md', 'kd_processing_status': 'keep'}
        ),
        ContentBlockDTO(
            block_id="block_2",
            file_id="file_1",
            block_type=BlockType.HEADING,
            text="Test content 2",
            metadata={'original_path': '/path/to/file.md'}
        )
    ]
    
    # 模拟storage.get_blocks_for_analysis返回
    mock_storage.get_blocks_for_analysis.return_value = test_blocks
    
    # 调用加载决策方法
    result = test_engine.load_decisions()
    
    # 验证存储接口的调用
    mock_storage.get_blocks_for_analysis.assert_called_once()
    
    # 验证决策被正确加载到内存
    assert len(test_engine.block_decisions) == 2
    assert test_engine._decisions_loaded is True
    
    # 验证方法返回True表示成功
    assert result is True


def test_save_decisions_uses_storage_calls(test_engine: KnowledgeDistillerEngine, mock_storage: MagicMock):
    """测试save_decisions方法使用存储接口保存决策"""
    # 准备测试数据
    test_engine.block_decisions = {
        '/path/to/file.md::block_1::text': 'keep',
        '/path/to/file.md::block_2::heading': 'delete'
    }
    
    # 模拟get_block返回的数据
    block1 = ContentBlockDTO(
        block_id="block_1",
        file_id="file_1",
        block_type=BlockType.TEXT,
        text="Test content 1",
        metadata={}
    )
    block2 = ContentBlockDTO(
        block_id="block_2",
        file_id="file_1",
        block_type=BlockType.HEADING,
        text="Test content 2",
        metadata={}
    )
    
    mock_storage.get_block.side_effect = lambda block_id: {
        "block_1": block1,
        "block_2": block2
    }.get(block_id)
    
    # 调用保存决策方法
    result = test_engine.save_decisions()
    
    # 验证存储接口的调用
    assert mock_storage.get_block.call_count == 2
    mock_storage.get_block.assert_has_calls([call("block_1"), call("block_2")], any_order=True)
    
    # 验证save_blocks被调用，并且传递了正确的参数
    assert mock_storage.save_blocks.call_count == 1
    # 验证第一个参数是file_id
    assert mock_storage.save_blocks.call_args[1]['file_id'] == "file_1"
    # 验证第二个参数是ContentBlockDTO对象列表
    blocks_arg = mock_storage.save_blocks.call_args[1]['blocks']
    assert isinstance(blocks_arg, list)
    assert len(blocks_arg) == 2
    
    # 验证metadata已更新
    for block in blocks_arg:
        assert 'kd_processing_status' in block.metadata
    
    # 验证方法返回True表示成功
    assert result is True


def test_apply_decisions_uses_storage_calls(test_engine: KnowledgeDistillerEngine, mock_storage: MagicMock):
    """测试apply_decisions方法使用存储接口获取文件和块数据"""
    # 准备测试数据
    test_engine.block_decisions = {
        '/path/to/file.md::block_1::text': 'keep',
        '/path/to/file.md::block_2::heading': 'delete'
    }
    
    # 创建文件记录
    file_record = FileRecordDTO(
        file_id="file_1",
        original_path="/path/to/file.md"
    )
    
    # 创建块记录
    blocks = [
        ContentBlockDTO(
            block_id="block_1",
            file_id="file_1",
            block_type=BlockType.TEXT,
            text="Test content 1",
            metadata={'original_path': '/path/to/file.md'}
        ),
        ContentBlockDTO(
            block_id="block_2",
            file_id="file_1",
            block_type=BlockType.HEADING,
            text="Test content 2",
            metadata={'original_path': '/path/to/file.md'}
        )
    ]
    
    # 模拟存储接口返回
    mock_storage.list_files.return_value = [file_record]
    mock_storage.get_blocks_by_file.return_value = blocks
    
    # 调用应用决策方法
    result = test_engine.apply_decisions()
    
    # 验证存储接口的调用
    mock_storage.list_files.assert_called_once()
    mock_storage.get_blocks_by_file.assert_called_once_with("file_1")
    
    # 验证只有未标记为删除的块生成了输出
    assert len(result) == 1  # 一个输出文件
    
    # 验证生成的内容只包含保留的块
    for output_file, content in result.items():
        assert "Test content 1" in content
        assert "Test content 2" not in content


def test_run_analysis_uses_storage_calls(test_engine: KnowledgeDistillerEngine, 
                                      mock_storage: MagicMock,
                                      sample_content_blocks: List[MagicMock]):
    """测试run_analysis方法使用存储接口进行数据持久化"""
    # 跳过预过滤器步骤
    test_engine.skip_prefilter = True
    
    # 模拟方法返回
    with patch.object(test_engine, '_gather_input_files', return_value=[Path('/path/to/test.md')]), \
         patch.object(test_engine, '_process_documents', return_value=True), \
         patch.object(test_engine, '_merge_code_blocks_step', return_value=True), \
         patch.object(test_engine, 'load_decisions', return_value=True), \
         patch.object(test_engine, '_initialize_decisions', return_value=True), \
         patch.object(test_engine, 'md5_analyzer') as mock_md5, \
         patch.object(test_engine, 'semantic_analyzer') as mock_semantic, \
         patch.object(test_engine, '_collect_analysis_results', return_value={'documents': [], 'blocks': [], 'analysis_results': []}), \
         patch.object(test_engine, '_collect_decisions', return_value=[]), \
         patch.object(test_engine, 'save_results', return_value=True), \
         patch('knowledge_distiller_kd.core.engine.init_db'):
        
        # 模拟MD5分析器返回
        mock_md5.find_md5_duplicates.return_value = ([], {})
        
        # 模拟语义分析器返回
        mock_semantic.load_semantic_model.return_value = True
        test_engine.semantic_duplicates = []
        
        # 调用分析方法
        result = test_engine.run_analysis()
        
        # 验证方法链中的存储调用
        # 1. process_documents会调用register_file和save_blocks
        assert test_engine._process_documents.called
        # 2. load_decisions会调用get_blocks_for_analysis
        assert test_engine.load_decisions.called
        # 3. save_results会保存分析结果
        assert test_engine.save_results.called
        
        # 验证分析成功完成
        assert result is True
        assert test_engine._analysis_completed is True 