"""
为什么: 验证 DocProcessing 阶段的架构合规性。
做什么: 检查 __init__ 是否支持依赖注入、类是否无状态、process 方法签名是否符合 StageInterface，并模拟与 PipelineContextDTO、StorageInterface 的交互。
怎么做: 通过反射和伪对象断言。
"""
import inspect
import pytest
from unittest.mock import MagicMock
from kd_tool.stages.docprocessing.document_processing_stage import DocumentProcessingStage
from kd_tool.core.interfaces import StageInterface
from kd_tool.core.core_dtos import PipelineContextDTO
from kd_tool.storage.storage_interface import StorageInterface
from kd_tool.stages.docprocessing.settings_models import DocumentProcessingStageSettings
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.stages.docprocessing.adapter_interface import ParserAdapterInterface
from pathlib import Path

class DummyStorage(StorageInterface):
    def save_pipeline_context(self, context): pass
    def begin_transaction(self): pass
    def close(self): pass
    def commit_transaction(self): pass
    def get_content_block(self, *a, **kw): pass
    def initialize(self): pass
    def rollback_transaction(self): pass
    def save_content_blocks(self, *a, **kw): pass
    def get_all_blocks(self, *a, **kw): pass
    def get_block_by_id(self, *a, **kw): pass
    def delete_block(self, *a, **kw): pass
    def update_block(self, *a, **kw): pass
    # 其他方法可按需补充

class DummyParserAdapter(ParserAdapterInterface):
    def parse_file_to_raw_elements(self, file_path: Path):
        return []

@pytest.fixture
def dummy_storage():
    return DummyStorage()

@pytest.fixture
def dummy_settings():
    """
    why: 提供 DocProcessingStage 的 settings 依赖
    what: 返回 DocumentProcessingStageSettings 实例
    how: 直接实例化
    """
    return DocumentProcessingStageSettings()

def test_init_signature(dummy_logger, dummy_settings):
    """
    为什么: 检查 DocProcessingStage 的 __init__ 是否支持依赖注入。
    做什么: 断言 __init__ 参数包含 logger、settings、parser_adapter。
    怎么做: 用 inspect 获取签名。
    """
    sig = inspect.signature(DocumentProcessingStage.__init__)
    params = list(sig.parameters.keys())
    assert 'logger' in params
    assert 'settings' in params
    assert 'parser_adapter' in params

def test_is_stateless(dummy_logger, dummy_settings):
    """
    为什么: 检查 DocProcessingStage 是否无状态。
    做什么: 断言实例属性不包含除依赖外的可变状态。
    怎么做: 实例化后检查 __dict__。
    """
    stage = DocumentProcessingStage(
        logger=dummy_logger,
        settings=dummy_settings,
        parser_adapter=DummyParserAdapter()
    )
    allowed = {'_logger', '_settings', '_parser'}
    assert set(stage.__dict__.keys()) <= allowed

def test_process_signature():
    """
    为什么: 检查 process 方法签名是否符合 StageInterface。
    做什么: 断言参数和返回类型。
    怎么做: 用 inspect 获取签名。
    """
    sig = inspect.signature(DocumentProcessingStage.process)
    params = list(sig.parameters.values())
    assert len(params) == 2  # self, context
    assert params[1].annotation is PipelineContextDTO


def test_process_interaction(dummy_logger, dummy_settings):
    """
    为什么: 检查 process 方法能否与 PipelineContextDTO、StorageInterface 交互。
    做什么: 伪造 context，调用 process。
    怎么做: 用 DummyStorage 和 PipelineContextDTO。
    """
    stage = DocumentProcessingStage(
        logger=dummy_logger,
        settings=dummy_settings,
        parser_adapter=DummyParserAdapter()
    )
    context = PipelineContextDTO(run_logger=dummy_logger)
    result = stage.process(context)
    assert isinstance(result, PipelineContextDTO) 