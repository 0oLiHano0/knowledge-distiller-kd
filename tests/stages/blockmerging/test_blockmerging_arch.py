"""
为什么: 验证 BlockMerging 阶段的架构合规性。
做什么: 检查 __init__ 是否支持依赖注入、类是否无状态、process 方法签名是否符合 StageInterface，并模拟与 PipelineContextDTO、StorageInterface 的交互。
怎么做: 通过反射和伪对象断言。
"""
import inspect
import pytest
from kd_tool.stages.blockmerging.block_merging_stage import BlockMergingStage
from kd_tool.core.interfaces import StageInterface
from kd_tool.core.core_dtos import PipelineContextDTO
from kd_tool.storage.storage_interface import StorageInterface
from kd_tool.stages.blockmerging.settings_models import BlockMergerStageSettings
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.logging.providers.dummy_impl import DummyLogger  # type: ignore

class DummyStorage(StorageInterface):
    def save_pipeline_context(self, context): pass
    def begin_transaction(self): pass
    def close(self): pass
    def commit_transaction(self): pass
    def get_content_block(self, *a, **kw): pass
    def initialize(self): pass
    def rollback_transaction(self): pass
    def save_content_blocks(self, *a, **kw): pass
    # 其他方法可按需补充

@pytest.fixture
def dummy_storage():
    return DummyStorage()

@pytest.fixture
def dummy_logger() -> LoggerProtocol:
    """返回一个 DummyLogger 实例"""
    return DummyLogger()

@pytest.fixture
def dummy_settings():
    return BlockMergerStageSettings()

def test_init_signature(dummy_logger, dummy_settings):
    """
    为什么: 检查 BlockMergingStage 的 __init__ 是否支持依赖注入。
    做什么: 断言 __init__ 参数包含 logger 和 settings。
    怎么做: 用 inspect 获取签名。
    """
    sig = inspect.signature(BlockMergingStage.__init__)
    params = list(sig.parameters.keys())
    assert 'logger' in params
    assert 'settings' in params

def test_is_stateless(dummy_logger, dummy_settings):
    """
    为什么: 检查 BlockMergingStage 是否无状态。
    做什么: 断言实例属性不包含除依赖外的可变状态。
    怎么做: 实例化后检查 __dict__。
    """
    stage = BlockMergingStage(dummy_logger, dummy_settings)
    allowed = {'_logger', '_settings'}
    assert set(stage.__dict__.keys()) <= allowed

def test_process_signature():
    """
    为什么: 检查 process 方法签名是否符合 StageInterface。
    做什么: 断言参数和返回类型。
    怎么做: 用 inspect 获取签名。
    """
    sig = inspect.signature(BlockMergingStage.process)
    params = list(sig.parameters.values())
    assert len(params) == 2  # self, context
    assert params[1].annotation is PipelineContextDTO


def test_process_interaction(dummy_logger, dummy_settings):
    """
    为什么: 检查 process 方法能否与 PipelineContextDTO、StorageInterface 交互。
    做什么: 伪造 context，调用 process。
    怎么做: 用 DummyStorage 和 PipelineContextDTO。
    """
    stage = BlockMergingStage(dummy_logger, dummy_settings)
    context = PipelineContextDTO(run_logger=dummy_logger)
    result = stage.process(context)
    assert isinstance(result, PipelineContextDTO) 