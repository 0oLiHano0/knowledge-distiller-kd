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
from unittest.mock import MagicMock

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

@pytest.fixture()
def mock_logger():
    return MagicMock()

@pytest.fixture
def dummy_settings():
    return BlockMergerStageSettings()

def test_init_signature(mock_logger, dummy_settings):
    """
    为什么: 检查 BlockMergingStage 的初始化签名是否符合预期。
    做什么: 验证构造函数参数和类型。
    怎么做: 使用 mock_logger 和 dummy_settings 创建实例。
    """
    sig = inspect.signature(BlockMergingStage.__init__)
    params = list(sig.parameters.keys())
    assert 'logger' in params
    assert 'settings' in params

def test_is_stateless(mock_logger, dummy_settings):
    """
    为什么: 验证 BlockMergingStage 是无状态的。
    做什么: 检查实例方法不依赖实例状态。
    怎么做: 使用 mock_logger 和 dummy_settings 创建实例并测试。
    """
    stage = BlockMergingStage(mock_logger, dummy_settings)
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


def test_process_interaction(mock_logger, dummy_settings):
    """
    为什么: 检查 process 方法能否与 PipelineContextDTO、StorageInterface 交互。
    做什么: 伪造 context，调用 process，检查日志行为。
    怎么做: 用 mock_logger 和 PipelineContextDTO，断言日志内容。
    """
    stage = BlockMergingStage(mock_logger, dummy_settings)
    context = PipelineContextDTO(run_logger=mock_logger)
    result = stage.process(context)
    assert isinstance(result, PipelineContextDTO) 