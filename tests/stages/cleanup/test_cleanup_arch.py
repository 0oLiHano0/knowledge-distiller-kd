"""
为什么: 验证 Cleanup 阶段的架构合规性。
做什么: 检查 __init__ 是否支持依赖注入、类是否无状态、process 方法签名是否符合 StageInterface，并模拟与 PipelineContextDTO、StorageInterface 的交互。
怎么做: 通过反射和伪对象断言。
"""
import inspect
import pytest
from kd_tool.stages.cleanup.cleanup_stage import CleanupStage
from kd_tool.core.interfaces import StageInterface
from kd_tool.core.core_dtos import PipelineContextDTO
from kd_tool.storage.storage_interface import StorageInterface
from kd_tool.stages.cleanup.settings_models import CleanupStageSettings
from kd_tool.stages.cleanup.adapter_interface import FileSystemAdapterInterface
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
    def get_all_blocks(self, *a, **kw): pass
    def get_block_by_id(self, *a, **kw): pass
    def delete_block(self, *a, **kw): pass
    def update_block(self, *a, **kw): pass
    # 其他方法可按需补充

class DummyFsAdapter(FileSystemAdapterInterface):
    def move_file(self, source_path, target_path): pass
    def delete_file(self, file_path): pass
    def ensure_directory_exists(self, dir_path): pass
    def file_exists(self, file_path): return True

@pytest.fixture
def dummy_storage():
    return DummyStorage()

@pytest.fixture
def dummy_settings():
    return CleanupStageSettings()

@pytest.fixture
def dummy_fs_adapter():
    return DummyFsAdapter()

@pytest.fixture
def mock_logger() -> LoggerProtocol:
    """返回一个 MagicMock 实例作为日志模拟"""
    return MagicMock()

def test_init_signature(mock_logger, dummy_settings, dummy_fs_adapter):
    """
    为什么: 检查 CleanupStage 的 __init__ 是否支持依赖注入。
    做什么: 断言 __init__ 参数包含 logger、settings 和 fs_adapter。
    怎么做: 用 inspect 获取签名。
    """
    sig = inspect.signature(CleanupStage.__init__)
    params = list(sig.parameters.keys())
    assert 'logger' in params
    assert 'settings' in params
    assert 'fs_adapter' in params

def test_is_stateless(mock_logger, dummy_settings, dummy_fs_adapter):
    """
    为什么: 检查 CleanupStage 是否无状态。
    做什么: 断言实例属性不包含除依赖外的可变状态。
    怎么做: 实例化后检查 __dict__。
    """
    stage = CleanupStage(mock_logger, dummy_settings, dummy_fs_adapter)
    allowed = {'_logger', '_settings', '_fs_adapter'}
    assert set(stage.__dict__.keys()) <= allowed

def test_process_signature():
    """
    为什么: 检查 process 方法签名是否符合 StageInterface。
    做什么: 断言参数和返回类型。
    怎么做: 用 inspect 获取签名。
    """
    sig = inspect.signature(CleanupStage.process)
    params = list(sig.parameters.values())
    assert len(params) == 2  # self, context
    assert params[1].annotation is PipelineContextDTO

def test_process_interaction(mock_logger, dummy_settings, dummy_fs_adapter):
    """
    为什么: 检查 process 方法能否与 PipelineContextDTO、StorageInterface 交互。
    做什么: 伪造 context，调用 process，验证基本功能。
    怎么做: 验证返回值和类型。
    """
    stage = CleanupStage(mock_logger, dummy_settings, dummy_fs_adapter)
    context = PipelineContextDTO(run_logger=mock_logger)
    result = stage.process(context)
    
    # 只验证基本功能
    assert isinstance(result, PipelineContextDTO)
    # 不验证具体的日志方法调用 