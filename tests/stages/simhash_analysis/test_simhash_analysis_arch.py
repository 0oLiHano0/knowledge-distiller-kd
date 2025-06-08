"""
为什么: 验证 SimHashAnalysis 阶段的架构合规性。
做什么: 检查 __init__ 签名（DI）、无状态性、process 方法签名（符合 StageInterface），以及与 PipelineContextDTO、StorageInterface 的交互。
怎么做: 通过反射和伪对象断言。
"""
import inspect
import pytest
from unittest.mock import MagicMock
from kd_tool.stages.simhash_analysis.simhash_analysis_stage import SimHashAnalysisStage
from kd_tool.core.interfaces import StageInterface
from kd_tool.core.core_dtos import PipelineContextDTO
from kd_tool.storage.storage_interface import StorageInterface
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.stages.simhash_analysis.settings_models import SimHashAnalysisStageSettings
from kd_tool.stages.simhash_analysis.adapter_interface import SimHashAdapterInterface
import numpy as np

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

@pytest.fixture
def dummy_storage():
    return DummyStorage()

@pytest.fixture
def dummy_logger() -> LoggerProtocol:
    """返回一个 mock logger 实例"""
    return MagicMock()

@pytest.fixture
def dummy_settings():
    return SimHashAnalysisStageSettings()

class DummyAdapter(SimHashAdapterInterface):
    def calculate_simhash(self, text): return 0
    def calculate_similarity(self, hash1, hash2): return 1.0
    def calculate_hamming_distance(self, hash1, hash2): return 0
    def find_similar_pairs(self, hashes, threshold): return []

@pytest.fixture
def dummy_adapter():
    return DummyAdapter()

def test_init_signature(dummy_logger, dummy_settings, dummy_adapter):
    """
    为什么: 检查 SimHashAnalysisStage 的 __init__ 是否支持依赖注入。
    做什么: 断言 __init__ 参数包含 logger、settings、adapter。
    怎么做: 用 inspect 获取签名。
    """
    sig = inspect.signature(SimHashAnalysisStage.__init__)
    params = list(sig.parameters.keys())
    assert 'logger' in params
    assert 'settings' in params
    assert 'adapter' in params

def test_is_stateless(dummy_logger, dummy_settings, dummy_adapter):
    """
    为什么: 检查 SimHashAnalysisStage 是否无状态。
    做什么: 断言实例属性不包含除依赖外的可变状态。
    怎么做: 实例化后检查 __dict__。
    """
    stage = SimHashAnalysisStage(dummy_logger, dummy_settings, dummy_adapter)
    allowed = {'_logger', '_settings', '_adapter'}
    assert set(stage.__dict__.keys()) <= allowed

def test_process_signature():
    """
    为什么: 检查 process 方法签名是否符合 StageInterface。
    做什么: 断言参数和返回类型。
    怎么做: 用 inspect 获取签名。
    """
    sig = inspect.signature(SimHashAnalysisStage.process)
    params = list(sig.parameters.values())
    assert len(params) == 2  # self, context
    assert params[1].annotation is PipelineContextDTO


def test_process_interaction(dummy_logger, dummy_settings, dummy_adapter):
    """
    为什么: 检查 process 方法能否与 PipelineContextDTO、StorageInterface 交互，并断言日志行为。
    做什么: 伪造 context，调用 process，检查 DummyLogger 记录。
    怎么做: 用 DummyAdapter 和 PipelineContextDTO，断言日志内容。
    """
    DummyLogger.pop_records()  # type: ignore  # 清空历史日志

    stage = SimHashAnalysisStage(dummy_logger, dummy_settings, dummy_adapter)
    context = PipelineContextDTO(run_logger=dummy_logger)
    result = stage.process(context)
    assert isinstance(result, PipelineContextDTO)

    records = DummyLogger.pop_records()  # type: ignore
    # 断言没有 error 日志
    assert all(r["level"] != "ERROR" for r in records)
    # 断言有日志消息包含 "processed" 或你期望的关键词
    assert any("processed" in r["msg"] for r in records) 