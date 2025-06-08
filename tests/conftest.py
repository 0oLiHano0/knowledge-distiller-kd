# 全局 Fixtures
from __future__ import annotations

import pytest
from kd_tool.logging.protocols import LoggerProtocol
from tests.logging.dummy_logger import MockLogger

@pytest.fixture
def dummy_logger() -> LoggerProtocol:
    """返回一个符合 LoggerProtocol 的 mock logger 实例"""
    return MockLogger()

def pytest_configure(config):
    config.option.asyncio_mode = "auto"
    config.option.asyncio_default_fixture_loop_scope = "function"