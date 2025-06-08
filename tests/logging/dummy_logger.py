# =====================================================
# tests/logging/dummy_logger.py
# =====================================================
"""Mock logger implementation for testing."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.logging.settings import LoggingConfigDTO

class MockLogger(LoggerProtocol):
    """符合 LoggerProtocol 的 mock 实现"""
    
    def __init__(self):
        self._mock = MagicMock()
    
    def debug(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self._mock.debug(msg, extra=extra)
    
    def info(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self._mock.info(msg, extra=extra)
    
    def warning(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self._mock.warning(msg, extra=extra)
    
    def error(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self._mock.error(msg, extra=extra)
    
    def exception(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self._mock.exception(msg, extra=extra)
    
    def success(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self._mock.success(msg, extra=extra)
    
    def bind(self, **ctx: Any) -> "LoggerProtocol":
        return self

    @classmethod
    def configure(cls, cfg: LoggingConfigDTO) -> "LoggerProtocol":
        return cls()

# 保留原有的测试函数，但使用新的 MockLogger
def test_logger_recording():
    mock_logger = MockLogger()
    mock_logger.info("hello", extra={"x": 1})
    mock_logger.error("oops")

    assert mock_logger._mock.info.call_count == 1
    assert mock_logger._mock.info.call_args[0][0] == "hello"
    assert mock_logger._mock.info.call_args[1]["extra"] == {"x": 1}
    assert mock_logger._mock.error.call_count == 1