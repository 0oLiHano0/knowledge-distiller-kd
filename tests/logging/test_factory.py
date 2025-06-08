# =====================================================
# tests/logging/test_factory.py
# =====================================================
"""Tests for LoggerFactory registration and retrieval."""
from __future__ import annotations

import pytest

from kd_tool.logging.factory import LoggerFactory, register  # type: ignore
from kd_tool.logging.settings import LoggingConfigDTO
from kd_tool.logging.errors import LoggingConfigError


class _TempLogger:
    @classmethod
    def configure(cls, cfg):
        return cls()
    def debug(self, msg: str, *, extra=None): ...
    def info(self, msg: str, *, extra=None): ...
    def warning(self, msg: str, *, extra=None): ...
    def error(self, msg: str, *, extra=None): ...
    def exception(self, msg: str, *, extra=None): ...
    def success(self, msg: str, *, extra=None): ...
    def bind(self, **ctx):
        return self


def test_register_and_get():
    name = "_temp"
    register(name)(_TempLogger)  # decorator returns class

    logger = LoggerFactory.create(LoggingConfigDTO(level="INFO"), impl=name)  # type: ignore[arg-type]
    assert isinstance(logger, _TempLogger)


def test_get_missing_impl():
    with pytest.raises(LoggingConfigError):
        LoggerFactory.create(LoggingConfigDTO(level="INFO"), impl="nonexistent")  # type: ignore[arg-type]
