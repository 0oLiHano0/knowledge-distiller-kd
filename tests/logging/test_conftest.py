# =====================================================
# tests/logging/test_conftest.py
# =====================================================
"""Shared fixtures for logging‑layer unit tests."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from kd_tool.logging.factory import LoggerFactory
from kd_tool.logging.settings import LoggingConfigDTO

@pytest.fixture()
def mock_logger():
    """Return a mock logger instance."""
    return MagicMock()

@pytest.fixture()
def loguru_logger(tmp_path, monkeypatch):
    """Return a LoguruLogger instance that writes to *tmp_path* only."""
    cfg = LoggingConfigDTO(
        level="DEBUG",
        console=False,
        file_enabled=True,
        file_path=tmp_path / "test.log",
    )
    logger = LoggerFactory.create(cfg, impl="loguru")
    yield logger