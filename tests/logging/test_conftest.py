# =====================================================
# tests/logging/test_conftest.py
# =====================================================
"""Shared fixtures for logging‑layer unit tests."""
from __future__ import annotations

import pytest

from kd_tool.logging.factory import LoggerFactory  # type: ignore
from kd_tool.logging.settings import LoggingConfigDTO


@pytest.fixture()
def dummy_logger(monkeypatch):
    """Return a DummyLogger instance registered as ``impl='dummy'``.

    The fixture isolates global state by clearing DummyLogger records
    before each use.
    """
    # Import inside to avoid hard dependency for projects that
    # don't enable dummy provider outside tests.
    from kd_tool.logging.providers.dummy_impl import DummyLogger  # type: ignore

    cfg = LoggingConfigDTO(level="DEBUG", console=False, file_enabled=False)
    logger = LoggerFactory.create(cfg, impl="dummy")  # type: ignore
    # Ensure a clean slate.
    DummyLogger.pop_records()
    yield logger
    # Teardown: clean records again (safety for xdist runs)
    DummyLogger.pop_records()


@pytest.fixture()
def loguru_logger(tmp_path, monkeypatch):
    """Return a LoguruLogger instance that writes to *tmp_path* only."""
    cfg = LoggingConfigDTO(
        level="DEBUG",
        console=False,
        file_enabled=True,
        file_path=tmp_path / "test.log",
    )
    logger = LoggerFactory.create(cfg, impl="loguru")  # type: ignore
    yield logger