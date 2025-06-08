# =====================================================
# tests/logging/dummy_logger.py
# =====================================================
"""Behavioural tests for DummyLogger provider."""
from __future__ import annotations

from kd_tool.logging.providers.dummy_impl import DummyLogger  # type: ignore


def test_dummy_recording(dummy_logger):
    dummy_logger.info("hello", extra={"x": 1})
    dummy_logger.error("oops")

    records = DummyLogger.pop_records()
    assert len(records) == 2
    assert records[0]["msg"] == "hello"
    assert records[0]["extra"] == {"x": 1}