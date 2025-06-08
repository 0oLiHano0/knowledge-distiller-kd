# =====================================================
# tests/logging/dummy_logger.py
# =====================================================
"""Behavioural tests for logging functionality."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

def test_logger_recording():
    mock_logger = MagicMock()
    mock_logger.info("hello", extra={"x": 1})
    mock_logger.error("oops")

    assert mock_logger.info.call_count == 1
    assert mock_logger.info.call_args[0][0] == "hello"
    assert mock_logger.info.call_args[1]["extra"] == {"x": 1}
    assert mock_logger.error.call_count == 1