# =====================================================
# tests/logging/test_errors.py
# =====================================================
"""Confirm custom errors expose expected attributes."""
from __future__ import annotations

from kd_tool.logging.errors import LoggingConfigError, ErrorSeverity


def test_config_error_repr():
    err = LoggingConfigError("bad level foo")
    assert err.code == "CONFIG"
    assert err.severity is ErrorSeverity.FATAL
    assert "bad level foo" in str(err)