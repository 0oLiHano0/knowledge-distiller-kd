# =====================================================
# tests/logging/test_loguru_impl.py
# =====================================================
"""Tests that Loguru implementation obeys config DTO."""
from __future__ import annotations

from pathlib import Path

from kd_tool.logging.settings import LoggingConfigDTO
from kd_tool.logging.factory import LoggerFactory


def _read_log(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text().splitlines()


def test_log_file_written(tmp_path):
    log_file = tmp_path / "out.log"
    logger = LoggerFactory.create(
        LoggingConfigDTO(level="INFO", console=False, file_enabled=True, file_path=log_file),
        impl="loguru",
    )
    logger.info("line1")
    logger.debug("hidden")  # below INFO

    lines = _read_log(log_file)
    assert any("line1" in l for l in lines)
    assert all("hidden" not in l for l in lines)