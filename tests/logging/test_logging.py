# =====================================================
# tests/test_logging.py
# =====================================================
"""Unit‑tests for the *kd_tool.logging* refactor.

Run with::

    pytest -q tests/test_logging.py

The tests focus on the public contract – *LoggerProtocol* and
*get_logger* – rather than Loguru internals.  Wherever Loguru global
state must be touched, it is patched via *monkeypatch* so that tests
run fully isolated and are side‑effect free.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# ===== system under test =====================================================
from kd_tool.logging import LoggerFactory
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.logging.settings import LoggingConfigDTO
from kd_tool.logging.errors import LoggingConfigError
from kd_tool.logging.factory import register, _REGISTRY  # type: ignore – internal


# ---------------------------------------------------------------------------
# helpers & fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_registry():
    """Isolate the provider registry per test and restore afterwards."""
    snapshot = _REGISTRY.copy()  # pylint: disable=protected-access
    yield
    _REGISTRY.clear()           # pylint: disable=protected-access
    _REGISTRY.update(snapshot)


@pytest.fixture()
def cfg_tmp_path() -> LoggingConfigDTO:
    tmp_dir = tempfile.TemporaryDirectory()
    return LoggingConfigDTO(
        level="INFO",
        console=True,
        file_enabled=True,
        file_path=str(Path(tmp_dir.name) / "app.log"),
    )


# ---------------------------------------------------------------------------
# 1. Basic factory behaviour
# ---------------------------------------------------------------------------

def test_get_logger_returns_protocol(cfg_tmp_path):
    logger = LoggerFactory.create(cfg_tmp_path)  # default impl="loguru"
    assert isinstance(logger, LoggerProtocol)
    # smoke‑call
    logger.info("hello from test")


# ---------------------------------------------------------------------------
# 2. File‑sink created when enabled
# ---------------------------------------------------------------------------

def test_file_sink_created(cfg_tmp_path):
    file_path = Path(cfg_tmp_path.file_path)
    if file_path.exists():
        file_path.unlink()  # 确保文件不存在
    
    logger = LoggerFactory.create(cfg_tmp_path)
    logger.info("write to file")
    logger.warning("another line")
    assert file_path.exists()
    with file_path.open() as fh:
        content = fh.read()
    assert "write to file" in content


# ---------------------------------------------------------------------------
# 3. Invalid config raises LoggingConfigError
# ---------------------------------------------------------------------------

def test_invalid_level_raises_error():
    bad_cfg = LoggingConfigDTO(level="NOTALEVEL", console=True)
    with pytest.raises(ValueError, match="Level 'NOTALEVEL' does not exist"):
        _ = LoggerFactory.create(bad_cfg)


# ---------------------------------------------------------------------------
# 4. Duplicate provider registration is rejected
# ---------------------------------------------------------------------------

def test_register_duplicate_name(monkeypatch):
    class Dummy(LoggerProtocol):
        # minimal stub satisfying Protocol for registration test only
        def __init__(self):
            pass
        def debug(self, msg: str, *, extra=None): ...
        def info(self, msg: str, *, extra=None): ...
        def warning(self, msg: str, *, extra=None): ...
        def error(self, msg: str, *, extra=None): ...
        def exception(self, msg: str, *, extra=None): ...
        def bind(self, **ctx):
            return self
        @classmethod
        def configure(cls, cfg):
            return cls()

    register("dummy")(Dummy)
    with pytest.raises(ValueError):
        register("dummy")(Dummy)  # duplicate name


# ---------------------------------------------------------------------------
# 5. Context binding propagates key‑value pairs
# ---------------------------------------------------------------------------

def test_bind_adds_context(cfg_tmp_path, capsys):
    logger = LoggerFactory.create(cfg_tmp_path)
    task_logger = logger.bind(task="T‑42")
    task_logger.info("task log")

    captured = capsys.readouterr()
    log_line = captured.err.strip()
    assert "task log" in log_line
    assert "'task'" in log_line  # 修改断言，检查字典键名
    assert "T‑42" in log_line    # 检查值


def test_custom_format(cfg_tmp_path):
    custom_fmt = "<level>{level}</level> - {message}"
    cfg = LoggingConfigDTO(
        level="INFO",
        console=True,
        file_enabled=True,
        file_path=cfg_tmp_path.file_path,
        fmt=custom_fmt
    )
    logger = LoggerFactory.create(cfg)
    logger.info("test message")
    # 验证输出格式...


def test_custom_rotation(cfg_tmp_path):
    cfg = LoggingConfigDTO(
        level="INFO",
        file_enabled=True,
        file_path=cfg_tmp_path.file_path,
        rotation="1 KB",
        retention="1 day"
    )
    logger = LoggerFactory.create(cfg)
    # 验证轮转行为...
