"""
为什么: 验证 LoggerFactory 的依赖注入和 logger 创建逻辑，确保日志工厂可配置、可扩展。
做什么: 检查 LoggerFactory 是否能正确注入 SettingsDTO 并返回符合要求的 logger。
怎么做: 使用 fixture 模拟 SettingsDTO，断言 logger 类型和配置。
"""

import pytest
from kd_tool.logging import LoggerFactory, LoggerProtocol
from kd_tool.logging.settings import LoggingSettingsDTO
import io
import tempfile
from pathlib import Path

@pytest.fixture
def fake_settings():
    return LoggingSettingsDTO(
        level="DEBUG",
        log_serialize_json=False,
        log_file=None,
        rotation="00:00",
        retention="10 days"
    )

def test_logger_factory_returns_protocol(fake_settings):
    """
    为什么: 保证 LoggerFactory 返回的 logger 一定符合 LoggerProtocol。
    做什么: 检查 get_logger 返回值类型。
    怎么做: 断言 isinstance。
    """
    factory = LoggerFactory(settings=fake_settings)
    logger = factory.get_logger()
    assert isinstance(logger, LoggerProtocol)

def test_logger_factory_applies_settings(fake_settings, monkeypatch):
    """
    为什么: 验证 LoggerFactory 是否应用了 SettingsDTO 的配置。
    做什么: 检查 logger 的日志级别和格式。
    怎么做: 用临时文件作为 log_file，断言日志内容。
    """
    from loguru import logger as _loguru_logger

    # 只 patch remove，add 用 loguru 原生
    monkeypatch.setattr(_loguru_logger, "remove", lambda: None)

    # 用临时文件作为 log_file
    with tempfile.NamedTemporaryFile(mode="r+", delete=True) as tmpfile:
        fake_settings.log_file = Path(tmpfile.name)
        factory = LoggerFactory(settings=fake_settings)
        logger = factory.get_logger(task_id="test_task")
        logger.debug("测试日志内容")
        logger.info("another log")
        # flush loguru sink
        import time; time.sleep(0.05)
        tmpfile.seek(0)
        log_contents = tmpfile.read()
        assert "测试日志内容" in log_contents
        assert "test_task" in log_contents
        assert "DEBUG" in log_contents or "INFO" in log_contents 