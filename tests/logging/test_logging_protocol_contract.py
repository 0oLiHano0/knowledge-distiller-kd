"""
为什么(why): 验证日志协议(LoggerProtocol)的契约符合性，确保所有日志实现都能被正确识别和替换。
做什么(what): 检查 LoggerFactory 返回的 logger 是否实现了 LoggerProtocol 所需方法。
怎么做(how): 通过 runtime_checkable + isinstance 断言。
"""

import pytest
from kd_tool.logging import LoggerProtocol, LoggerFactory
from kd_tool.logging.settings import LoggingSettingsDTO

@pytest.mark.parametrize("logger_instance", [
    lambda: LoggerFactory(
        LoggingSettingsDTO(
            level="DEBUG",
            log_serialize_json=False,
            log_file=None,
            rotation="00:00",
            retention="10 days"
        )
    ).get_logger(),
])
def test_logger_protocol_contract(logger_instance):
    """
    为什么: 保证所有 logger 实现都符合 LoggerProtocol。
    做什么: 检查 logger 是否实现了 LoggerProtocol 的所有方法。
    怎么做: 通过 isinstance 断言。
    """
    logger = logger_instance() if callable(logger_instance) else logger_instance
    assert isinstance(logger, LoggerProtocol) 