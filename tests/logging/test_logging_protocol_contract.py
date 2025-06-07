"""
为什么: 验证日志协议契约的实现正确性
做什么: 测试 LoggerProtocol 的所有必需方法和属性
怎么做: 使用 pytest 编写单元测试
"""

import pytest
from typing import Dict, Any
from pathlib import Path

from kd_tool.logging.factory import LoggerFactory, LoggerProtocol
from kd_tool.logging.settings import LoggingConfigDTO, LogLevel
from kd_tool.logging.errors import LoggingError, ErrorContext, ErrorType


@pytest.fixture
def logger():
    """创建一个测试用的 logger 实例"""
    return LoggerFactory.create_logger()


def test_protocol_implementation(logger):
    """验证 logger 实现了 LoggerProtocol"""
    assert isinstance(logger, LoggerProtocol)


def test_basic_logging_methods(logger):
    """验证基础日志方法"""
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
    logger.exception("exception message")
    logger.success("success message")
    logger.trace("trace message")


def test_context_management(logger):
    """验证上下文管理方法"""
    # 初始上下文应为空
    assert logger.get_context() == {}

    # 绑定上下文
    logger = logger.bind(task_id="test_task")
    assert logger.get_context() == {"task_id": "test_task"}

    # 清除上下文
    logger.clear_context()
    assert logger.get_context() == {}


def test_task_management(logger):
    """验证任务管理方法"""
    # 设置任务ID
    logger = logger.with_task("test_task")
    assert logger.get_context()["task_id"] == "test_task"

    # 验证任务ID格式
    with pytest.raises(LoggingError) as exc_info:
        logger.with_task("invalid task id")
    assert exc_info.value.error_type == ErrorType.INVALID_TASK_ID


def test_hook_management(logger):
    """验证钩子管理方法"""
    before_called = False
    after_called = False

    def before(level: str, data: Dict[str, Any]) -> None:
        nonlocal before_called
        before_called = True

    def after(level: str, data: Dict[str, Any]) -> None:
        nonlocal after_called
        after_called = True

    logger.set_hooks(before=before, after=after)
    logger.info("test message")

    assert before_called
    assert after_called


def test_configuration(logger):
    """验证配置方法"""
    config = LoggingConfigDTO(
        level=LogLevel.DEBUG,
        fmt="{message}",
        serialize=True,
        log_file=Path("test.log"),
    )
    logger.configure(config)


def test_error_handling(logger):
    """验证错误处理方法"""
    try:
        raise ValueError("test error")
    except Exception as e:
        logger.error_with_context("error occurred", e, {"context": "test"})
        logger.trace_error("error occurred", e, {"context": "test"})


def test_handler_management(logger):
    """验证处理器管理方法"""
    # 添加处理器
    handler_id = logger.add(
        sink=lambda msg: None,
        level="DEBUG",
        format="{message}",
    )

    # 移除处理器
    logger.remove(handler_id)


@pytest.mark.asyncio
async def test_async_methods(logger):
    """验证异步方法"""
    await logger.async_debug("debug message")
    await logger.async_info("info message")
    await logger.async_warning("warning message")
    await logger.async_error("error message")
    await logger.async_critical("critical message")
    await logger.async_exception("exception message")
    await logger.async_success("success message")
    await logger.async_trace("trace message")


def test_restricted_key_binding(logger):
    """验证受限键绑定"""
    with pytest.raises(LoggingError) as exc_info:
        logger.bind(_internal="test")
    assert exc_info.value.error_type == ErrorType.CONTEXT_BIND_ERROR


def test_context_isolation(logger):
    """验证上下文隔离"""
    logger1 = logger.bind(task_id="task1")
    logger2 = logger.bind(task_id="task2")

    assert logger1.get_context() == {"task_id": "task1"}
    assert logger2.get_context() == {"task_id": "task2"}
