"""
为什么: 验证 LoggerFactory 的依赖注入和 logger 创建逻辑，确保日志工厂可配置、可扩展
做什么: 检查 LoggerFactory 是否能正确创建符合要求的 logger
怎么做: 使用 pytest 编写单元测试
"""

import pytest
from typing import Dict, Any, Optional
from pathlib import Path

from kd_tool.logging.factory import LoggerFactory, LoggerProtocol
from kd_tool.logging.settings import LoggingConfigDTO, LogLevel
from kd_tool.logging.errors import LoggingError, ErrorContext, ErrorType


@pytest.fixture
def fake_settings() -> LoggingConfigDTO:
    """提供一份调试级别的日志配置"""
    return LoggingConfigDTO(
        level=LogLevel.INFO,
        fmt="{message}",
        serialize=True,
        log_file=Path("app.log"),
    )


@pytest.fixture
def logger(fake_settings):
    """通过 LoggerFactory 构造 logger"""
    return LoggerFactory.create_logger()


def test_logger_creation(logger):
    """验证 logger 创建成功且实现了 LoggerProtocol"""
    assert logger is not None
    assert isinstance(logger, LoggerProtocol)


def test_logger_methods(logger):
    """验证 logger 实现了所有必要的方法"""
    # 基础日志方法
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
    logger.exception("exception message")
    logger.success("success message")
    logger.trace("trace message")

    # 上下文管理
    logger.bind(task_id="test_task")
    logger.with_task("test_task")
    logger.get_context()
    logger.clear_context()

    # 错误处理
    logger.error_with_context("error", Exception("test error"), {})
    logger.trace_error("error", Exception("test error"), {})


def test_logger_context(logger):
    """验证 logger 的上下文管理功能"""
    # 初始上下文应为空
    assert logger.get_context() == {}

    # 绑定上下文
    logger = logger.bind(task_id="test_task")
    assert logger.get_context() == {"task_id": "test_task"}

    # 清除上下文
    logger.clear_context()
    assert logger.get_context() == {}


def test_logger_with_task(logger):
    """验证 with_task 方法"""
    logger = logger.with_task("test_task")
    assert logger.get_context() == {"task_id": "test_task"}


def test_logger_error_handling(logger):
    """验证错误处理功能"""
    try:
        raise ValueError("test error")
    except Exception as e:
        logger.error_with_context("error occurred", e, {"context": "test"})
        logger.trace_error("error occurred", e, {"context": "test"})


def test_logger_configuration(logger):
    """验证配置功能"""
    config = LoggingConfigDTO(
        level=LogLevel.DEBUG,
        fmt="{message}",
        serialize=True,
        log_file=Path("test.log"),
    )
    logger.configure(config)


def test_logger_hooks(logger):
    """验证钩子函数功能"""
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


def test_logger_async_methods(logger):
    """验证异步日志方法"""
    import asyncio

    async def test_async():
        await logger.async_debug("async debug")
        await logger.async_info("async info")
        await logger.async_warning("async warning")
        await logger.async_error("async error")
        await logger.async_critical("async critical")
        await logger.async_exception("async exception")
        await logger.async_success("async success")
        await logger.async_trace("async trace")

    asyncio.run(test_async())


def test_logger_add_remove(logger):
    """验证添加和移除处理器功能"""
    # 添加处理器
    handler_id = logger.add(
        sink=lambda msg: None,
        level="DEBUG",
        format="{message}",
    )

    # 移除处理器
    logger.remove(handler_id)


def test_logger_dummy(logger):
    """验证空日志记录器功能"""
    dummy = LoggerFactory.create_dummy_logger()
    assert dummy is not None
    assert isinstance(dummy, LoggerProtocol)


def test_logger_service(logger):
    """验证日志服务创建功能"""
    service = LoggerFactory.create_service(logger)
    assert service is not None
    assert isinstance(service, LoggerProtocol)
