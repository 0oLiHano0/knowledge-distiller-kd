"""
为什么: 验证日志上下文管理的正确性和线程安全性
做什么: 测试上下文绑定、异步安全性和配置API
怎么做: 使用 pytest 编写单元测试
"""

import pytest
import asyncio
from typing import Dict, Any
from pathlib import Path

from kd_tool.logging.factory import LoggerFactory, LoggerProtocol
from kd_tool.logging.settings import LoggingConfigDTO, LogLevel
from kd_tool.logging.errors import LoggingError, ErrorContext, ErrorType


@pytest.fixture
def logger():
    """创建一个测试用的 logger 实例"""
    return LoggerFactory.create_logger()


def test_context_initialization(logger):
    """验证上下文初始化"""
    assert logger.get_context() == {}


def test_context_binding(logger):
    """验证上下文绑定"""
    # 绑定单个键值对
    logger = logger.bind(task_id="test_task")
    assert logger.get_context() == {"task_id": "test_task"}

    # 绑定多个键值对
    logger = logger.bind(stage="test_stage", user="test_user")
    assert logger.get_context() == {
        "task_id": "test_task",
        "stage": "test_stage",
        "user": "test_user"
    }


def test_restricted_key_binding(logger):
    """验证受限键的绑定"""
    # 尝试绑定受限键
    with pytest.raises(LoggingError) as exc_info:
        logger.bind(_internal="test")
    assert exc_info.value.error_type == ErrorType.CONTEXT_BIND_ERROR


def test_context_isolation(logger):
    """验证上下文隔离"""
    # 创建两个独立的上下文
    logger1 = logger.bind(task_id="task1")
    logger2 = logger.bind(task_id="task2")

    assert logger1.get_context() == {"task_id": "task1"}
    assert logger2.get_context() == {"task_id": "task2"}


@pytest.mark.asyncio
async def test_async_context_isolation(logger):
    """验证异步上下文隔离"""
    async def task1():
        logger1 = logger.bind(task_id="task1")
        await asyncio.sleep(0.1)
        assert logger1.get_context() == {"task_id": "task1"}

    async def task2():
        logger2 = logger.bind(task_id="task2")
        await asyncio.sleep(0.1)
        assert logger2.get_context() == {"task_id": "task2"}

    await asyncio.gather(task1(), task2())


def test_context_clearing(logger):
    """验证上下文清除"""
    logger = logger.bind(task_id="test_task", stage="test_stage")
    logger.clear_context()
    assert logger.get_context() == {}


def test_context_rebinding(logger):
    """验证上下文重新绑定"""
    logger = logger.bind(task_id="old_task")
    logger = logger.bind(task_id="new_task")
    assert logger.get_context() == {"task_id": "new_task"}


def test_task_id_validation(logger):
    """验证任务ID格式"""
    # 有效的任务ID
    logger = logger.with_task("valid-task-123")
    assert logger.get_context()["task_id"] == "valid-task-123"

    # 无效的任务ID
    with pytest.raises(LoggingError) as exc_info:
        logger.with_task("invalid task id")
    assert exc_info.value.error_type == ErrorType.INVALID_TASK_ID


def test_logging_hooks(logger):
    """验证日志钩子"""
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


def test_configuration_api(logger):
    """验证配置API"""
    config = LoggingConfigDTO(
        level=LogLevel.DEBUG,
        fmt="{message}",
        serialize=True,
        log_file=Path("test.log"),
    )
    logger.configure(config)


@pytest.mark.asyncio
async def test_async_logging_methods(logger):
    """验证异步日志方法"""
    logger = logger.bind(task_id="test_task")
    
    await logger.async_debug("debug message")
    await logger.async_info("info message")
    await logger.async_warning("warning message")
    await logger.async_error("error message")
    await logger.async_critical("critical message")
    await logger.async_exception("exception message")
    await logger.async_success("success message")
    await logger.async_trace("trace message")


def test_error_handling(logger):
    """验证错误处理"""
    try:
        raise ValueError("test error")
    except Exception as e:
        logger.error_with_context("error occurred", e, {"context": "test"})
        logger.trace_error("error occurred", e, {"context": "test"})
