"""
====================开发指引======================
kd_tool/logging/service.py - v4.4
==================================

**【文件定位】**

* 包结构：kd_tool.logging
* 所属模块：日志服务层 - 核心服务类
* 职责：日志服务实现

**【模块职责（SRP）】**

* 提供 LoggerProtocol 的 Loguru 实现，保持无状态，支持上下文管理

**【依赖关系与注入】**

* 外部依赖：

  * LoggerProtocol：日志接口
  * LoggingError：错误处理
  * contextvars：上下文管理
* 依赖注入方式：构造函数注入
* Mock需求：支持 LoggerProtocol 的 Mock 实现

**【输入输出规范】**

* 输入：

  * **init**: logger: Optional[Any] = None
  * bind: **kwargs: Any
  * with_task: task_id: str
* 输出：

  * bind: LoguruLogger
  * with_task: LoguruLogger
* 异常：

  * LoggingError：日志操作失败
  * ContextBindError：上下文绑定失败

**【核心架构约束】**

1. 无状态性：

   * 禁止在实例中保存状态
   * 所有状态通过 contextvars 管理
   * 每次操作返回新实例

2. 上下文管理：

   * 禁止链式绑定
   * 一次性绑定所有上下文
   * 使用 contextvars 确保隔离

3. 错误处理：

   * 使用 LoggingError 及其子类
   * 提供详细的错误上下文
   * 支持错误追踪

4. 类型安全：

   * 完整的类型提示
   * 运行时类型检查
   * 参数验证

**【接口与DTO规范】**

1. 日志方法：

   * debug/info/warning/error/exception/success/trace
   * 异步版本：async_*
   * 错误处理：error_with_context/trace_error

2. 上下文管理：

   * bind：一次性绑定
   * with_task：任务ID绑定
   * get_context/clear_context

3. 测试支持：

   * get_test_context
   * reset_test_context

**【日志与安全】**

1. 日志记录：

   * 关键操作记录
   * 错误详情记录
   * 上下文信息记录

2. 安全处理：

   * 敏感信息脱敏
   * 上下文隔离
   * 错误信息保护

**【任务清单】**

1. [待完成] 优化上下文绑定机制

   * 禁止链式绑定
   * 一次性绑定所有上下文
   * 增强上下文验证

2. [待完成] 增强错误处理

   * 完善错误上下文
   * 增强错误追踪
   * 优化错误消息

3. [待完成] 改进配置管理

   * 增强配置验证
   * 运行时配置检查
   * 配置更新机制

4. [待完成] 完善测试支持

   * 增加单元测试
   * 添加集成测试
   * 性能测试

**【其他说明】**

* 需要确保与 Loguru 的兼容性
* 支持异步日志记录
* 保持向后兼容性

"""

from __future__ import annotations
from typing import Optional, Any, Dict, Callable, Union, cast, TypeAlias, Self, ClassVar, Awaitable
import asyncio
import traceback
from copy import copy
from contextvars import ContextVar
from datetime import datetime, timezone
import threading
import inspect
import anyio

from loguru import logger as _loguru_logger
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.logging.errors import LoggingError, ErrorContext, ErrorType, ContextBindError, ValidationError
from kd_tool.logging.settings import LoggingConfigDTO, FORBID_DUP_KEYS

# 类型定义
HookFn = Callable[[str, Dict[str, Any]], None]
AsyncHookFn = Callable[[str, Dict[str, Any]], Awaitable[None]]

# 定义 Loguru 的类型别名
FormatFunction: TypeAlias = Callable[[Dict[str, Any]], str]
FilterFunction: TypeAlias = Callable[[Dict[str, Any]], bool]
FilterDict: TypeAlias = Dict[str, str]

async def _call_hook(hook: Union[HookFn, AsyncHookFn], level: str, ctx: Dict[str, Any]) -> None:
    """
    调用钩子函数，支持同步和异步钩子。

    Args:
        hook: 钩子函数
        level: 日志级别
        ctx: 日志上下文
    """
    if inspect.iscoroutinefunction(hook):
        await hook(level, ctx)
    else:
        hook(level, ctx)

def _wrap_hook(hook: Optional[Union[HookFn, AsyncHookFn]]) -> Optional[HookFn]:
    """
    包装钩子函数，确保异步兼容性。
    
    Args:
        hook: 原始钩子函数，可以是同步或异步函数
        
    Returns:
        Optional[HookFn]: 包装后的同步钩子函数
        
    Note:
        - 使用 anyio 处理异步函数，兼容已有事件循环
        - 同步函数直接返回
        - None 返回 None
    """
    if hook is None:
        return None
        
    def sync_wrapper(level: str, ctx: Dict[str, Any]) -> None:
        try:
            anyio.run(_call_hook, hook, level, ctx)
        except Exception as exc:
            _loguru_logger.opt(exception=exc).error("钩子函数执行失败: {}", exc)
            
    return sync_wrapper

class LoguruLogger(LoggerProtocol):
    """
    WHY: 为 Loguru 提供轻量级包装，确保与 LoggerProtocol 兼容
    WHAT: 封装 Loguru 的日志记录功能，保持无状态设计
    HOW: 使用 contextvars 管理上下文，所有方法委托给底层 logger
    """

    __slots__ = ("_logger", "_before", "_after", "_level")
    _CTX: ClassVar[ContextVar[Dict[str, Any] | None]] = ContextVar("log_ctx", default=None)
    _LOCK = threading.Lock()

    def __init__(self, logger: Optional[Any] = None) -> None:
        """
        初始化日志记录器。

        Args:
            logger: 底层 logger 实例，默认为 None
        """
        self._logger = logger or _loguru_logger
        self._before: Optional[HookFn] = None
        self._after: Optional[HookFn] = None
        self._level: str = "INFO"  # 默认日志级别

    def _prepare_context(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """准备日志上下文。"""
        base = self._CTX.get() or {}
        ctx = {**base, **(extra or {}), "timestamp": datetime.now(timezone.utc)}
        return ctx

    def _log(self, level: str, msg: str, ctx: Dict[str, Any]) -> None:
        """
        执行日志记录。

        Args:
            level: 日志级别
            msg: 日志消息
            ctx: 日志上下文

        Note:
            - 钩子函数的异常不会影响主日志逻辑
            - 钩子异常会被记录但不会传播
        """
        # 调用前置钩子
        if self._before:
            try:
                self._before(level, ctx)
            except Exception as exc:
                self._logger.opt(exception=exc).warning("before-hook error")

        # 记录日志，让 loguru 的 sink 处理格式化
        self._logger.bind(**ctx).log(level, msg)

        # 调用后置钩子
        if self._after:
            try:
                self._after(level, ctx)
            except Exception as exc:
                self._logger.opt(exception=exc).warning("after-hook error")

    def _call(self, level: str, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """同步调用日志记录。"""
        try:
            # 检查日志级别
            if _loguru_logger.level(level).no < _loguru_logger.level(self._level).no:
                return

            ctx = self._prepare_context(extra)
            self._log(level, msg, ctx)
        except Exception as exc:
            self._logger.exception(
                "日志记录失败",
                extra={
                    "level": level,
                    "message": msg,
                    "context": ctx,
                    "error": str(exc)
                }
            )
            raise

    async def _acall(self, level: str, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """异步调用日志记录。"""
        try:
            # 检查日志级别
            if _loguru_logger.level(level).no < _loguru_logger.level(self._level).no:
                return

            ctx = self._prepare_context(extra)
            await asyncio.to_thread(self._log, level, msg, ctx)
        except Exception as exc:
            self._logger.exception(
                "异步日志记录失败",
                extra={
                    "level": level,
                    "message": msg,
                    "context": ctx,
                    "error": str(exc)
                }
            )
            raise

    def bind(self, **kwargs: Any) -> Self:
        """
        绑定上下文到日志记录器。

        Args:
            **kwargs: 要绑定的上下文键值对

        Returns:
            Self: 绑定上下文后的新日志记录器实例

        Raises:
            ContextBindError: 当尝试重复绑定受限键时
        """
        try:
            with self._LOCK:
                # 检查重复绑定
                current_ctx = self._CTX.get()
                if current_ctx:
                    raise ContextBindError(
                        "禁止链式 bind",
                        context=ErrorContext.create(
                            error_type=ErrorType.CONTEXT_BIND,
                            details={
                                "current_context": current_ctx,
                                "attempted_bind": kwargs
                            }
                        )
                    )

                # 检查受限键
                dup_keys = FORBID_DUP_KEYS.intersection(kwargs.keys())
                if dup_keys:
                    raise ContextBindError(
                        f"尝试重复绑定受限键: {', '.join(dup_keys)}",
                        context=ErrorContext.create(
                            error_type=ErrorType.CONTEXT_BIND,
                            details={
                                "duplicate_keys": list(dup_keys),
                                "attempted_bind": kwargs
                            }
                        )
                    )

                # 创建新的上下文
                self._CTX.set(kwargs)

                # 返回新的实例
                new_logger = copy(self)
                return new_logger

        except Exception as exc:
            if not isinstance(exc, ContextBindError):
                raise ContextBindError(
                    f"上下文绑定失败: {str(exc)}",
                    context=ErrorContext.create(
                        error_type=ErrorType.CONTEXT_BIND,
                        details={
                            "attempted_bind": kwargs,
                            "error": str(exc)
                        }
                    )
                ) from exc
            raise

    def with_task(self, task_id: str) -> Self:
        """
        绑定任务ID到日志记录器。

        Args:
            task_id: 任务ID

        Returns:
            Self: 绑定任务ID后的新日志记录器实例

        Raises:
            ValidationError: 当任务ID格式无效时
        """
        if not task_id or not isinstance(task_id, str):
            raise ValidationError(
                "无效的任务ID",
                context=ErrorContext.create(
                    error_type=ErrorType.VALIDATION,
                    details={
                        "task_id": task_id,
                        "type": type(task_id).__name__
                    }
                )
            )
        return self.bind(task_id=task_id)

    def get_context(self) -> Dict[str, Any]:
        """
        获取当前日志记录器的上下文。

        Returns:
            当前上下文字典，如果上下文不是字典类型则返回空字典
        """
        ctx = self._CTX.get()
        return ctx if isinstance(ctx, dict) else {}

    def clear_context(self) -> None:
        """
        清除当前日志记录器的上下文。
        """
        self._CTX.set(None)

    def error_with_context(
        self,
        msg: str,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """
        记录带上下文的错误日志。

        Args:
            msg: 日志消息
            error: 异常对象
            **kwargs: 额外的上下文信息

        Raises:
            LoggingError: 日志记录失败时抛出
        """
        try:
            error_context = {
                "error": str(error),
                "error_type": type(error).__name__,
                "stack_trace": traceback.format_tb(error.__traceback__) if error.__traceback__ else "",
                **kwargs
            }
            self.error(msg, extra=error_context)
        except Exception as exc:
            raise LoggingError(
                f"错误日志记录失败: {str(exc)}",
                context=ErrorContext.create(
                    error_type=ErrorType.LOG_WRITE,
                    details={
                        "original_error": str(error),
                        "context": kwargs,
                        "error": str(exc)
                    }
                )
            ) from exc

    def trace_error(
        self,
        msg: str,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """
        记录带上下文的错误跟踪日志。

        Args:
            msg: 日志消息
            error: 异常对象
            **kwargs: 额外的上下文信息

        Raises:
            LoggingError: 日志记录失败时抛出
        """
        try:
            error_context = {
                "error": str(error),
                "error_type": type(error).__name__,
                "stack_trace": traceback.format_tb(error.__traceback__) if error.__traceback__ else "",
                **kwargs
            }
            self.trace(msg, extra=error_context)
        except Exception as exc:
            raise LoggingError(
                f"错误跟踪日志记录失败: {str(exc)}",
                context=ErrorContext.create(
                    error_type=ErrorType.LOG_WRITE,
                    details={
                        "original_error": str(error),
                        "context": kwargs,
                        "error": str(exc)
                    }
                )
            ) from exc

    def get_test_context(self) -> Dict[str, Any]:
        """
        获取测试上下文。

        Returns:
            测试上下文字典，如果上下文不是字典类型则返回空字典
        """
        ctx = self._CTX.get()
        return ctx if isinstance(ctx, dict) else {}

    def reset_test_context(self) -> None:
        """
        重置测试上下文。
        """
        self._CTX.set(None)

    def set_hooks(
        self,
        before: Optional[HookFn] = None,
        after: Optional[HookFn] = None,
    ) -> None:
        """
        设置日志钩子函数。

        Args:
            before: 日志记录前的钩子函数
            after: 日志记录后的钩子函数
        """
        self._before = _wrap_hook(before)
        self._after = _wrap_hook(after)

    def configure(self, config: LoggingConfigDTO) -> None:
        """
        配置日志记录器。

        Args:
            config: 日志配置DTO

        Note:
            - _level 仅用于 wrapper 级别的过滤
            - Loguru 的全局过滤取决于 sink 配置
            - 要改变 sink 的过滤级别，需要重新配置 sink
        """
        with self._LOCK:
            self._level = config.level.value
            # 注意：这里不设置 logger.level()，因为 Loguru 的过滤是在 sink 级别进行的
            # 如果需要改变 sink 的过滤级别，需要重新配置 sink

    def add(
        self,
        sink: Any,
        *,
        level: str = "INFO",
        format: Optional[Union[str, FormatFunction]] = None,
        filter: Optional[Union[str, FilterFunction, FilterDict]] = None,
        colorize: bool = False,
        serialize: bool = False,
        backtrace: bool = True,
        diagnose: bool = True,
        enqueue: bool = False,
        catch: bool = True,
        **kwargs: Any,
    ) -> int:
        """添加日志处理器。"""
        try:
            if not hasattr(self._logger, "add"):
                raise LoggingError(
                    "底层 logger 不支持添加处理器",
                    context=ErrorContext.create(
                        error_type=ErrorType.LOGGER_INIT,
                        details={
                            "logger_type": type(self._logger).__name__,
                            "sink": str(sink),
                            "level": level
                        }
                    )
                )
            
            with self._LOCK:
                # 构建参数字典，只在值不为 None 时添加参数
                add_kwargs = {
                    "level": level,
                    "colorize": colorize,
                    "serialize": serialize,
                    "backtrace": backtrace,
                    "diagnose": diagnose,
                    "enqueue": enqueue,
                    "catch": catch,
                    **kwargs
                }
                
                if format is not None:
                    add_kwargs["format"] = format
                if filter is not None:
                    add_kwargs["filter"] = filter
                    
                return self._logger.add(sink, **{k: v for k, v in add_kwargs.items() if v is not None})

        except Exception as exc:
            self._logger.exception(
                "添加日志处理器失败",
                extra={
                    "sink": str(sink),
                    "level": level,
                    "error": str(exc)
                }
            )
            raise

    def remove(self, sink_id: int) -> None:
        """
        移除日志处理器。

        Args:
            sink_id: 处理器ID
        """
        with self._LOCK:
            self._logger.remove(sink_id)

    # 基础日志方法
    def debug(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        self._call("DEBUG", msg, extra=extra)

    def info(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        self._call("INFO", msg, extra=extra)

    def warning(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        self._call("WARNING", msg, extra=extra)

    def error(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        self._call("ERROR", msg, extra=extra)

    def critical(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        self._call("CRITICAL", msg, extra=extra)

    def success(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        self._call("SUCCESS", msg, extra=extra)

    def trace(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        self._call("TRACE", msg, extra=extra)

    # 异步方法实现
    async def async_debug(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        await asyncio.to_thread(self.debug, msg, extra=extra)

    async def async_info(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        await asyncio.to_thread(self.info, msg, extra=extra)

    async def async_warning(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        await asyncio.to_thread(self.warning, msg, extra=extra)

    async def async_error(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        await asyncio.to_thread(self.error, msg, extra=extra)

    async def async_critical(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        await asyncio.to_thread(self.critical, msg, extra=extra)

    async def async_success(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        await asyncio.to_thread(self.success, msg, extra=extra)

    async def async_trace(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        await asyncio.to_thread(self.trace, msg, extra=extra)
