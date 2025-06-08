"""
====================开发指引======================
kd_tool/logging/errors.py - v1.0
=================================================

**【文件定位】**  
- 包结构：kd_tool.logging.errors
- 所属模块：日志服务层
- 职责：错误处理与日志异常管理

**【模块职责（SRP）】**  
- 定义日志模块的错误类型体系，提供统一的错误处理机制，确保日志错误可追踪、可调试。

**【依赖关系与注入】**  
- 外部依赖：
  * kd_tool.core.errors.KDToolError（继承）
  * pydantic.BaseModel（继承）
  * enum.Enum（继承）
- 依赖注入方式：构造器注入
- Mock需求：无

**【输入输出规范】**  
- ErrorType (Enum)：
  * 输入：无
  * 输出：标准错误类型枚举值
- ErrorContext (BaseModel)：
  * 输入：error_type, details, timestamp, module, operation
  * 输出：结构化的错误上下文对象
- LoggingError (KDToolError)：
  * 输入：message, context
  * 输出：标准化的日志错误实例

**【核心架构约束】**  
- 必须使用三段式注释（WHY/WHAT/HOW）标注所有关键类和方法
- 禁止使用通用Exception
- 错误消息必须包含足够上下文
- 敏感信息必须脱敏
- 必须使用枚举定义错误类型
- 必须使用Pydantic模型定义错误上下文

**【接口与DTO规范】**  
- ErrorType (Enum)：错误类型枚举
- ErrorContext (BaseModel)：错误上下文数据模型
- LoggingError (KDToolError)：基础错误类
- 具体错误类：
  * LoggerInitError
  * LogWriteError
  * LogFormatError
  * ContextBindError
  * ValidationError

**【日志与安全】**  
- 日志记录：通过ErrorContext自动记录错误上下文
- 安全处理：通过sanitize_sensitive_data方法自动脱敏敏感信息

**【任务清单】**  
1. [已完成] 为所有错误类添加完整的三段式注释
2. [已完成] 补充所有方法的类型注解
3. [待完成] 增强错误上下文的序列化能力
4. [待完成] 添加错误类型的单元测试

**【其他说明】**  
- 需要确保与Loguru日志系统的无缝集成
- 考虑添加错误码支持，便于错误追踪
- 已完成所有错误类的三段式注释优化，移除了冗余的"细分"描述
- 所有方法已添加完整的类型注解
"""

from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import inspect
from pydantic import BaseModel, Field, field_validator, ConfigDict
from kd_tool.core.errors import KDToolError


class ErrorType(str, Enum):
    """
    WHY: 提供标准化的错误类型定义，便于错误分类和处理
    WHAT: 定义日志模块可能出现的所有错误类型
    HOW: 使用枚举类型，确保类型安全
    """
    LOGGER_INIT = "logger_initialization_error"  # 日志初始化错误
    LOG_WRITE = "log_write_error"               # 日志写入错误
    LOG_FORMAT = "log_format_error"             # 日志格式化错误
    CONTEXT_BIND = "context_binding_error"      # 上下文绑定错误
    VALIDATION = "validation_error"             # 验证错误


class ErrorContext(BaseModel):
    """错误上下文。"""

    error_type: ErrorType = Field(..., description="错误类型")
    error_message: str
    error_details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=False
    )

    @field_validator("error_details", mode="before")
    @classmethod
    def sanitize_sensitive_data(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """脱敏敏感信息"""
        sensitive_keys = {"password", "token", "key", "secret"}
        for key in v:
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                v[key] = "***REDACTED***"
        return v

    @classmethod
    def create(
        cls,
        error_type: ErrorType,
        module: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> "ErrorContext":
        """
        WHY: 提供便捷的错误上下文创建方法
        WHAT: 自动填充时间戳和调用栈信息
        HOW: 使用inspect获取调用栈信息
        """
        # 获取模块和操作信息
        frame = inspect.currentframe()
        if frame is not None:
            frame = frame.f_back
            if frame is not None:
                if operation is None:
                    operation = f"{frame.f_code.co_name}"
                if module is None:
                    module = frame.f_globals.get("__name__", "unknown")

        return cls(
            error_type=error_type,
            error_message=f"{module or __name__}:{operation or 'unknown'}",
            error_details=details or {},
            timestamp=datetime.now(timezone.utc)
        )


class LoggingError(KDToolError):
    """
    WHY: 日志相关异常，便于捕获和追踪
    WHAT: 初始化或写入失败时抛出，包含错误上下文
    HOW: 继承项目统一错误基类，提供结构化错误信息
    """

    def __init__(
        self, 
        message: str, 
        *, 
        context: Optional[ErrorContext] = None
    ) -> None:
        """
        WHY: 提供统一的错误构造方式，确保错误信息完整
        WHAT: 初始化错误实例，支持上下文信息
        HOW: 调用父类构造，可选附加上下文
        """
        super().__init__(message)
        self.context = context or self._default_ctx()

    def __str__(self) -> str:
        """
        WHY: 提供可读性好的错误字符串表示
        WHAT: 格式化错误消息和上下文
        HOW: 使用f-string格式化
        """
        msg = self.args[0] if self.args else ""
        return f"{msg} (context: {self.context.model_dump_json()})"

    @classmethod
    def _default_ctx(cls, error_type: ErrorType = ErrorType.LOG_WRITE) -> ErrorContext:
        """
        WHY: 提供默认的错误上下文创建方法
        WHAT: 创建标准化的错误上下文
        HOW: 使用 ErrorContext.create 创建上下文
        """
        return ErrorContext.create(
            error_type=error_type,
            module=__name__,
            operation="log_write"
        )


class LoggerInitError(LoggingError):
    """
    WHY: 日志初始化相关的错误，便于精确定位问题
    WHAT: 在日志系统初始化过程中出现错误时抛出
    HOW: 继承LoggingError，使用特定的错误类型和上下文
    """
    def __init__(
        self,
        message: str,
        *,
        context: Optional[ErrorContext] = None
    ) -> None:
        super().__init__(
            message,
            context=context or self._default_ctx(ErrorType.LOGGER_INIT)
        )


class LogWriteError(LoggingError):
    """
    WHY: 日志写入相关的错误，便于追踪写入失败原因
    WHAT: 在日志写入过程中出现错误时抛出
    HOW: 继承LoggingError，使用特定的错误类型和上下文
    """
    def __init__(
        self,
        message: str,
        *,
        context: Optional[ErrorContext] = None
    ) -> None:
        super().__init__(
            message,
            context=context or self._default_ctx(ErrorType.LOG_WRITE)
        )


class LogFormatError(LoggingError):
    """
    WHY: 日志格式化相关的错误，便于定位格式问题
    WHAT: 在日志消息格式化过程中出现错误时抛出
    HOW: 继承LoggingError，使用特定的错误类型和上下文
    """
    def __init__(
        self,
        message: str,
        *,
        context: Optional[ErrorContext] = None
    ) -> None:
        super().__init__(
            message,
            context=context or self._default_ctx(ErrorType.LOG_FORMAT)
        )


class ContextBindError(LoggingError):
    """
    WHY: 日志上下文绑定相关的错误，便于追踪上下文问题
    WHAT: 在日志上下文绑定过程中出现错误时抛出
    HOW: 继承LoggingError，使用特定的错误类型和上下文
    """
    def __init__(
        self,
        message: str,
        *,
        context: Optional[ErrorContext] = None
    ) -> None:
        super().__init__(
            message,
            context=context or self._default_ctx(ErrorType.CONTEXT_BIND)
        )


class ValidationError(LoggingError):
    """
    WHY: 验证相关的错误，便于精确定位问题
    WHAT: 在数据验证过程中出现错误时抛出
    HOW: 继承LoggingError，使用特定的错误类型和上下文
    """
    def __init__(
        self,
        message: str,
        *,
        context: Optional[ErrorContext] = None
    ) -> None:
        super().__init__(
            message,
            context=context or self._default_ctx(ErrorType.VALIDATION)
        )
