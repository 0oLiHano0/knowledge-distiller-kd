"""
====================开发指引======================
kd_tool/logging/protocols.py - v4.2
=================================================

**【文件定位】**  
- 包结构：kd_tool.logging.protocols
- 所属模块：日志服务层 - 核心接口层
- 职责：定义日志模块的接口契约

**【模块职责（SRP）】**  
- 定义日志记录器和上下文的标准接口，确保日志系统的一致性和可扩展性
- 提供与 Loguru 兼容的接口定义，但不直接依赖 Loguru

**【依赖关系与注入】**  
- 外部依赖：
  * typing.Protocol：接口定义
  * pydantic.BaseModel：数据模型
  * datetime.datetime：时间戳
  * uuid：唯一标识生成
- 依赖注入方式：接口定义，不涉及具体注入
- Mock需求：支持所有接口的 Mock 实现

**【输入输出规范】**  
- LoggerProtocol：
  * 输入：标准日志方法参数
  * 输出：void
  * 异常：LoggingError 及其子类
- LoggingContextProtocol：
  * 输入：上下文操作方法参数
  * 输出：Any/Dict[str, Any]
  * 异常：LoggingError 及其子类
- LoggingContextDTO：
  * 输入：上下文数据
  * 输出：Dict[str, Any]
  * 异常：ValueError

**【核心架构约束】**  
1. 单一职责原则 (SRP)：
   - 每个接口/类必须只承担一种定义明确、高度内聚的职责
   - 禁止在接口中混合业务逻辑和实现细节

2. 高内聚、低耦合：
   - 接口内部元素必须紧密相关
   - 接口之间必须通过清晰定义的契约交互
   - 禁止直接共享可变内部状态

3. 清晰的接口契约：
   - 必须使用 Protocol 定义接口
   - 必须包含完整的类型提示
   - 必须定义明确的前置/后置条件

4. 面向可测试性：
   - 所有接口必须支持 Mock 实现
   - 必须提供测试支持方法
   - 必须支持依赖注入

5. 开闭原则：
   - 接口必须对扩展开放
   - 接口必须对修改封闭
   - 必须支持新功能通过扩展实现

**【接口与DTO规范】**  
1. LoggerProtocol：
   - 基础日志方法：debug/info/warning/error/success/trace
   - 上下文管理：bind/get_context/with_context/clear_context
   - 错误处理：error_with_context/trace_error
   - 配置管理：add/remove/configure
   - 测试支持：get_test_context/reset_test_context

2. LoggingContextProtocol：
   - 上下文操作：get/set/update/clear
   - 数据转换：to_dict/validate
   - 验证规则：必须验证输入数据
   - 安全处理：必须脱敏敏感信息

3. LoggingContextDTO：
   - 字段：log_id/level/timestamp/module/function/line/extra
   - 方法：create/to_dict
   - 验证器：validate_module/sanitize_sensitive_data
   - 不可变性：必须使用 frozen=True

**【日志与安全】**  
1. 日志记录规范：
   - 必须记录关键操作
   - 必须包含上下文信息
   - 必须使用结构化日志
   - 必须支持异步日志

2. 安全处理：
   - 必须脱敏敏感信息
   - 必须验证输入数据
   - 必须保护上下文数据
   - 必须支持上下文隔离

**【LoggingService 使用规范】**

1. 上下文绑定规则：
   - 禁止重复绑定：task_id, stage_name
   - 禁止链式绑定：所有上下文必须一次性绑定
   - 异步安全：使用 contextvars 确保上下文隔离

2. 使用模式：
   ```python
   # 1. 在流程入口处绑定 task_id
   logger = LoggingService(base_logger)
   task_logger = logger.with_task("task_123")
   
   # 2. 一次性绑定所有上下文
   context_logger = task_logger.bind(
       stage_name="preprocessing",
       custom_key="value"
   )
   
   # 3. 在异步环境中使用
   async def process_task(task_id: str):
       logger = LoggingService(base_logger)
       task_logger = logger.with_task(task_id)
       # 上下文自动隔离
       await process()
   ```

3. 错误处理：
   - 重复绑定受限键：抛出 ContextBindError
   - 上下文操作失败：抛出 LoggingError
   - 异步上下文丢失：自动隔离

4. 最佳实践：
   - 在流程入口处绑定 task_id
   - 一次性绑定所有需要的上下文
   - 避免链式绑定
   - 使用 with_task() 绑定任务 ID
   - 使用 bind() 绑定其他上下文

**【任务清单】**  
1. [已完成] 实现 LoggerProtocol 接口定义
2. [已完成] 实现 LoggingContextProtocol 接口定义
3. [已完成] 实现 LoggingContextDTO 数据模型
4. [已完成] 添加完整类型提示
5. [已完成] 添加三段式注释
6. [待完成] 增强错误处理机制
7. [待完成] 完善测试支持方法
8. [待完成] 添加性能优化考虑
9. [待完成] 增强接口文档
10. [待完成] 添加更多单元测试

**【其他说明】**  
- 接口设计已考虑与 Loguru 的兼容性
- 支持异步日志记录
- 支持结构化日志输出
- 支持上下文验证和序列化
- 支持敏感信息脱敏
- 支持测试环境配置
- 遵循架构设计规则的所有核心约束
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable, Any, Dict, Optional, Union, Callable, Awaitable, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
import inspect
import uuid

from kd_tool.logging.settings import LogLevel, LoggingConfigDTO


class LogField(str, Enum):
    """
    WHY: 提供标准化的日志字段枚举，避免硬编码
    WHAT: 定义日志系统中常用的字段名
    HOW: 使用 str 枚举，确保序列化兼容性
    """
    TASK_ID = "task_id"
    STAGE = "stage"
    EVENT = "event"
    MODULE = "module"
    FUNCTION = "function"
    LINE = "line"
    TIMESTAMP = "timestamp"
    LEVEL = "level"
    MESSAGE = "message"
    EXTRA = "extra"
    EXCEPTION = "exception"
    STACK_TRACE = "stack_trace"
    CONTEXT = "context"

    @classmethod
    def get_all_fields(cls) -> Dict[str, str]:
        """
        获取所有字段名称映射

        Returns:
            Dict[str, str]: 字段名称映射字典
        """
        return {field.name: field.value for field in cls}

    @classmethod
    def get_field_names(cls) -> List[str]:
        """
        获取所有字段名称列表

        Returns:
            List[str]: 字段名称列表
        """
        return [field.name for field in cls]

    @classmethod
    def get_field_values(cls) -> List[str]:
        """
        获取所有字段值列表

        Returns:
            List[str]: 字段值列表
        """
        return [field.value for field in cls]

    def __str__(self) -> str:
        """
        字符串表示

        Returns:
            str: 字段值
        """
        return self.value


class LoggingContextDTO(BaseModel):
    """
    WHY: 定义日志上下文的数据模型，确保日志记录的可追溯性
    WHAT: 提供日志记录时的上下文信息，不包含任务状态
    HOW: 使用 Pydantic 模型，支持上下文验证和序列化

    [架构设计规则符合性]
    1. 单一职责：
       - 仅负责日志记录上下文
       - 不包含任务状态信息
       - 不重复存储 task_id

    2. 上下文绑定：
       - 不直接管理 task_id
       - 通过 LoggerProtocol.bind() 绑定上下文
       - 避免链式绑定

    3. 数据安全：
       - 自动脱敏敏感信息
       - 不存储任务状态
       - 最小化上下文信息
    """
    model_config = ConfigDict(frozen=True, validate_assignment=True)

    log_id: str = Field(..., description="日志记录唯一标识")
    level: LogLevel = Field(..., description="日志级别")
    timestamp: datetime = Field(default_factory=datetime.now, description="日志时间戳")
    module: str = Field(..., description="日志来源模块")
    function: str = Field(..., description="日志来源函数")
    line: int = Field(..., description="日志来源行号")
    extra: Dict[str, Any] = Field(default_factory=dict, description="额外日志信息")
    task_id: Optional[str] = None
    stage_name: Optional[str] = None

    @field_validator("module")
    @classmethod
    def validate_module(cls, v: str) -> str:
        """验证模块名"""
        if not v or "." not in v:
            raise ValueError(f"Invalid module name: {v}")
        return v

    @field_validator("extra")
    @classmethod
    def validate_unique_keys(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证 extra 字段中是否包含受限键
        
        Args:
            v: 待验证的 extra 字典
            
        Returns:
            Dict[str, Any]: 验证后的字典
            
        Raises:
            ValueError: 包含受限键时抛出
        """
        from kd_tool.logging.service import _FORBID_DUP_KEYS
        dup_keys = _FORBID_DUP_KEYS.intersection(v.keys())
        if dup_keys:
            raise ValueError(f"Extra 字段包含受限键: {', '.join(dup_keys)}")
        return v

    @field_validator("extra")
    @classmethod
    def sanitize_sensitive_data(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """脱敏敏感信息"""
        sensitive_keys = {"password", "token", "key", "secret"}
        for key in v:
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                v[key] = "***REDACTED***"
        return v

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "log_id": self.log_id,
            "level": self.level,
            "timestamp": self.timestamp.isoformat(),
            "module": self.module,
            "function": self.function,
            "line": self.line,
            "extra": self.extra
        }

    @classmethod
    def create(
        cls,
        level: LogLevel,
        module: Optional[str] = None,
        function: Optional[str] = None,
        line: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> "LoggingContextDTO":
        """
        WHY: 提供便捷的上下文创建方法
        WHAT: 自动填充日志上下文信息
        HOW: 使用 inspect 获取调用信息

        [使用说明]
        1. 在 Stage 中使用：
           ```python
           # 在 Stage 中
           def process(self, context: PipelineContextDTO) -> PipelineContextDTO:
               # 使用已绑定 task_id 的 logger
               logger = context.run_logger
               # 创建日志上下文
               log_context = LoggingContextDTO.create(
                   level=LogLevel.INFO,
                   extra={"stage": "processing"}
               )
               # 记录日志
               logger.info("Processing started", **log_context.to_dict())
           ```

        2. 在 Service 中使用：
           ```python
           # 在 Service 中
           def some_method(self, logger: LoggerProtocol):
               # 创建日志上下文
               log_context = LoggingContextDTO.create(
                   level=LogLevel.INFO,
                   extra={"method": "some_method"}
               )
               # 记录日志
               logger.info("Method called", **log_context.to_dict())
           ```
        """
        frame = inspect.currentframe()
        if frame is not None:
            frame = frame.f_back
            if frame is not None:
                if function is None:
                    function = frame.f_code.co_name
                if module is None:
                    module = frame.f_globals.get("__name__", "unknown")
                if line is None:
                    line = frame.f_lineno

        return cls(
            log_id=str(uuid.uuid4()),
            level=level,
            module=module or "unknown",
            function=function or "unknown",
            line=line or 0,
            extra=extra or {}
        )


@runtime_checkable
class LoggingContextProtocol(Protocol):
    """
    WHY: 定义日志上下文的标准接口，确保上下文管理的一致性
    WHAT: 提供上下文数据的访问和管理方法
    HOW: 使用 Protocol 定义接口，确保类型安全

    [方法契约]
    1. get/set/update:
       - 前置条件：key 不为空
       - 后置条件：数据被正确访问/修改
       - 异常：KeyError, ValueError

    2. clear:
       - 前置条件：无
       - 后置条件：所有数据被清除
       - 异常：无

    3. to_dict/validate:
       - 前置条件：无
       - 后置条件：返回有效数据/验证结果
       - 异常：ValueError
    """
    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any) -> None: ...
    def update(self, **kwargs: Any) -> None: ...
    def clear(self) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...
    def validate(self) -> bool: ...


@runtime_checkable
class LoggerProtocol(Protocol):
    """
    WHY: 定义日志记录器的标准接口，确保日志系统的一致性和可扩展性
    WHAT: 提供统一的日志记录方法，支持同步和异步操作
    HOW: 使用 Protocol 定义接口，确保类型安全

    [方法契约]
    1. 日志级别方法：
       - 前置条件：msg 不为空
       - 后置条件：日志被正确记录
       - 异常：LoggingError

    2. 上下文管理：
       - 前置条件：无
       - 后置条件：上下文被正确绑定/清除
       - 异常：ContextBindError

    3. 配置管理：
       - 前置条件：config 有效
       - 后置条件：配置被正确应用
       - 异常：LoggingError
    """

    def debug(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        记录调试级别日志。

        Args:
            msg: 日志消息
            extra: 额外的上下文信息
        """
        ...

    def info(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        记录信息级别日志。

        Args:
            msg: 日志消息
            extra: 额外的上下文信息
        """
        ...

    def warning(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        记录警告级别日志。

        Args:
            msg: 日志消息
            extra: 额外的上下文信息
        """
        ...

    def error(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        记录错误级别日志。

        Args:
            msg: 日志消息
            extra: 额外的上下文信息
        """
        ...

    def critical(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        记录严重错误级别日志。

        Args:
            msg: 日志消息
            extra: 额外的上下文信息
        """
        ...

    def success(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        记录成功级别日志。

        Args:
            msg: 日志消息
            extra: 额外的上下文信息
        """
        ...

    def trace(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        记录跟踪级别日志。

        Args:
            msg: 日志消息
            extra: 额外的上下文信息
        """
        ...

    # 异步日志方法
    async def async_debug(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        异步记录调试级别日志。

        Args:
            msg: 日志消息
            extra: 额外的上下文信息
        """
        ...

    async def async_info(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        异步记录信息级别日志。

        Args:
            msg: 日志消息
            extra: 额外的上下文信息
        """
        ...

    async def async_warning(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        异步记录警告级别日志。

        Args:
            msg: 日志消息
            extra: 额外的上下文信息
        """
        ...

    async def async_error(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        异步记录错误级别日志。

        Args:
            msg: 日志消息
            extra: 额外的上下文信息
        """
        ...

    async def async_critical(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        异步记录严重错误级别日志。

        Args:
            msg: 日志消息
            extra: 额外的上下文信息
        """
        ...

    async def async_success(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        异步记录成功级别日志。

        Args:
            msg: 日志消息
            extra: 额外的上下文信息
        """
        ...

    async def async_trace(self, msg: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        异步记录跟踪级别日志。

        Args:
            msg: 日志消息
            extra: 额外的上下文信息
        """
        ...

    def bind(self, **kwargs: Any) -> "LoggerProtocol":
        """
        绑定上下文到日志记录器。

        Args:
            **kwargs: 要绑定的上下文键值对

        Returns:
            绑定上下文后的新日志记录器实例

        Raises:
            ContextBindError: 当尝试链式绑定上下文时
        """
        ...

    def with_task(self, task_id: str) -> "LoggerProtocol":
        """
        绑定任务ID到日志记录器。

        Args:
            task_id: 任务ID

        Returns:
            绑定任务ID后的新日志记录器实例

        Raises:
            ValidationError: 当任务ID格式无效时
        """
        ...

    def get_context(self) -> Dict[str, Any]:
        """
        获取当前日志记录器的上下文。

        Returns:
            当前上下文字典
        """
        ...

    def clear_context(self) -> None:
        """
        清除当前日志记录器的上下文。
        """
        ...

    def error_with_context(self, msg: str, error: Exception, context: Dict[str, Any]) -> None:
        """
        记录带上下文的错误日志。

        Args:
            msg: 日志消息
            error: 异常对象
            context: 上下文信息
        """
        ...

    def trace_error(self, msg: str, error: Exception, context: Dict[str, Any]) -> None:
        """
        记录带上下文的错误跟踪日志。

        Args:
            msg: 日志消息
            error: 异常对象
            context: 上下文信息
        """
        ...

    def add(self, sink: Any, level: str = "DEBUG", **kwargs: Any) -> int:
        """
        添加日志处理器。

        Args:
            sink: 日志输出目标
            level: 日志级别
            **kwargs: 其他配置参数

        Returns:
            处理器ID
        """
        ...

    def remove(self, handler_id: int) -> None:
        """
        移除日志处理器。

        Args:
            handler_id: 处理器ID
        """
        ...

    def configure(self, config: LoggingConfigDTO) -> None:
        """
        配置日志记录器。

        Args:
            config: 日志配置DTO
        """
        ...

    def get_test_context(self) -> Dict[str, Any]:
        """
        获取测试上下文。

        Returns:
            测试上下文字典
        """
        ...
