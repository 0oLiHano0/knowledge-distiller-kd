# kd_tool/logging/errors.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from kd_tool.core.errors import KDToolError


class ErrorSeverity(Enum):
    FATAL = auto()     # 需要立即终止或降级启动
    RECOVERABLE = auto()  # 可通过重试/降级继续
    WARNING = auto()      # 仅提醒


@dataclass(frozen=True)
class LoggingError(KDToolError):
    code: str
    severity: ErrorSeverity = ErrorSeverity.FATAL
    detail: str | None = None

    def __str__(self) -> str:  # 统一可序列化
        return f"[{self.code}] {self.detail or super().__str__()}"


class LoggingConfigError(LoggingError):
    def __init__(self, detail: str):
        super().__init__("CONFIG", ErrorSeverity.FATAL, detail)


class LogSinkUnavailableError(LoggingError):
    def __init__(self, sink: str, detail: str | None = None):
        super().__init__("SINK_LOST", ErrorSeverity.RECOVERABLE, detail or f"sink={sink}")


class LogHookError(LoggingError):
    def __init__(self, hook_name: str, detail: str | None = None):
        super().__init__("HOOK_FAIL", ErrorSeverity.RECOVERABLE, detail or f"hook={hook_name}")
