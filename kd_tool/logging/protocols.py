# =====================================================
# kd_tool/logging/protocols.py
# =====================================================
"""最小化的、与实现无关的日志协议，用于 *kd_tool*。

业务代码应该 **只依赖这个接口**。任何具体的实现（Loguru、structlog、JSON 记录器、Loki 等）都可以通过实现 :class:`LoggerProtocol` 并注册到 :pyfunc:`kd_tool.logging.factory.register` 来使用。
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from kd_tool.logging.settings import LoggingConfigDTO


@runtime_checkable
class LoggerProtocol(Protocol):
    """*kd_tool* 所需的日志表面。

    每个实现 **必须** 支持这些方法。它们是大多数 Python 记录器已经暴露的严格子集。
    """

    # ──────────────────────────────────────────────────────────
    # 分级日志
    # ──────────────────────────────────────────────────────────
    def debug(self, msg: str, *, extra: dict[str, Any] | None = None) -> None: ...
    def info(self, msg: str, *, extra: dict[str, Any] | None = None) -> None: ...
    def warning(self, msg: str, *, extra: dict[str, Any] | None = None) -> None: ...
    def error(self, msg: str, *, extra: dict[str, Any] | None = None) -> None: ...
    def exception(self, msg: str, *, extra: dict[str, Any] | None = None) -> None: ...
    def success(self, msg: str, *, extra: dict[str, Any] | None = None) -> None: ...

    # ──────────────────────────────────────────────────────────
    # 上下文绑定（结构化日志）
    # ──────────────────────────────────────────────────────────
    def bind(self, **ctx: Any) -> "LoggerProtocol": ...

    # ──────────────────────────────────────────────────────────
    # 工厂配置
    # ──────────────────────────────────────────────────────────
    @classmethod
    def configure(cls, cfg: LoggingConfigDTO) -> "LoggerProtocol": ...
