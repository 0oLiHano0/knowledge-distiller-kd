# =====================================================
# kd_tool/logging/providers/loguru_impl.py
# =====================================================
"""默认提供者 *Loguru* (https://github.com/Delgan/loguru)."""
from __future__ import annotations

import sys
from typing import Any, Final

from loguru import logger as _loguru

from kd_tool.logging.factory import register
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.logging.settings import LoggingConfigDTO

@register("loguru")
class LoguruLogger(LoggerProtocol):
    """由 *Loguru* 提供支持的 :class:`LoggerProtocol` 具体实现。"""

    _raw: Final = _loguru  # keep reference to global loguru logger

    # ──────────────────────────────── class helpers ───────────────────────────────
    @classmethod
    def configure(cls, cfg: LoggingConfigDTO) -> "LoguruLogger":
        """根据 *cfg* 配置 *Loguru* 的 sink 并返回新的包装器。"""
        try:
            cls._raw.remove()  # 清除之前添加的 sink
        except (ValueError, RuntimeError):
            # 当没有添加 sink 或已经清除时。
            pass

        # 控制台
        if cfg.console:
            cls._raw.add(
                sys.stderr,
                level=cfg.level,
                enqueue=False,
                backtrace=True,
                diagnose=True,
                format=cfg.fmt
            )

        # 可选的文件 sink
        if cfg.file_enabled:
            cls._raw.add(
                cfg.file_path,
                level=cfg.level,
                rotation=cfg.rotation,
                retention=cfg.retention,
                format=cfg.fmt,
                delay=True,  # 延迟文件创建，直到第一次写入
                backtrace=True,  # 添加回溯信息
                diagnose=True    # 添加诊断信息
            )

        # 返回包装器实例
        return cls()

    # ──────────────────────────────── 代理方法 ───────────────────────────────
    def debug(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:  # noqa: D401
        self._raw.debug(msg, extra=extra or {})

    def info(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self._raw.info(msg, extra=extra or {})

    def warning(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self._raw.warning(msg, extra=extra or {})

    def error(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self._raw.error(msg, extra=extra or {})

    def exception(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        # loguru 有 .exception 方法，它会自动捕获当前的异常信息
        self._raw.exception(msg, extra=extra or {})

    def success(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        # Loguru 没有原生的 success 级别，我们使用 info 级别并添加成功标记
        self._raw.info(f"✅ {msg}", extra=extra or {})

    # 上下文绑定返回新的包装器，所以它符合 :class:`LoggerProtocol` 契约。
    def bind(self, **ctx: Any) -> "LoguruLogger":
        bound_raw = self._raw.bind(**ctx)
        wrapper = object.__new__(LoguruLogger)
        object.__setattr__(wrapper, "_raw", bound_raw)
        return wrapper
