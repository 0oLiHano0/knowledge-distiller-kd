# =====================================================
# kd_tool/logging/factory.py
# =====================================================
"""注册 + 获取具体日志记录器实例的入口点。"""
from __future__ import annotations

from typing import Dict, Type

from kd_tool.logging.errors import LoggingConfigError
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.logging.settings import LoggingConfigDTO


_REGISTRY: Dict[str, Type[LoggerProtocol]] = {}


def register(name: str):
    """注册一个日志记录器实现。

    Args:
        name: 实现名称，用于后续通过 :class:`LoggerFactory` 获取。

    Returns:
        装饰器函数，接受一个 :class:`LoggerProtocol` 实现类。

    Raises:
        ValueError: 如果 *name* 已经被注册。
    """
    def decorator(cls: Type[LoggerProtocol]) -> Type[LoggerProtocol]:
        if name in _REGISTRY:
            raise ValueError(f"日志记录器实现 '{name}' 已注册")
        _REGISTRY[name] = cls
        return cls
    return decorator


class LoggerFactory:
    @staticmethod
    def create(cfg: LoggingConfigDTO, *, impl: str = "loguru") -> LoggerProtocol:
        provider_cls = _REGISTRY.get(impl)
        if provider_cls is None:
            raise LoggingConfigError(f"日志记录器实现 '{impl}' 未注册")
        return provider_cls.configure(cfg)
