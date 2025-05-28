# LoggerProtocol，最小接口契约

"""最小日志接口契约，不依赖 loguru 运行时实体。"""
# kd_tool/logging/protocols.py
from __future__ import annotations
from typing import Protocol, runtime_checkable

@runtime_checkable
class LoggerProtocol(Protocol):
    """
    WHY : 为解耦业务与具体日志库  
    WHAT: 描述日志对象最小能力  
    HOW : 采用 typing.Protocol 定义结构化接口
    """
    def debug(self, msg: str, **kw) -> None: ...
    def info(self, msg: str, **kw) -> None: ...
    def warning(self, msg: str, **kw) -> None: ...
    def error(self, msg: str, **kw) -> None: ...
    def exception(self, msg: str, **kw) -> None: ...
    def bind(self, **kw) -> "LoggerProtocol": ...
    def trace(self, msg: str, **kw) -> None:
        self.debug(msg, **kw)
