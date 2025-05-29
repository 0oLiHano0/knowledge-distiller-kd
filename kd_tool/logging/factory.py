"""
=================================================
factory.py - 日志工厂 (v4.1)
=================================================

**模块功能**:

- **核心职责**: 创建 `LoggerProtocol` 实例。
- LoggerFactory：唯一与 loguru 耦合的地方，负责按配置装配 logger
---
"""


# kd_tool/logging/factory.py
from __future__ import annotations

from loguru import logger as _loguru_logger
from kd_tool.logging.settings import LoggingSettingsDTO
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.logging.errors import LoggingError

class LoggerFactory:
    """
    WHY : 集中构建日志实例并注入  
    WHAT: 生成带全局/任务上下文的 LoggerProtocol  
    HOW : 仅此文件与 loguru 耦合，遵循 DI
    """

    def __init__(self, settings: LoggingSettingsDTO) -> None:
        """
        WHY : 立即应用配置，确保确定性  
        WHAT: 保存基础 logger  
        HOW : 若失败抛 LoggingError
        """
        try:
            self._base = _loguru_logger.bind(app="kd_tool")
            self._configure(settings)
        except Exception as exc:   # noqa: BLE001
            raise LoggingError("日志配置失败") from exc

    def get_logger(self, *, task_id: str | None = None) -> LoggerProtocol:
        """
        WHY : 为调用方提供上下文日志  
        WHAT: 返回 LoggerProtocol 子实例  
        HOW : 使用 loguru.bind 追加字段
        """
        return self._base if task_id is None else self._base.bind(task_id=task_id)

    # ---------------- private ----------------
    def _configure(self, cfg: LoggingSettingsDTO) -> None:
        """
        WHY : 根据 DTO 设置输出目标  
        WHAT: 配置 stdout 及可选文件 sink  
        HOW : 调用 loguru.add
        """
        _loguru_logger.remove()                           # 清除默认 sink
        fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {extra[task_id]:-<8} | {message}"
        _loguru_logger.add(
            sink=lambda m: print(m, end=""),
            level=cfg.level,
            format=fmt,
            enqueue=True,
            serialize=cfg.log_serialize_json,
        )
        if cfg.log_file:
            _loguru_logger.add(
                cfg.log_file,
                rotation=cfg.rotation,
                retention=cfg.retention,
                level=cfg.level,
                format=fmt,
                enqueue=True,
                serialize=cfg.log_serialize_json,
            )
