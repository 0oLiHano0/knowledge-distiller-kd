"""
日志工厂模块
-----------
提供统一的日志配置接口。
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, List, Optional, Dict

from loguru import logger as _loguru_logger

from kd_tool.logging.settings import LoggingConfigDTO
from kd_tool.logging.service import LoguruLogger


class LoggingConfigError(RuntimeError):
    """
    WHY: 提供专门的日志配置错误类型
    WHAT: 表示日志配置过程中的错误
    HOW: 继承 RuntimeError，提供更具体的错误语义
    """
    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化配置错误。

        Args:
            message: 错误消息
            details: 错误详情
        """
        super().__init__(message)
        self.details = details or {}


# 默认配置缓存
_cfg_default: Optional[LoggingConfigDTO] = None


def _safe_add_sink(*, sink: Any, **cfg: Any) -> int:
    """
    安全添加日志处理器。

    Args:
        sink: 日志输出目标
        **cfg: 处理器配置参数

    Returns:
        int: 处理器ID

    Note:
        仅在值非 None 时才传参，避免静态类型告警
    """
    kwargs = {k: v for k, v in cfg.items() if v is not None}
    return _loguru_logger.add(sink, **kwargs)


def configure_logging(*, use_env: bool = False) -> List[int]:
    """
    配置日志系统。

    Args:
        use_env: 是否从环境变量加载配置

    Returns:
        List[int]: 新增的处理器ID列表

    Raises:
        LoggingConfigError: 配置失败时

    Note:
        可安全重复调用（第二次开始仅增量修改）
    """
    global _cfg_default
    cfg: Optional[LoggingConfigDTO] = None  # 提前定义，异常时可引用
    try:
        # 加载配置
        cfg = LoggingConfigDTO.from_env() if use_env else (_cfg_default or LoggingConfigDTO.default())
        if not use_env:
            _cfg_default = cfg  # 缓存，避免多次 default() I/O

        # 避免重复 remove：仅第一次执行清空
        if not getattr(configure_logging, "_has_init", False):
            _loguru_logger.remove()
            configure_logging._has_init = True  # type: ignore[attr-defined]

        added: List[int] = []

        # 配置控制台输出
        if cfg.console.enabled:
            added.append(
                _safe_add_sink(
                    sink=sys.stdout,
                    level=cfg.level.value,
                    format=cfg.format,
                    colorize=cfg.console.colorize,
                    backtrace=cfg.console.backtrace,
                    diagnose=cfg.console.diagnose,
                    enqueue=cfg.console.enqueue,
                    catch=cfg.console.catch,
                    serialize=cfg.serialize,
                )
            )

        # 配置文件输出
        if cfg.file.enabled:
            path = Path(cfg.file.path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            added.append(
                _safe_add_sink(
                    sink=str(path),
                    level=cfg.level.value,
                    format=cfg.file.format,
                    rotation=cfg.file.rotation,
                    retention=cfg.file.retention,
                    compression=cfg.file.compression,
                    serialize=cfg.file.serialize,
                    enqueue=True,
                    catch=True,
                )
            )

        return added

    except Exception as exc:  # pylint: disable=broad-except
        raise LoggingConfigError(
            f"日志配置失败: {exc}",
            details={
                "config": cfg.model_dump() if cfg else None,
                "error": str(exc),
                "use_env": use_env,
                "has_init": getattr(configure_logging, "_has_init", False)
            }
        ) from exc


def create_logger(*, use_env: bool = False) -> LoguruLogger:
    """
    创建并配置日志记录器。

    Args:
        use_env: 是否从环境变量加载配置

    Returns:
        LoguruLogger: 配置好的日志记录器实例

    Raises:
        LoggingConfigError: 配置失败时
    """
    configure_logging(use_env=use_env)
    return LoguruLogger(_loguru_logger)
