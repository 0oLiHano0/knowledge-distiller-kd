# =====================================================
# kd_tool/logging/settings.py
# =====================================================
"""日志记录器配置的轻量级 DTO。

调用应用程序（CLI、API 服务器、测试框架）负责将 YAML / ENV / CLI 标志转换为这个数据类，
在调用 :pyfunc:`kd_tool.logging.get_logger` 之前。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class LoggingConfigDTO:
    """日志记录层理解的最小设置。"""

    level: str = "INFO"  # 例如 "DEBUG" / "WARNING"
    console: bool = True  # 总是记录到 stdout/stderr？
    file_enabled: bool = False  # 启用文件 sink？
    file_path: str = "kd_tool.log"  # 当 *file_enabled* 为 True 时使用的路径

    # 新增配置项
    rotation: str = "10 MB"  # 日志文件轮转大小
    retention: str = "7 days"  # 日志文件保留时间
    fmt: str = (  # 日志格式
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level:<8}</level> | "
        "{name}:{function}:{line} - "
        "{message} "
        "<cyan>{extra}</cyan>"
    )
