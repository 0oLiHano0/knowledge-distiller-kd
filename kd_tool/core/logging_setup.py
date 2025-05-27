"""
此模块负责根据提供的设置，集中配置 Loguru 日志系统。

职责:
- 初始化并配置 Loguru 的 sinks (控制台, 文件)。
- 设置日志格式、级别、轮换、异步、序列化和其他参数。
- 提供配置日志系统的核心函数 `setup_logging`。

约束:
- 此模块【必须】在应用启动时【最先】被调用，【早于】任何需要日志记录的组件实例化。
- 【不应】包含任何业务逻辑。
- 配置【必须】完全依赖于 `LoggingSettings`。
- 强制依赖注入。
"""
import sys
from typing import Optional
from loguru import logger, Logger
from kd_tool.core.logging.logging_settings_models import LoggingSettings
from kd_tool.core.errors import KDToolError


class LoggingSetupError(KDToolError):
    """当日志配置失败时抛出的异常。"""

    def __init__(self, message: str, original_exception: Optional[Exception
        ]=None, **kwargs):
        super().__init__(f'日志配置错误: {message}', original_exception=
            original_exception, module='logging_setup', **kwargs)


def setup_logging(settings: LoggingSettings) ->Logger:
    """
    根据提供的设置配置全局 Loguru 日志记录器。

    它会清除任何现有的配置，并根据 `LoggingSettings` 设置新的 sinks。
    此函数不包含 `get_logger`，以强制依赖注入。

    参数:
        settings (LoggingSettings): 日志记录的配置对象。

    返回:
        Logger: 配置好的 Loguru 日志记录器实例。

    可能抛出的异常:
        LoggingSetupError: 如果在配置过程中发生错误。
    """
    print(f'INFO: [logging_setup] 开始配置日志系统，级别: {settings.log_level}...')
    try:
        logger.remove()
        print('DEBUG: [logging_setup] 已移除现有的日志 handlers。')
        if settings.log_to_console:
            logger.add(sys.stderr, level=settings.log_level.upper(), format
                =settings.log_format, colorize=True, enqueue=settings.
                enqueue, serialize=settings.serialize, backtrace=True,
                diagnose=True)
            print(
                f'DEBUG: [logging_setup] 已添加控制台 sink。级别: {settings.log_level}, 异步: {settings.enqueue}, JSON序列化: {settings.serialize}'
                )
        if settings.log_to_file:
            if not settings.log_file_path:
                raise LoggingSetupError(
                    "已启用文件日志，但 'log_file_path' 未在 LoggingSettings 中提供。")
            logger.add(settings.log_file_path, level=settings.log_level.
                upper(), format=settings.log_format, rotation=settings.
                log_rotation, retention=None, compression='zip', enqueue=
                settings.enqueue, serialize=settings.serialize, backtrace=
                True, diagnose=True, encoding='utf-8')
            print(
                f"DEBUG: [logging_setup] 已添加文件 sink 至 '{settings.log_file_path}'。级别: {settings.log_level}, 轮换: {settings.log_rotation}, 异步: {settings.enqueue}, JSON序列化: {settings.serialize}"
                )
        print('INFO: [logging_setup] 日志系统配置成功。')
        return logger
    except Exception as e:
        print(f'CRITICAL: [logging_setup] 日志系统配置失败: {e}')
        raise LoggingSetupError(message=str(e), original_exception=e) from e
