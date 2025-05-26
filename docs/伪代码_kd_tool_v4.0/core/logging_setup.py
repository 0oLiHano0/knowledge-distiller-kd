# kd_tool/core/logging_setup.py (v4.6 - 确认导入)
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

# --- 第三方库导入 ---
from loguru import logger, Logger # 导入 logger 和 Logger 类型

# --- 项目内部模块导入 ---

# [指令] 必须从 core/logging/logging_settings_models.py 导入 LoggingSettings
from .logging.logging_settings_models import LoggingSettings
from kd_tool.core.errors import KDToolError

# --- 自定义异常 ---
class LoggingSetupError(KDToolError):
    """当日志配置失败时抛出的异常。"""
    def __init__(self, message: str, original_exception: Optional[Exception] = None, **kwargs):
        # 调用 KDToolError 的构造函数，并添加一个特定的上下文信息 'module'
        super().__init__(f"日志配置错误: {message}",
                         original_exception=original_exception,
                         module="logging_setup", **kwargs)

# --- 核心配置函数 ---

def setup_logging(settings: LoggingSettings) -> Logger:
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
    # 使用 print 进行初始状态的打印，因为此时 logger 可能还未完全配置好或配置失败
    print(f"INFO: [logging_setup] 开始配置日志系统，级别: {settings.log_level}...")

    try:
        # 1. 清理现有配置 (handlers)
        # 这是关键步骤，以避免在重新配置或多次调用时出现重复的 handlers
        logger.remove()
        print("DEBUG: [logging_setup] 已移除现有的日志 handlers。")

        # 2. 配置控制台 sink (如果启用)
        if settings.log_to_console:
            logger.add(
                sys.stderr,  # 标准错误输出是日志的常见选择
                level=settings.log_level.upper(), # 确保级别是大写
                format=settings.log_format,
                colorize=True, # 为控制台输出启用颜色
                enqueue=settings.enqueue, # 根据配置决定是否异步
                serialize=settings.serialize, # 根据配置决定是否序列化为 JSON
                backtrace=True, # 开启更详细的异常回溯
                diagnose=True   # 开启更丰富的诊断信息
            )
            print(f"DEBUG: [logging_setup] 已添加控制台 sink。级别: {settings.log_level}, 异步: {settings.enqueue}, JSON序列化: {settings.serialize}")

        # 3. 配置文件 sink (如果启用)
        if settings.log_to_file:
            if not settings.log_file_path:
                # 尽管 Pydantic 模型会进行验证，但在此处进行防御性检查仍然是好习惯
                raise LoggingSetupError("已启用文件日志，但 'log_file_path' 未在 LoggingSettings 中提供。")

            logger.add(
                settings.log_file_path,
                level=settings.log_level.upper(),
                format=settings.log_format,
                rotation=settings.log_rotation, # 例如: "500 MB", "1 day", "00:00"
                retention=None, # 可选：配置日志保留策略，例如 "10 days" (当前未在settings中定义，可按需添加)
                compression="zip", # 对轮换的日志文件进行压缩
                enqueue=settings.enqueue,
                serialize=settings.serialize,
                backtrace=True,
                diagnose=True,
                encoding="utf-8" # 明确指定文件编码
            )
            print(f"DEBUG: [logging_setup] 已添加文件 sink 至 '{settings.log_file_path}'。级别: {settings.log_level}, 轮换: {settings.log_rotation}, 异步: {settings.enqueue}, JSON序列化: {settings.serialize}")

        # 4. (可选) 未来可以考虑根据需要为特定模块设置不同的日志级别
        # 例如: logger.level("SOME_VERBOSE_MODULE", level="DEBUG")

        print("INFO: [logging_setup] 日志系统配置成功。")
        return logger # 返回配置好的 logger 实例，供依赖注入使用

    except Exception as e:
        # 如果配置过程中发生任何错误，则捕获并抛出自定义的 LoggingSetupError
        # 这样上层调用者 (如 application_builder) 可以知道日志配置失败
        # 在抛出异常前，原始错误已经被打印，有助于调试配置问题
        print(f"CRITICAL: [logging_setup] 日志系统配置失败: {e}")
        raise LoggingSetupError(message=str(e), original_exception=e) from e