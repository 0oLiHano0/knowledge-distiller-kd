"""
=================================================
service.py - 日志服务 (v4.1)
=================================================

**模块功能**:

- **核心职责**: 提供 `LoggerProtocol` 的轻量包装，保持无状态。
- LoggingService：业务/测试可注入的轻薄包装，不污染原对象（无状态）
---
"""


# kd_tool/logging/service.py
from kd_tool.logging.protocols import LoggerProtocol

class LoggingService:
    """
    WHY : 为业务模块提供薄包装，方便 Mock  
    WHAT: 封装底层 logger 并保持无状态  
    HOW : 所有方法委托给注入对象
    """
    def __init__(self, logger: LoggerProtocol) -> None:
        self._logger = logger    # 依赖注入

    # -------- 快捷转发 ----------
    def debug(self, msg: str, **kw) -> None: self._logger.debug(msg, **kw)
    def info(self, msg: str, **kw) -> None: self._logger.info(msg, **kw)
    def warning(self, msg: str, **kw) -> None: self._logger.warning(msg, **kw)
    def error(self, msg: str, **kw) -> None: self._logger.error(msg, **kw)
    def exception(self, msg: str, **kw) -> None: self._logger.exception(msg, **kw)

    def with_task(self, task_id: str) -> "LoggingService":
        """
        WHY : 绑定任务 ID 形成新对象  
        WHAT: 返回新 LoggingService 实例  
        HOW : 调用 bind 生成子 logger
        """
        return LoggingService(self._logger.bind(task_id=task_id))

    def log(self, msg: str):
        """WHY: 统一日志接口；WHAT: 记录日志；HOW: TODO: 实现日志记录。"""
        pass
