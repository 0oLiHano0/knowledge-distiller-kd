"""
=================================================
errors.py - 日志错误 (v4.1)
=================================================

**模块功能**:

- **核心职责**: 定义 `LoggingError`，作为日志操作的错误类型。
- LoggingError 必须继承 KDToolError

---
"""

# kd_tool/logging/errors.py
from kd_tool.core.errors import KDToolError


class LoggingError(KDToolError):
    """
    WHY : 细分日志相关异常，便于捕获
    WHAT: 初始化或写入失败时抛出
    HOW : 继承项目统一错误基类
    """
