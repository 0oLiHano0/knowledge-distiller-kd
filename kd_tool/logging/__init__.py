# 对外只暴露 LoggerProtocol 和 LoggerFactory


# kd_tool/logging/__init__.py
"""
WHY : 对外集中导出  
WHAT: 暴露 LoggerFactory 与 LoggerProtocol  
HOW : 供其他模块绝对导入
"""
from kd_tool.logging.protocols import LoggerProtocol   # noqa: F401
from kd_tool.logging.factory import LoggerFactory      # noqa: F401
