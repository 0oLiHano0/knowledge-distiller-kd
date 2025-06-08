# =====================================================
# kd_tool/logging/__init__.py
# =====================================================
"""
kd_tool.logging 模块的公共接口层，通过导入机制实现业务逻辑与底层实现的松耦合。
"""
from __future__ import annotations

from kd_tool.logging.factory import LoggerFactory
from kd_tool.logging.settings import LoggingConfigDTO
from kd_tool.logging import providers  # 这样 loguru_impl.py 会被 import，注册逻辑会自动执行