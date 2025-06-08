# =====================================================
# kd_tool/logging/providers/__init__.py
# =====================================================
"""注册默认的日志提供者。"""
from __future__ import annotations

from .loguru_impl import LoguruLogger  # 只导入，不要再 register
