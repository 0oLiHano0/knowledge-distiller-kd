"""
=================================================
errors.py - P09 清理阶段错误定义 (v4.7)
=================================================

**模块功能**:

- 定义 P09 清理阶段可能抛出的特定异常。

---
"""

from kd_tool.core.errors import KDToolError
from pathlib import Path


class CleanupError(KDToolError):
    """清理阶段的基础错误类型。"""

    pass


class FileOperationError(CleanupError):
    """当执行文件系统操作 (移动、删除) 失败时抛出。"""

    def __init__(self, file_path: Path, operation: str, original_error: Exception):
        self.file_path = file_path
        self.operation = operation
        self.original_error = original_error
        super().__init__(
            f"执行文件操作 '{operation}' 于 '{file_path}' 时失败: {original_error}"
        )


class TrashDirectoryError(CleanupError):
    """当垃圾箱目录无效或无法访问时抛出。"""

    pass


class DecisionResolutionError(CleanupError):
    """当无法将决策映射到具体文件或操作时抛出。"""

    pass


class CleanupStageError(KDToolError):
    """WHY: 清理阶段通用异常；WHAT: 统一捕获；HOW: 继承 KDToolError。"""

    pass
