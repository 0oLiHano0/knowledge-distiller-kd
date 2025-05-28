# kd_tool/storage/errors.py
"""
WHY  : 存储层专属错误类型集合。
WHAT : 细分异常，提高可观测性与可测试性。
HOW  : 继承 KDToolError，供业务精确捕获。
"""
from kd_tool.core.errors import KDToolError


class StorageError(KDToolError):
    """存储通用错误。"""


class StorageInitializationError(StorageError):
    """初始化失败。"""
    def __init__(self, backend: str, original: Exception) -> None:
        super().__init__(f"{backend} 初始化失败: {original}")
        self.backend = backend
        self.original = original


class DuplicateContentError(StorageError):
    """重复数据冲突。"""


class RecordNotFoundError(StorageError):
    """记录不存在。"""
