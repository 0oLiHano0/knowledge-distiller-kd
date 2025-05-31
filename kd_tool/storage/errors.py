# kd_tool/storage/errors.py
"""
WHY: 定义存储相关自定义异常。
WHAT: 仅声明异常类，便于后续扩展。
HOW: 继承 KDToolError，方法体留白。
"""
from kd_tool.core.errors import KDToolError


class StorageError(KDToolError):
    """WHY: 存储通用异常；WHAT: 统一捕获；HOW: 继承 KDToolError。"""
    pass


class StorageInitializationError(KDToolError):
    """存储初始化相关错误。"""
    pass


class DuplicateContentError(StorageError):
    """WHY: 重复内容异常；WHAT: 检测到重复内容时抛出；HOW: 继承 StorageError。"""
    pass


class RecordNotFoundError(StorageError):
    """WHY: 记录未找到异常；WHAT: 查询无结果时抛出；HOW: 继承 StorageError。"""
    pass
