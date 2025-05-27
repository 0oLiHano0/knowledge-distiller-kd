"""
定义存储层特定的自定义异常。

架构决策与约束:
- 此模块是存储层所有自定义异常的权威定义来源。
- 所有存储层实现 (如 SQLiteStorage) 抛出的、旨在被上层捕获的特定错误，都应在此模块中定义或从此模块导入。
- 所有此模块中定义的异常都应继承自 StorageError 基类。
- StorageError 基类【必须】继承自应用级基础异常 KDToolError (定义在 core.errors.py)。
- 异常类应通过调用 KDToolError 的构造函数并使用 **kwargs 来传递和存储有助于调试和错误处理的上下文信息。
  这些信息将存储在 KDToolError 实例的 context_info 属性中。
- StorageInterface 的文档字符串应明确指出其方法可能抛出的、在此文件中定义的异常类型。
"""
from typing import Any, Optional
from kd_tool.core.errors import KDToolError


class StorageError(KDToolError):
    """
    存储层操作相关的自定义异常基类。
    所有其他存储特定异常都应从此类继承。
    它本身继承自 KDToolError，以融入应用统一的异常体系。
    """

    def __init__(self, message: str, original_exception: Optional[Exception
        ]=None, **kwargs: Any):
        """
        构造 StorageError。

        参数:
            message (str): 错误的主要描述信息。
            original_exception (Optional[Exception]): 导致此错误的原始底层异常（如果有）。
            **kwargs: 传递给 KDToolError 基类的其他上下文信息。
                      这些信息将存储在 KDToolError 实例的 context_info 属性中。
        """
        super().__init__(message, original_exception=original_exception, **
            kwargs)


class StorageConfigurationError(StorageError):
    """
    当存储配置无效或不完整时抛出。
    例如，缺少必要的连接字符串或后端类型不支持。
    """

    def __init__(self, message: str, setting_key: Optional[str]=None, **
        kwargs: Any):
        full_message = f'存储配置错误: {message}'
        if setting_key:
            full_message += f" (相关配置项: '{setting_key}')"
        super().__init__(full_message, setting_key=setting_key, **kwargs)


class StorageConnectionError(StorageError):
    """
    当数据库连接失败时抛出，通常在 `initialize()` 期间。
    """

    def __init__(self, message: str, connection_details: Optional[str]=None,
        original_exception: Optional[Exception]=None, **kwargs: Any):
        full_message = f'存储连接错误: {message}'
        if connection_details:
            full_message += f' (连接详情: {connection_details})'
        super().__init__(full_message, original_exception=
            original_exception, connection_details=connection_details, **kwargs
            )


class StorageOperationError(StorageError):
    """
    当通用的存储操作 (如读/写数据库) 失败时抛出。
    这是一个相对通用的错误，当没有更具体的错误类型适用时使用。
    """

    def __init__(self, operation: str, original_exception: Optional[
        Exception]=None, details: Optional[str]=None, **kwargs: Any):
        message = f"存储操作 '{operation}' 失败。"
        if details:
            message += f' 详情: {details}'
        super().__init__(message, original_exception=original_exception,
            operation=operation, details=details, **kwargs)


class RecordNotFoundError(StorageError):
    """
    当尝试访问一个在存储中不存在的特定记录时抛出。
    """

    def __init__(self, record_type: str, record_id: Any, details: Optional[
        str]=None, **kwargs: Any):
        message = f"类型为 '{record_type}' 的记录 (ID/标识: {record_id}) 未找到。"
        if details:
            message += f' {details}'
        super().__init__(message, record_type=record_type, record_id=
            record_id, details=details, **kwargs)


class DuplicateRecordError(StorageError):
    """
    当尝试创建一个因违反唯一性约束而导致重复的记录时抛出。
    例如，尝试插入具有已存在的主键或唯一键的记录。
    """

    def __init__(self, record_type: str, record_identifier: Any, details:
        Optional[str]=None, existing_record_id: Optional[Any]=None, **
        kwargs: Any):
        message = (
            f"尝试创建的类型为 '{record_type}' 的记录已存在 (基于标识: {record_identifier})。")
        if existing_record_id:
            message += f' 已存在记录的 ID: {existing_record_id}.'
        if details:
            message += f' {details}'
        super().__init__(message, record_type=record_type,
            record_identifier=record_identifier, existing_record_id=
            existing_record_id, details=details, **kwargs)


class TransactionError(StorageError):
    """
    与事务管理相关的错误 (begin, commit, rollback)。
    """

    def __init__(self, operation: str, message: Optional[str]=None,
        original_exception: Optional[Exception]=None, **kwargs: Any):
        detailed_message = f"事务操作 '{operation}' 失败。"
        if message:
            detailed_message += f' {message}'
        super().__init__(detailed_message, original_exception=
            original_exception, operation=operation, **kwargs)
