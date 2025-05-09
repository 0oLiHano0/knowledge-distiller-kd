"""
错误处理模块，提供统一的错误处理机制。

此模块定义了项目中使用的所有错误类型和错误处理工具。
"""

from typing import Optional, Dict, Any, Union
import logging
from pathlib import Path
import traceback

# 获取日志记录器
logger = logging.getLogger(__name__)

class KDError(Exception):
    """知识蒸馏工具的基础错误类"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

class FileOperationError(KDError):
    """文件操作相关错误"""
    pass

class ModelError(KDError):
    """模型相关错误"""
    pass

class AnalysisError(KDError):
    """分析过程相关错误"""
    pass

class UserInputError(KDError):
    """用户输入相关错误"""
    pass

class ConfigurationError(KDError):
    """配置相关错误"""
    pass

def handle_error(error: Exception, context: Optional[str] = None) -> None:
    """
    统一处理错误，记录日志并决定是否继续执行。

    Args:
        error: 发生的错误
        context: 错误发生的上下文信息

    Note:
        - 对于 KDError 及其子类，会记录详细信息
        - 对于其他异常，会记录完整的堆栈跟踪
        - 根据错误类型决定是否继续执行
    """
    error_context = f" in {context}" if context else ""
    
    if isinstance(error, KDError):
        # 处理自定义错误
        logger.error(f"错误{error_context}: {error}")
        if error.details:
            logger.debug(f"错误详情: {error.details}")
    else:
        # 处理其他异常
        logger.error(f"未预期的错误{error_context}: {error}", exc_info=True)
        logger.debug(f"完整堆栈跟踪:\n{traceback.format_exc()}")

def safe_file_operation(func):
    """
    文件操作的安全包装器，自动处理文件相关的错误。

    Args:
        func: 要包装的文件操作函数

    Returns:
        包装后的函数

    Note:
        - 自动处理文件不存在、权限错误等常见问题
        - 提供详细的错误信息
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise FileOperationError(
                f"文件不存在: {e.filename}",
                error_code="FILE_NOT_FOUND",
                details={"filename": e.filename}
            )
        except PermissionError as e:
            raise FileOperationError(
                f"权限不足: {e.filename}",
                error_code="PERMISSION_DENIED",
                details={"filename": e.filename}
            )
        except Exception as e:
            raise FileOperationError(
                f"文件操作失败: {str(e)}",
                error_code="FILE_OPERATION_FAILED",
                details={"original_error": str(e)}
            )
    return wrapper

def validate_file_path(path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    验证文件路径的有效性。

    Args:
        path: 要验证的路径
        must_exist: 路径是否必须存在

    Returns:
        验证后的 Path 对象

    Raises:
        FileOperationError: 当路径无效时
    """
    try:
        path_obj = Path(path).resolve()
        if must_exist and not path_obj.exists():
            raise FileOperationError(
                f"路径不存在: {path}",
                error_code="PATH_NOT_EXIST",
                details={"path": str(path)}
            )
        return path_obj
    except Exception as e:
        raise FileOperationError(
            f"无效的路径: {path}",
            error_code="INVALID_PATH",
            details={"path": str(path), "error": str(e)}
        ) 