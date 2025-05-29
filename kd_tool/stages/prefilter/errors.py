"""
WHY: 定义预过滤阶段相关自定义异常。
WHAT: 仅声明异常类，便于后续扩展。
HOW: 继承 KDToolError，方法体留白。
"""
from kd_tool.core.errors import KDToolError


class PrefilterStageError(KDToolError):
    """WHY: 预过滤阶段通用异常；WHAT: 统一捕获；HOW: 继承 KDToolError。"""
    pass


class PrefilterError(KDToolError):
    """PrefilterStage 相关的基本异常。"""
    pass


class CzkawkaExecutionError(PrefilterError):
    """当 Czkawka 工具执行失败时抛出。"""

    def __init__(self, command: str, return_code: int, error_output: str):
        message = (
            f"Czkawka 执行失败 (返回码: {return_code}). 命令: '{command}'. 输出: {error_output}"
            )
        super().__init__(message=message, command=command, return_code=
            return_code, error_output=error_output)


class CzkawkaParseError(PrefilterError):
    """当解析 Czkawka 输出失败时抛出。"""

    def __init__(self, message: str, raw_output: str):
        full_message = f'Czkawka 输出解析失败: {message}'
        super().__init__(message=full_message, raw_output=raw_output)
