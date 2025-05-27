"""
定义 Prefilter 阶段相关的自定义异常。
v4.x 更新: 修改 __init__ 调用方式，以匹配 KDToolError 使用 **kwargs 
            传递上下文信息的方式。
"""
from kd_tool.core.errors import KDToolError


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
