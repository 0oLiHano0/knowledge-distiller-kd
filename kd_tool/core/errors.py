"""
定义应用核心的、通用的自定义异常基类。

架构决策与约束:
- WHY: 统一异常体系，便于全局捕获和追踪。
- WHAT: 定义 KDToolError 及其核心属性和行为。
- HOW: 仅依赖标准库，所有自定义异常均继承自 KDToolError。
"""

from typing import Optional, Any, Dict


class KDToolError(Exception):
    """
    WHY: KD-Tool通用基础异常。
    WHAT: 作为所有自定义异常的基类。
    HOW: 支持原始异常链。
    """

    def __init__(self, message: str, original_exception: Exception = None, **kwargs):
        super().__init__(message)
        self.original_exception = original_exception
        self.context_info = kwargs if kwargs else {}

    def __str__(self):
        base_str = self.args[0]
        if self.context_info:
            base_str += f" | context: {self.context_info}"
        if self.original_exception:
            base_str += f" (Caused by: {type(self.original_exception).__name__}: {str(self.original_exception)})"
        return base_str


class ConfigError(KDToolError):
    """配置相关错误。"""


class DependencyInjectionError(KDToolError):
    """依赖注入相关错误。"""


class OrchestratorError(KDToolError):
    """编排器相关错误。"""

    def __init__(
        self,
        message: str,
        original_exception: Optional[Exception] = None,
        **kwargs: Any,
    ):
        """
        WHY: 构造异常时收集所有关键信息。
        WHAT: message 为主描述，original_exception 支持异常链，kwargs 存储上下文。
        HOW: 赋值到实例属性，供 __str__ 和外部访问。
        """
        super().__init__(message)
        self.message: str = message
        self.original_exception: Optional[Exception] = original_exception
        self.context_info: Dict[str, Any] = kwargs  # 附加上下文信息

    def __str__(self) -> str:
        """
        WHY: 统一异常的字符串输出，便于日志和调试。
        WHAT: 输出 message 和原始异常信息。
        HOW: 拼接 message 和 original_exception 的类型与内容。
        """
        base_str = self.message
        # 如有原始异常，追加其类型和消息
        if self.original_exception:
            base_str += f" (Caused by: {type(self.original_exception).__name__}: {str(self.original_exception)})"
        return base_str
