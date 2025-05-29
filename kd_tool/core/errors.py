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
    WHY: 作为应用内所有自定义异常的统一祖先，便于全局捕获。
    WHAT: 提供 message、原始异常、上下文信息等核心属性。
    HOW: 通过 __init__ 传递并存储所有关键信息。
    """

    def __init__(self, message: str, original_exception: Optional[Exception]=None, **kwargs: Any):
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
            base_str += (
                f' (Caused by: {type(self.original_exception).__name__}: {str(self.original_exception)})'
            )
        return base_str
