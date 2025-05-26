"""
定义应用核心的、通用的自定义异常基类。

架构决策与约束:
- 此模块是应用级基础异常 KDToolError 的权威定义来源。
- 项目中所有其他模块的自定义异常基类 (如 StorageError, OrchestratorError) 
  【必须】继承自此文件中定义的 KDToolError。
- KDToolError 旨在提供统一的错误标识、基础的错误信息承载能力，并支持异常链。
- KDToolError 的构造函数签名和核心属性 (message, original_exception, context_info) 是固定的架构约定。
- KDToolError 的字符串表示 (__str__) 必须包含 message 和原始异常信息 (如果存在)；
  context_info 不默认包含在字符串表示中，但可通过属性访问。
- 通用错误属性如 error_code, is_user_facing 在当前版本不作为 KDToolError 的标准组成部分，
  以保持基类简洁，但保留未来按需添加的可能性。
"""
from typing import Optional, Any, Dict


class KDToolError(Exception):
    """
    KD Tool 应用中所有自定义异常的通用基类。

    职责:
    1. 作为应用内所有自定义异常的统一祖先。
    2. 提供通用的错误捕获点 (`except KDToolError:`).
    3. 承载核心错误信息: 描述性消息、可选的原始异常、以及可选的附加上下文。
    """

    def __init__(self, message: str, original_exception: Optional[Exception
        ]=None, **kwargs: Any):
        """
        构造 KDToolError。

        参数:
            message (str): 【必需】错误的主要描述信息。
            original_exception (Optional[Exception]): 【可选】导致此错误的原始底层异常（如果有）。
                                                   架构要求: 用于支持异常链，便于追踪错误的根源。
            **kwargs: 【可选】允许子类传递并存储额外的上下文信息。
                      这些信息将存储在 self.context_info 字典中。
        """
        super().__init__(message)
        self.message: str = message
        self.original_exception: Optional[Exception] = original_exception
        self.context_info: Dict[str, Any] = kwargs

    def __str__(self) ->str:
        """
        提供错误信息的字符串表示。

        架构约束:
        - 必须包含 self.message。
        - 如果 self.original_exception 存在，必须包含其类型和消息。
        - 【不】默认包含 self.context_info 的内容，以避免字符串过于冗长。
          需要 context_info 的调用者应直接访问该属性。
        """
        base_str = self.message
        if self.original_exception:
            base_str += (
                f' (Caused by: {type(self.original_exception).__name__}: {str(self.original_exception)})'
                )
        return base_str
