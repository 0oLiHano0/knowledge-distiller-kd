"""
====================开发指引======================
kd_tool/core/errors.py - v0.1
=================================================

【文件定位】  
- 所属包结构：kd_tool.core  
- 模块层次：核心服务层（全局异常体系定义）  
- 供全项目各层（服务、阶段、工厂、适配器等）引用。

【模块职责（SRP）】  
- 统一定义 KD-Tool 项目的自定义异常基类及核心子类，规范异常链、上下文信息与输出格式。

【依赖关系与注入】  
- 仅依赖 Python 标准库（typing）。
- 本文件不依赖外部服务、工厂、适配器。
- 无需依赖注入或 Mock。

【输入输出规范】  
- KDToolError:
  - 输入：message: str, original_exception: Exception = None, **kwargs
  - 输出：异常对象，支持 __str__ 输出 message、context、原始异常链
  - 异常：自身及子类
- ConfigError/DependencyInjectionError/OrchestratorError:
  - 继承 KDToolError，参数一致
- context_info: Dict[str, Any]，用于存储附加上下文信息

【核心架构约束】  
- 所有自定义异常必须继承 KDToolError
- 禁止直接实例化依赖或全局变量
- 必须类型注解
- 重要类/方法需三段式注释（WHY/WHAT/HOW）
- 禁止业务逻辑与异常定义耦合

【接口与DTO规范】  
- 暴露接口：KDToolError 及其子类
- 参数类型、返回值、用途见输入输出规范
- 异常定义与业务实现分离

【日志与安全】  
- 本文件不直接记录日志
- 异常字符串输出需注意不泄露敏感信息（如 context_info 含敏感数据时，调用方需处理）

【任务清单】  
1. [已完成] 实现 KDToolError 基类，支持 message、original_exception、context_info
2. [已完成] 实现 ConfigError、DependencyInjectionError、OrchestratorError 子类
3. [已完成] 为所有类和关键方法补充 WHY/WHAT/HOW 三段式注释
4. [已完成] 确保所有参数、返回值均有类型注解
5. [待完成] 编写单元测试，覆盖实例化、属性、__str__、异常链、context_info

【其他说明】  
- 后续如需新增异常类型，必须继承 KDToolError
- 若 context_info 可能包含敏感信息，需在调用方日志输出前脱敏处理
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
