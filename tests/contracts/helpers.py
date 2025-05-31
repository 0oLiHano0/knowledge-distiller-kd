# tests/contracts/helpers.py
"""
接口与契约测试的辅助函数。
"""
import inspect
import typing # 导入 typing 以便引用 Any
import sys # 导入 sys 用于检查模块是否已加载
from typing import Type, Callable, Any

# 架构师注释: 确保所有注释是中文，并尽量简洁。

def get_public_methods_from_class(klass: Type) -> dict[str, Callable]:
    """
    why: 获取类的公共API，用于接口契约比较。
    what: 无，这是一个辅助函数。
    ho w: 遍历类的成员，筛选出非私有函数或方法。
    """
    methods = {}
    for name, member in inspect.getmembers(klass):
        # 非私有，且是函数或方法
        if not name.startswith('_') and \
           (inspect.isfunction(member) or inspect.ismethod(member)):
            methods[name] = member
    return methods

def compare_method_signatures(
    interface_method: Callable,
    implementation_method: Callable,
    class_name: str,  # 实现类的名称
    method_name: str   # 正在比较的方法名
) -> None:
    """
    why: 严格比较接口与实现方法的签名，确保一致。
    what: 无，这是一个辅助函数，供测试用例调用。
    how: 使用inspect比较参数数量、名称、种类、注解、默认值及返回注解。
    """
    # 获取方法签名对象
    interface_sig = inspect.signature(interface_method)
    implementation_sig = inspect.signature(implementation_method)

    # 1. 比较参数数量
    if len(interface_sig.parameters) != len(implementation_sig.parameters):
         assert False, (f"契约错误({class_name}.{method_name}): 参数数量不符。"
                        f"接口{len(interface_sig.parameters)}个, 实现{len(implementation_sig.parameters)}个。"
                        f"\n  接口参数: {list(interface_sig.parameters.keys())}"
                        f"\n  实现参数: {list(implementation_sig.parameters.keys())}")

    # 2. 逐一比较参数
    for param_name, interface_param in interface_sig.parameters.items():
        if param_name not in implementation_sig.parameters:
            assert False, (f"契约错误({class_name}.{method_name}): "
                           f"实现缺少参数'{param_name}'.")
        
        impl_param = implementation_sig.parameters[param_name]

        # 2a. 比较参数种类
        if interface_param.kind != impl_param.kind:
             assert False, (f"契约错误({class_name}.{method_name}): "
                            f"参数'{param_name}'种类不符。"
                            f"接口:{interface_param.kind}, 实现:{impl_param.kind}.")

        # 2b. 比较类型注解
        # 考虑 typing.Any 和 inspect.Parameter.empty 的特殊情况
        interface_param_annotation = interface_param.annotation
        implementation_param_annotation = impl_param.annotation

        # 特殊处理内置类型，确保 str == <class 'str'> 等比较通过
        if interface_param_annotation in (str, int, float, bool, list, dict, set, tuple, bytes):
             if implementation_param_annotation == type(interface_param_annotation):
                  implementation_param_annotation = interface_param_annotation # 标准化为同一表示

        if interface_param_annotation != typing.Any and interface_param_annotation != inspect.Parameter.empty and \
           interface_param_annotation != implementation_param_annotation:
             assert False, (f"契约错误({class_name}.{method_name}): "
                            f"参数'{param_name}'类型注解不符。"
                            f"接口:{interface_param_annotation}, 实现:{implementation_param_annotation}.")

        # 2c. 比较默认值
        if interface_param.default != impl_param.default:
             assert False, (f"契约错误({class_name}.{method_name}): "
                            f"参数'{param_name}'默认值不符。"
                            f"接口:{interface_param.default}, 实现:{impl_param.default}.")

    # 3. 比较返回类型注解
    # 考虑 typing.Any 和 inspect.Signature.empty 的特殊情况
    # 特别处理 NoneType 和 inspect.Signature.empty
    interface_return_annotation = interface_sig.return_annotation
    implementation_return_annotation = implementation_sig.return_annotation

    # 将 None 转换为 NoneType，方便比较
    if interface_return_annotation is None:
        interface_return_annotation = type(None)
    if implementation_return_annotation is None:
        implementation_return_annotation = type(None)

    # 如果接口有明确注解 (非 empty 或 Any)，则要求实现注解一致
    if interface_return_annotation not in (inspect.Signature.empty, typing.Any) and \
       interface_return_annotation != implementation_return_annotation:
         assert False, (f"契约错误({class_name}.{method_name}): "
                        f"返回类型注解不符。"
                        f"接口:{interface_sig.return_annotation}, 实现:{implementation_sig.return_annotation}.")