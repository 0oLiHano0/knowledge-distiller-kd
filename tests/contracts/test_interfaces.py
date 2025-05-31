# tests/contracts/test_interfaces.py
"""
测试模块: 接口与契约符合性。
确保实现类严格遵守其声明的接口定义。
"""
import pytest
import abc
import sys # 导入 sys 用于检查模块是否已加载
from typing import Type, Callable, Any

# 从本地辅助模块导入
from tests.contracts.helpers import get_public_methods_from_class, compare_method_signatures

# 导入待测试的接口和实现
# 确保 StorageInterface 可导入
from kd_tool.storage.storage_interface import StorageInterface
from kd_tool.storage.sqlite_storage import SQLiteStorage
# 导入 InMemoryStorage 作为测试替身，并验证其契约
from tests.storage.in_memory_storage import InMemoryStorage
# 以后可以导入更多:
# from kd_tool.core.interfaces import StageInterface
# from kd_tool.stages.prefilter.prefilter_stage import PrefilterStage


@pytest.mark.parametrize("interface_class, implementation_class", [
    (StorageInterface, SQLiteStorage),
    # 添加对 InMemoryStorage 的契约测试
    (StorageInterface, InMemoryStorage),
    # 示例: (StageInterface, PrefilterStage), # 当 PrefilterStage 伪代码完善后
])
def test_implementation_fulfills_contract(
    interface_class: Type[abc.ABC],
    implementation_class: Type
) -> None:
    """
    why: 验证实现类是否严格遵守接口的公共API契约。
    what:
        1. 实现类是否为接口的子类。
        2. 接口的所有公共方法是否都在实现中存在。
        3. 每个方法的签名是否完全匹配。
        4. 实现类是否未添加接口中没有的额外公共方法。
        (收集所有错误，一次性报告)。
    how:
        使用 inspect 获取方法和签名，然后逐一比较。
        利用 get_public_methods_from_class 和 compare_method_signatures 辅助函数。
        收集所有 AssertionError 并最后报告。
    """
    errors = [] # 初始化错误列表

    # 前置检查：确保接口是抽象基类
    if not isinstance(interface_class, abc.ABCMeta):
        errors.append(f"接口 {interface_class.__name__} 不是有效的抽象基类 (abc.ABCMeta)。")

    # 1. 实现类是否为接口的子类 (可选但推荐)
    if not issubclass(implementation_class, interface_class):
        errors.append(f"{implementation_class.__name__} 未声明实现/继承 {interface_class.__name__}。")

    # 获取公共方法
    interface_methods = get_public_methods_from_class(interface_class)
    implementation_methods = get_public_methods_from_class(implementation_class)

    # Filter out methods starting with '_' from implementation_methods
    implementation_methods = {
        name: method for name, method in implementation_methods.items()
        if not name.startswith('_')
    }

    # 2. 接口的所有公共方法是否都在实现中存在
    for method_name, if_method_obj in interface_methods.items():
        if method_name not in implementation_methods:
            errors.append(
                f"契约错误 ({implementation_class.__name__}): "
                f"缺少接口方法 '{method_name}' (定义于 {interface_class.__name__})。"
            )
            continue # 跳过签名比较，因为方法不存在

        impl_method_obj = implementation_methods[method_name]

        # 检查抽象方法的实现 (这里仍然使用 assert，因为这是结构性检查)
        # 如果希望收集此错误，也可以改为 if not ... errors.append
        # 为了一次性暴露所有签名错误，我们将签名比较的 AssertionError 也捕获并收集
        if hasattr(if_method_obj, "__isabstractmethod__") and \
           if_method_obj.__isabstractmethod__:
            if hasattr(impl_method_obj, "__isabstractmethod__") and \
               impl_method_obj.__isabstractmethod__:
                 errors.append(
                     f"契约错误 ({implementation_class.__name__}): "
                     f"方法 '{method_name}' 在接口中是抽象的, 但实现中未被具体实现。"
                 )

        # 3. 每个方法的签名是否完全匹配
        try:
            compare_method_signatures(
                if_method_obj,
                impl_method_obj,
                class_name=implementation_class.__name__,
                method_name=method_name
            )
        except AssertionError as e:
            # 捕获签名比较的 AssertionError，并添加到错误列表中
            errors.append(f"方法签名不匹配 for {implementation_class.__name__}.{method_name}:\n  {e}")

    # 4. 实现类是否未添加接口中没有的额外公共方法
    # 这是我们新加入的严格规则，收集错误而不是立即失败
    for method_name in implementation_methods:
        if method_name not in interface_methods:
            errors.append(
                f"契约错误 ({implementation_class.__name__}): "
                f"实现了额外公共方法 '{method_name}' (未在接口 {interface_class.__name__} 中定义)。"
            )

    # 最后，如果收集到任何错误，则测试失败并报告所有错误
    assert not errors, "\n".join(errors)

# print(f"  [OK] 契约检查通过: {interface_class.__name__} vs {implementation_class.__name__}") # 这个打印语句会干扰输出，移除或注释掉