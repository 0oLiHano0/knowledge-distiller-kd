# tests/contracts/test_interfaces.py
"""
测试模块: 接口与契约符合性。
确保实现类严格遵守其声明的接口定义。
"""
import pytest
import abc
from typing import Type

# 从本地辅助模块导入
from tests.contracts.helpers import get_public_methods_from_class, compare_method_signatures

# 导入待测试的接口和实现
from kd_tool.storage.Storage_interface import StorageInterface
from kd_tool.storage.sqlite_storage import SQLiteStorage
# 以后可以导入更多:
# from kd_tool.core.interfaces import StageInterface
# from kd_tool.stages.prefilter.prefilter_stage import PrefilterStage


@pytest.mark.parametrize("interface_class, implementation_class", [
    (StorageInterface, SQLiteStorage),
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
    how:
        使用 inspect 获取方法和签名，然后逐一比较。
        利用 get_public_methods_from_class 和 compare_method_signatures 辅助函数。
    """
    # 前置检查：确保接口是抽象基类
    assert isinstance(interface_class, abc.ABCMeta), \
        f"接口 {interface_class.__name__} 不是有效的抽象基类 (abc.ABCMeta)。"

    # 1. 实现类是否为接口的子类 (可选但推荐)
    assert issubclass(implementation_class, interface_class), \
        (f"{implementation_class.__name__} 未声明实现/继承 {interface_class.__name__}。")

    # 获取公共方法
    interface_methods = get_public_methods_from_class(interface_class)
    implementation_methods = get_public_methods_from_class(implementation_class)

    # 2. 接口的所有公共方法是否都在实现中存在
    for method_name, if_method_obj in interface_methods.items():
        assert method_name in implementation_methods, \
            (f"契约错误 ({implementation_class.__name__}): "
             f"缺少接口方法 '{method_name}' (定义于 {interface_class.__name__})。")

        impl_method_obj = implementation_methods[method_name]

        # 检查抽象方法的实现
        if hasattr(if_method_obj, "__isabstractmethod__") and \
           if_method_obj.__isabstractmethod__:
            assert not (hasattr(impl_method_obj, "__isabstractmethod__") and
                        impl_method_obj.__isabstractmethod__), \
                (f"契约错误 ({implementation_class.__name__}): "
                 f"方法 '{method_name}' 在接口中是抽象的, 但实现中未被具体实现。")

        # 3. 每个方法的签名是否完全匹配
        try:
            compare_method_signatures(
                if_method_obj,
                impl_method_obj,
                class_name=implementation_class.__name__,
                method_name=method_name
            )
        except AssertionError as e:
            pytest.fail(f"方法签名不匹配 for {implementation_class.__name__}.{method_name}:\n  {e}")

    # 4. 实现类是否未添加接口中没有的额外公共方法
    # 这是我们新加入的严格规则
    for method_name in implementation_methods:
        assert method_name in interface_methods, \
            (f"契约错误 ({implementation_class.__name__}): "
             f"实现了额外公共方法 '{method_name}' (未在接口 {interface_class.__name__} 中定义)。")
    
    print(f"  [OK] 契约检查通过: {interface_class.__name__} vs {implementation_class.__name__}")