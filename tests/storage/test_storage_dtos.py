"""
测试 storage 层相关的 DTOs 的架构符合性。

why: 确保 storage 模块定义的 DTOs 遵循 Pydantic 规范，字段类型明确，无业务逻辑。
what: 检查 DTO 类是否继承 BaseModel, 字段是否有类型提示，没有非 BaseModel 的公共方法。
how: 使用 inspect 检查类和方法属性。
"""

import inspect
import pytest
from pydantic import BaseModel

# TODO: 导入 storage 模块中的所有 DTOs 文件或模块，例如：
from kd_tool.storage.settings_models import StorageSettingsDTO # 假设设置 DTO 在此文件
# from kd_tool.storage.my_dto import MyStorageDTO # 如果有其他 DTOs

# TODO: 编写测试用例来验证 storage 层 DTOs 的 Pydantic 规则、字段类型等
# 例如：
# def test_storage_dto_validation():
#     data = {...}
#     dto = StorageDTO1(**data)
#     assert isinstance(dto, StorageDTO1)

# TODO: 添加其他测试用例，覆盖所有 storage 层 DTOs 

# 收集所有需要测试的 DTO 类
STORAGE_DTOS = [
    # TODO: 将实际的 Storage DTO 类添加到这里
    StorageSettingsDTO,
    # MyStorageDTO,
]

@pytest.mark.parametrize("dto_class", STORAGE_DTOS)
def test_dto_is_pydantic_model(dto_class):
    """
    why: 所有 Storage DTO 必须继承 Pydantic 的 BaseModel。
    what: 验证 DTO 类是否是 BaseModel 的子类。
    how: 使用 issubclass 进行检查。
    """
    assert issubclass(dto_class, BaseModel), f"{dto_class.__name__} 未继承 BaseModel"

@pytest.mark.parametrize("dto_class", STORAGE_DTOS)
def test_dto_fields_have_type_hints(dto_class):
    """
    why: 所有 Storage DTO 字段必须有类型注解。
    what: 验证 DTO 类定义的所有字段都有非 None 的类型注解。
    how: 检查 __annotations__ 属性。
    """
    hints = getattr(dto_class, '__annotations__', {})
    assert hints, f"{dto_class.__name__} 没有定义任何带有类型注解的字段"
    for field, typ in hints.items():
        assert typ is not None, f"{dto_class.__name__}.{field} 缺少类型注解"

@pytest.mark.parametrize("dto_class", STORAGE_DTOS)
def test_dto_no_business_methods(dto_class):
    """
    why: Storage DTO 不应包含业务逻辑方法。
    what: 验证 DTO 类除了 BaseModel 自身方法外，没有其他公共方法。
    how: 使用 inspect 检查成员，并排除 BaseModel 的方法。
    """
    # 获取 BaseModel 的所有公共方法
    base_model_methods = {name for name, obj in inspect.getmembers(BaseModel, inspect.isfunction) if not name.startswith("_")}
    
    errors = []
    for meth_name, meth in inspect.getmembers(dto_class, inspect.isfunction):
        if not meth_name.startswith("_") and meth_name not in base_model_methods:
             # 允许 field_validator 和 model_validator 这类 Pydantic 特有的类方法
            if meth_name not in {"field_validator", "model_validator"}:
                 errors.append(f"{dto_class.__name__} 不应有方法 {meth_name}")
                 
    assert not errors, "\n".join(errors) 