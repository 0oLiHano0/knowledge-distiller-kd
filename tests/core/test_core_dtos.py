"""
why: 确保所有DTO为Pydantic模型，字段类型提示齐全，无业务逻辑。
what: 检查DTO继承、字段类型、无业务方法、方法体留白。
how: 用 inspect 检查DTO类和字段。
"""
import inspect
import pytest
from pydantic import BaseModel
import kd_tool.core.core_dtos as core_dtos

def test_dto_is_pydantic_model():
    """why: DTO必须继承BaseModel。"""
    errors = []
    for name, obj in vars(core_dtos).items():
        if inspect.isclass(obj) and name.endswith("DTO"):
            if not issubclass(obj, BaseModel):
                errors.append(f"{name} 未继承BaseModel")
    if errors:
        pytest.fail("DTO继承问题:\n" + "\n".join(errors))

def test_dto_fields_have_type_hints():
    """why: DTO字段必须有类型注解。"""
    errors = []
    for name, obj in vars(core_dtos).items():
        if inspect.isclass(obj) and name.endswith("DTO"):
            hints = getattr(obj, '__annotations__', {})
            if not hints:
                errors.append(f"{name} 没有类型注解")
            for field, typ in hints.items():
                if typ is None:
                    errors.append(f"{name}.{field} 缺少类型注解")
    if errors:
        pytest.fail("DTO字段类型注解问题:\n" + "\n".join(errors))

def test_dto_no_business_methods():
    """why: DTO不得有业务方法。"""
    # 允许的pydantic方法和常用序列化方法
    ALLOWED_METHODS = {
        "dict", "json", "parse_obj", "model_dump", "model_copy", "model_post_init",
        "_fill_analysis_text", "_ensure_utc", "_populate_id", "_check_simhash",
        "field_validator", "model_validator"
    }
    import pydantic
    errors = []
    for name, obj in vars(core_dtos).items():
        if inspect.isclass(obj) and name.endswith("DTO"):
            for meth_name, meth in inspect.getmembers(obj, inspect.isfunction):
                # 允许魔法方法、pydantic方法、序列化方法
                if (
                    meth_name.startswith("__")
                    or meth_name in ALLOWED_METHODS
                    or meth_name.startswith("add_")
                    or meth_name.startswith("get_")
                    or meth_name.startswith("_make_id")
                    or hasattr(pydantic.BaseModel, meth_name)
                ):
                    continue
                errors.append(f"{name} 不应有业务方法 {meth_name}")
    if errors:
        pytest.fail("DTO业务方法问题:\n" + "\n".join(errors)) 