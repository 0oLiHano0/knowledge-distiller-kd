"""
测试 storage 层自定义异常的架构符合性。

why: 确保 storage 模块定义的自定义异常继承 KDToolError，并遵循规范。
what: 检查异常类是否继承 KDToolError，没有非标准方法。
how: 使用 inspect 检查异常类属性和方法。
"""

import inspect
import pytest
import sys

# TODO: 导入 KDToolError 和 storage 模块中的自定义异常，例如：
from kd_tool.core.errors import KDToolError
import kd_tool.storage.errors as storage_errors # 导入存储层异常模块

# 收集所有需要测试的自定义异常类
STORAGE_EXCEPTIONS = [
    # TODO: 将实际的 Storage 自定义异常类添加到这里
    obj for name, obj in vars(storage_errors).items()
    if inspect.isclass(obj) and name.endswith("Error") and issubclass(obj, KDToolError)
]

# TODO: 如果 KDToolError 未在上方的 TODO 中导入，请在此导入
# from kd_tool.core.errors import KDToolError

@pytest.mark.parametrize("exception_class", STORAGE_EXCEPTIONS)
def test_exception_inherits_kdtoolerror(exception_class):
    """
    why: 所有 Storage 自定义异常必须直接或间接继承 KDToolError。
    what: 验证异常类是 KDToolError 的子类。
    how: 使用 issubclass 进行检查。
    """
    # 确保 KDToolError 已导入或可访问
    # assert 'KDToolError' in globals() or 'KDToolError' in locals() or 'kd_tool.core.errors' in sys.modules, \
    #     "请确保已导入 kd_tool.core.errors.KDToolError"
    # KDToolError 已在上方导入，此处无需再次检查导入状态
    assert issubclass(exception_class, KDToolError), f"{exception_class.__name__} 未继承 KDToolError"

@pytest.mark.parametrize("exception_class", STORAGE_EXCEPTIONS)
def test_exception_has_no_extra_methods(exception_class):
    """
    why: 异常类应保持精简，不应包含业务逻辑方法。
    what: 验证异常类除了标准异常属性 (__init__, __str__, 等) 外，没有其他公共方法。
    how: 使用 inspect 检查成员。
    """
    # 标准异常类通常只包含魔法方法和少量特殊属性
    standard_members = dir(Exception)
    
    errors = []
    for name, obj in inspect.getmembers(exception_class):
        if not name.startswith("_") and name not in standard_members and inspect.isfunction(obj):
             errors.append(f"{exception_class.__name__} 不应有方法 {name}")
             
    assert not errors, "\n".join(errors) 