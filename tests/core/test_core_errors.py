"""
why: 确保所有自定义异常继承KDToolError，无业务逻辑。
what: 检查异常继承、方法体留白、无业务方法。
how: 用 inspect 检查异常类。
"""
import inspect
import kd_tool.core.errors as core_errors

def test_all_errors_inherit_kdtoolerror():
    """why: 所有异常必须继承KDToolError。"""
    base = getattr(core_errors, "KDToolError", None)
    assert base, "未定义KDToolError"
    for name, obj in vars(core_errors).items():
        if inspect.isclass(obj) and name.endswith("Error") and obj is not base:
            assert issubclass(obj, base), f"{name} 未继承KDToolError"

def test_error_methods_no_logic():
    """why: 异常类方法体只能为pass/ellipsis/TODO/raise。"""
    for name, obj in vars(core_errors).items():
        if inspect.isclass(obj) and name.endswith("Error"):
            for meth_name, meth in inspect.getmembers(obj, inspect.isfunction):
                if meth_name.startswith("_"):
                    continue
                src = inspect.getsource(meth)
                assert ("pass" in src or "..." in src or "TODO" in src or "raise" in src), f"{name}.{meth_name} 方法体应留白" 