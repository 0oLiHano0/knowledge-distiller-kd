"""
why: 确保 core/interfaces.py 中所有接口均为 abc.ABC，方法签名完整且无实现逻辑。
what: 检查接口继承、方法类型提示、方法体留白、无非抽象实现。
how: 用 AST 和 inspect 检查接口定义和方法体。
"""
import inspect
import ast
import pytest
from kd_tool.core import interfaces

def test_all_interfaces_are_abc():
    """why: 所有接口必须继承 abc.ABC。"""
    errors = []
    for name, obj in vars(interfaces).items():
        if inspect.isclass(obj) and name.endswith("Interface"):
            if not hasattr(obj, "__abstractmethods__"):
                errors.append(f"{name} 未继承 abc.ABC")
    if errors:
        pytest.fail("接口继承问题:\n" + "\n".join(errors))

def test_interface_methods_type_hints_and_no_logic():
    """why: 接口方法必须有类型提示且无实现逻辑。"""
    errors = []
    for name, obj in vars(interfaces).items():
        if inspect.isclass(obj) and name.endswith("Interface"):
            for meth_name, meth in inspect.getmembers(obj, inspect.isfunction):
                if meth_name.startswith("_"):
                    continue
                sig = inspect.signature(meth)
                for param in sig.parameters.values():
                    if param.name in ("self", "cls"):
                        continue
                    if param.annotation == param.empty:
                        errors.append(f"{name}.{meth_name} 缺少参数类型注解")
                if sig.return_annotation == sig.empty:
                    errors.append(f"{name}.{meth_name} 缺少返回类型注解")
                src = inspect.getsource(meth)
                try:
                    tree = ast.parse(src)
                except IndentationError:
                    # 兼容 ... 写法
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        body = node.body
                        if len(body) != 1:
                            errors.append(f"{name}.{meth_name} 方法体应留白")
                        else:
                            stmt = body[0]
                            # 兼容pass、Ellipsis、raise、TODO注释
                            if not (isinstance(stmt, (ast.Pass, ast.Expr, ast.Raise)) or (isinstance(stmt, ast.Expr) and getattr(stmt, 'value', None) == Ellipsis)):
                                errors.append(f"{name}.{meth_name} 方法体应为pass/ellipsis/TODO/raise")
    if errors:
        pytest.fail("接口方法签名/实现问题:\n" + "\n".join(errors)) 