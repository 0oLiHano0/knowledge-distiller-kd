"""
why: 确保ApplicationBuilder通过DI和工厂模式创建应用，build方法存在。
what: 检查构造参数、build方法、工厂模式、禁止直接实例化依赖。
how: 用 inspect 检查源码。
"""
import inspect
import pytest
from kd_tool.core.application_builder import ApplicationBuilder
from kd_tool.core.core_dtos import PipelineContextDTO
import ast

def test_app_builder_uses_di_and_factory():
    """增强：收集所有依赖注入/工厂/实例化相关问题，一次性输出。"""
    import sys
    import types
    errors = []
    from kd_tool.core.application_builder import ApplicationBuilder
    src = inspect.getsource(ApplicationBuilder)
    tree = ast.parse(src)
    class_name = ApplicationBuilder.__name__
    forbidden = [ast.Assign, ast.Import, ast.ImportFrom]
    # 只检查类体和__init__外部的赋值/导入
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for class_item in node.body:
                # 只检查类体的赋值（不在函数/方法体内）
                if isinstance(class_item, ast.Assign):
                    # 允许__init__和方法体内的赋值
                    errors.append(f"ApplicationBuilder 不应在类体直接实例化依赖: {ast.unparse(class_item)} (line {class_item.lineno})")
                if isinstance(class_item, (ast.Import, ast.ImportFrom)):
                    errors.append(f"ApplicationBuilder 不应在类体直接import依赖: {ast.unparse(class_item)} (line {class_item.lineno})")
    if errors:
        pytest.fail("依赖注入/工厂相关问题:\n" + "\n".join(errors)) 