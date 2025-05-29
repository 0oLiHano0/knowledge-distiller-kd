"""
Architecture rule tests for KD-Tool v4 prototype.
Run with:  poetry run pytest -q
"""

import ast
import inspect
import pkgutil
import importlib
from pathlib import Path

import kd_tool  # 直接导入包


# ---------- 工具函数 ----------
def iter_source_files(package) -> list[Path]:
    """Return all .py files under the given package root."""
    pk_loader = pkgutil.get_loader(package.__name__)
    assert pk_loader and hasattr(pk_loader, "path")
    root = Path(pk_loader.path).parent
    return [p for p in root.rglob("*.py") if p.is_file()]


def ast_body_only_has_pass_or_ellipsis(tree: ast.AST) -> bool:
    """True if every function / method body is pass/ellipsis / TODO comment."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Empty body or single pass / ellipsis allowed
            body = node.body
            if len(body) == 1 and isinstance(body[0], (ast.Pass, ast.Expr)):
                if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    # Ellipsis literal or string TODO comment
                    continue
                if isinstance(body[0], ast.Pass):
                    continue
            return False
    return True


# ---------- T1: 所有模块应可导入 ----------
def test_all_modules_importable():
    for module_info in pkgutil.walk_packages(kd_tool.__path__, kd_tool.__name__ + "."):
        importlib.import_module(module_info.name)


# ---------- T2: 伪代码不得含真实业务逻辑 ----------
def test_only_pass_or_todo():
    for file in iter_source_files(kd_tool):
        tree = ast.parse(file.read_text())
        assert ast_body_only_has_pass_or_ellipsis(
            tree
        ), f"文件 {file} 含有具体实现代码，请留空或加 TODO 注释"


# ---------- T3: 关键接口存在 ----------
def test_required_interfaces_exist():
    from kd_tool.core import interfaces

    for attr in (
        "StageInterface",
        "StorageInterface",
        "UoWInterface",
    ):
        assert hasattr(
            interfaces, attr
        ), f"{attr} 缺失，请在 core/interfaces.py 中定义抽象基类"


# ---------- T4: ApplicationBuilder 使用依赖注入 ----------
def test_application_builder_uses_di():
    from kd_tool.core.application_builder import ApplicationBuilder

    src = inspect.getsource(ApplicationBuilder)
    assert "Factory" in src, "ApplicationBuilder 应通过 Factory 注入依赖，而非手动实例化"
    assert "def build" in src, "缺少 build 方法"


# ---------- T5: Orchestrator 不应持有可变全局状态 ----------
def test_orchestrator_stateless():
    from kd_tool.core.orchestrator import Orchestrator

    attrs = [
        name
        for name, value in vars(Orchestrator).items()
        if not name.startswith("__") and not callable(value)
    ]
    assert (
        len(attrs) == 0
    ), f"Orchestrator 包含类级别可变属性 {attrs}，应保持无状态"
