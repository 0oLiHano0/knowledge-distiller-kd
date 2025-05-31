"""
why: 确保Orchestrator无状态、DI、只调度不含业务逻辑。
what: 检查无类级可变属性、构造参数DI、方法体留白或调度。
how: 用 inspect 检查Orchestrator类。
"""
import inspect
from kd_tool.core.orchestrator import Orchestrator

def test_orchestrator_stateless():
    """why: Orchestrator不得有类级可变属性。"""
    attrs = [k for k, v in vars(Orchestrator).items() if not k.startswith("__") and not callable(v)]
    assert not attrs, f"Orchestrator 有类级可变属性: {attrs}"

def test_orchestrator_init_di():
    """why: 构造参数必须为依赖注入。"""
    sig = inspect.signature(Orchestrator.__init__)
    assert len(sig.parameters) > 1, "Orchestrator 构造参数过少，未体现DI"

def test_orchestrator_methods_no_logic():
    """why: 关键方法体只能为pass/ellipsis/TODO/raise或调度。"""
    for meth_name, meth in inspect.getmembers(Orchestrator, inspect.isfunction):
        if meth_name.startswith("_"):
            continue
        src = inspect.getsource(meth)
        assert ("pass" in src or "..." in src or "TODO" in src or "raise" in src or "self." in src), f"{meth_name} 方法体应留白或仅调度" 