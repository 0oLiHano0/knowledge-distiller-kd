"""
why: 确保ApplicationBuilder通过DI和工厂模式创建应用，build方法存在。
what: 检查构造参数、build方法、工厂模式、禁止直接实例化依赖。
how: 用 inspect 检查源码。
"""
import inspect
import pytest
from kd_tool.core.application_builder import ApplicationBuilder
from kd_tool.core.core_dtos import PipelineContextDTO

def test_app_builder_uses_di_and_factory():
    """增强：收集所有依赖注入/工厂/实例化相关问题，一次性输出。"""
    errors = []
    sig = inspect.signature(ApplicationBuilder.__init__)
    if len(sig.parameters) <= 1:
        errors.append("ApplicationBuilder 构造参数过少，未体现DI")
    src = inspect.getsource(ApplicationBuilder)
    if "Factory" not in src:
        errors.append("ApplicationBuilder 未用工厂模式")
    if "def build" not in src:
        errors.append("缺少build方法")
    forbidden = ["= ", "new ", "import "]
    for idx, line in enumerate(src.splitlines()):
        # 跳过常见局部变量赋值（如stages = {}等）
        if any(f in line for f in forbidden) and "Factory" not in line and "__init__" not in line:
            line_strip = line.strip()
            # 允许局部变量赋值
            if line_strip.startswith(("stages =", "result =", "data =", "output =", "errors =")):
                continue
            if not ("Factory" in line or "self." in line):
                errors.append(f"ApplicationBuilder 不应直接实例化依赖: {line_strip} (line {idx+1})")
    if errors:
        pytest.fail("依赖注入/工厂相关问题:\n" + "\n".join(errors)) 