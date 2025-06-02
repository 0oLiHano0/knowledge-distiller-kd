# tests/structure/test_static_arch_rules.py
"""
模块名称: 静态架构规则测试 (test_static_arch_rules.py)

why:
  此模块旨在通过静态分析Python代码文件，验证项目是否遵循了基础的、全局性的架构规则和约定。
  这些测试不依赖于运行时逻辑的正确性，而是检查代码的结构、导入方式、伪代码规范等。
  它是守护架构一致性和可维护性的第一道防线，尤其在伪代码和早期开发阶段。

what:
  - 验证所有模块是否可导入（包结构是否正确）。
  - 验证是否所有导入都是绝对导入。
  - 验证伪代码是否符合留白和注释规范。
  - 验证核心抽象接口是否在预期的模块中定义。
  - （未来可扩展）检查ORM模型是否被隔离在存储层之外。

how:
  - 使用 `pkgutil` 遍历包内所有模块。
  - 使用 `ast` (Abstract Syntax Tree) 模块解析Python源文件，检查导入语句、函数体等。
  - 使用 `importlib` 动态导入模块进行检查。
  - 断言（`assert`）用于验证规则是否被遵守，并在违反时提供清晰的错误信息。
"""

import ast
import pkgutil
import importlib
from pathlib import Path
import inspect # 用于后续可能的其他检查
import pytest # 确保导入 pytest 以使用其特性，如 pytest.fail
import importlib.util
import sys
import pydantic
import types

import kd_tool # 顶级包

# ---------- 工具函数 ----------
def iter_source_files(package) -> list[Path]:
    """
    why: 获取指定包下的所有 .py 文件路径，用于后续的静态分析。
    what: 遍历包目录，收集所有Python源文件。
    how: 使用 `pkgutil.get_loader` 获取包加载器，然后通过其路径递归查找。
         由于已创建 kd_tool/__init__.py，预期加载器应具有 path 属性。
    """
    pk_loader = pkgutil.get_loader(package.__name__)
    
    # 确保加载器有效并且可以获取路径信息
    assert pk_loader and (hasattr(pk_loader, "path") or hasattr(pk_loader, "get_filename")), \
        f"无法获取包 {package.__name__} 的路径信息。请确保它是具有 __init__.py 的常规包。"

    package_root_path_str = ""
    if hasattr(pk_loader, "path"): 
        package_root_path_str = pk_loader.path
    elif hasattr(pk_loader, "get_filename"):
        init_file_path = Path(pk_loader.get_filename(package.__name__))
        package_root_path_str = str(init_file_path.parent)
    
    assert package_root_path_str, f"未能从加载器 {pk_loader} 中确定包 {package.__name__} 的根路径。"

    root = Path(package_root_path_str)
    # 如果路径指向 __init__.py 文件，实际的包根目录是其父目录
    if root.is_file() and root.name == "__init__.py":
        root = root.parent
        
    return [p for p in root.rglob("*.py") if p.is_file()]


def ast_body_only_has_pass_or_ellipsis_or_todo_comment(tree: ast.AST) -> bool:
    """
    why: 验证函数/方法体是否符合伪代码的"逻辑留白"规范。
    what: 检查函数体是否仅包含 pass、ellipsis (...), 或符合特定前缀的TODO字符串常量。
          这个函数是控制伪代码"纯洁度"的核心。
    how: 遍历AST节点，检查 FunctionDef 和 AsyncFunctionDef 的函数体。
         - 允许空函数体 (例如接口中的抽象方法，或者尚未定义的函数)。
         - 允许函数体仅包含 `pass` 语句。
         - 允许函数体仅包含 `...` (Ellipsis)。
         - 允许函数体仅包含一个字符串常量，且该字符串以特定TODO前缀开头。
         - 除此之外的任何其他可执行语句都被视为"具体逻辑"。
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if not body:  # 允许空函数体
                continue

            # 规则1: 函数体只有一个节点
            if len(body) == 1:
                stmt = body[0]
                # 规则 1a: 允许单个 'pass'
                if isinstance(stmt, ast.Pass):
                    continue
                # 规则 1b: 允许单个 '...' (Ellipsis) 或符合TODO规范的字符串常量
                if isinstance(stmt, ast.Expr):
                    value_node = stmt.value
                    # 处理 '...' (Ellipsis)
                    if isinstance(value_node, ast.Ellipsis): # Python 3.8+ for ast.Ellipsis node
                        continue
                    if isinstance(value_node, ast.Constant) and value_node.value is Ellipsis: # Python 3.8+ for Constant(value=Ellipsis)
                        continue
                    # 在旧版本Python (e.g., <3.8), '...' 可能被解析为 NameConstant(value=Ellipsis)
                    if isinstance(value_node, ast.NameConstant) and value_node.value is Ellipsis: # For older Python versions
                        continue
                    
                    # 处理作为TODO注释的字符串常量
                    if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
                        comment_text = value_node.value.strip().upper()
                        if comment_text.startswith(("ARCHITECT_TODO:", "TODO:", "PSEUDO:", "IMPLEMENT_LATER:")):
                            continue
            
            # 如果不符合上述简单情况，则认为包含具体逻辑，除非所有语句都是上述允许的类型
            # （这个更复杂的检查逻辑可以后续添加，目前上面的单语句检查已经能覆盖很多情况）
            # 为了简化，我们先假设如果不是单 pass/ellipsis/TODO字符串，就可能包含逻辑
            # 除非我们在这里加入更细致的检查，允许函数体包含多个合规的TODO字符串或pass等组合。
            # 目前，如果函数体多于一个节点，且第一个节点不是上述简单情况，就标记为不合规。
            
            # 更严格的检查：遍历所有语句，看是否有不允许的类型
            contains_disallowed_logic = False
            for stmt_in_body in body:
                is_allowed_statement = False
                if isinstance(stmt_in_body, ast.Pass):
                    is_allowed_statement = True
                elif isinstance(stmt_in_body, ast.Expr):
                    value_node = stmt_in_body.value
                    if isinstance(value_node, ast.Ellipsis) or \
                       (isinstance(value_node, ast.Constant) and value_node.value is Ellipsis) or \
                       (isinstance(value_node, ast.NameConstant) and value_node.value is Ellipsis):
                        is_allowed_statement = True
                    elif isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
                        comment_text = value_node.value.strip().upper()
                        if comment_text.startswith(("ARCHITECT_TODO:", "TODO:", "PSEUDO:", "IMPLEMENT_LATER:")):
                            is_allowed_statement = True
                
                if not is_allowed_statement:
                    contains_disallowed_logic = True
                    break # 发现一个不合规的语句就足够了
            
            if contains_disallowed_logic:
                # print(f"发现具体逻辑在 {node.name} (L{node.lineno})") # 调试用
                return False # 发现具体实现代码
    return True


# ---------- 全局架构规则测试 ----------

def test_all_modules_importable():
    """
    why: 确保项目中的所有Python模块都是可导入的，没有基本的语法错误或导入错误。
         这是最基础的健康检查。
    what: 遍历 `kd_tool` 包及其所有子包中的模块，并尝试导入它们。
    how: 使用 `pkgutil.walk_packages` 查找模块，然后用 `importlib.import_module` 导入。
    """
    package = kd_tool
    prefix = package.__name__ + "."
    
    imported_modules_count = 0
    modules_with_errors = []
    for module_info in pkgutil.walk_packages(package.__path__, prefix):
        try:
            importlib.import_module(module_info.name)
            imported_modules_count += 1
        except Exception as e:
            # 收集所有导入失败的模块，而不是遇到第一个就失败，方便一次性看到所有问题
            modules_with_errors.append(f"模块 {module_info.name} 导入失败: {e}")
    
    if modules_with_errors:
        pytest.fail("\n".join(modules_with_errors))
    
    assert imported_modules_count > 0, "未能成功导入任何模块，请检查包结构和 __init__.py 文件。"


def test_pseudo_code_conformance():
    """
    why: 确保所有非接口、非DTO等特定豁免文件外的模块中的函数体符合"逻辑留白"规范。
         这是控制伪代码阶段代码"纯洁度"，确保架构意图清晰传达的核心测试。
    what: 遍历项目源文件，解析为AST，然后检查函数定义。
    how: 使用 `iter_source_files` 获取文件列表，然后用 `ast_body_only_has_pass_or_ellipsis_or_todo_comment` 检查。
         通过 `allowed_logic_paths_or_files` 列表来定义哪些文件或路径模式可以豁免此检查。
         您可以通过修改此列表来控制检查的严格程度和范围。
    """
    
    # --- 控制点：定义哪些文件/路径可以包含"必要结构代码"或"定义性代码" ---
    # 这个列表是您用来平衡"伪代码简洁性"和"可执行骨架必要性"的关键。
    # - 接口定义: 包含 `_interface.py` 或直接指定如 `kd_tool/core/interfaces.py`
    # - 数据模型: `dtos.py`, `settings_models.py`, `enums.py`, `models_sqlalchemy.py`
    # - 工厂和构建器: `factory.py`, `application_builder.py` (它们需要实例化和连接逻辑)
    # - CLI 和配置入口: `cli_main.py`, `config.py`
    # - 核心组件的骨架: 如 `orchestrator.py`, 各个 `xxx_stage.py`, `sqlite_storage.py`
    #   这些文件需要包含 `__init__` 中的依赖赋值，以及核心方法的框架（可能调用其他组件）。
    # - 测试工具和辅助模块: `in_memory_storage.py`, `helpers.py`, `protocols.py`
    # - 测试文件自身和 Pytest 配置文件: `tests/`, `conftest.py`
    allowed_logic_paths_or_files = [
        # 数据和定义类文件 (通常只包含声明)
        "models_sqlalchemy.py",
        "settings_models.py",
        "dtos.py",
        "enums.py",
        "protocols.py", # 如 logging/protocols.py

        # 错误定义文件
        "errors.py",  # 匹配所有阶段、存储、核心的错误定义文件

        # 服务骨架文件
        "service.py",  # 如日志服务骨架

        # SimHashAnalysis 阶段配置模型需将计算结果转换为0～1的值（允许结构性代码）
        "simhash_analysis/settings_models.py",

        # 接口定义文件 (文件名或特定路径)
        "_interface.py", # 约定接口文件名以 _interface.py 结尾
        "kd_tool/core/interfaces.py",
        "kd_tool/storage/storage_interface.py",
        "adapter_interface.py", # 各种适配器的接口

        # 工厂和构建器 (需要实例化和连接逻辑)
        "factory.py",
        "application_builder.py",

        # 入口和配置
        "cli_main.py",
        "config.py",

        # 核心组件骨架 (允许包含必要的结构性代码和连接代码)
        # 在这些文件中，具体的业务逻辑部分仍需用 TODO/PSEUDO 注释
        "kd_tool/core/orchestrator.py",
        "kd_tool/storage/sqlite_storage.py", # 实现接口，需要方法体
        "stage.py", # 匹配所有如 xxx_stage.py, yyy_stage.py 的文件
        "adapter.py", # 各种适配器的具体实现，通常会有逻辑

        # 测试相关的工具和辅助模块
        "in_memory_storage.py",
        "helpers.py", # 测试辅助函数

        # 测试目录和配置文件 (不检查测试代码自身)
        "tests/",
        "conftest.py",
    ]
    # 您可以根据需要添加更精确的路径，例如:
    # "kd_tool/stages/prefilter/prefilter_stage.py" 
    # 来精确控制某个特定文件。
    # --- 结束控制点 ---

    non_compliant_files = []
    source_files = iter_source_files(kd_tool)
    assert source_files, "未能获取到任何源文件进行检查。"

    # 获取 kd_tool 包的根目录路径，用于后续比较
    try:
        kd_tool_package_dir = Path(kd_tool.__file__).parent
    except AttributeError:
        pytest.fail("无法确定 kd_tool 包的根目录，可能是 __init__.py 问题或包未正确安装。")

    for file_path in source_files:
        path_str = str(file_path.resolve()) # 使用绝对路径进行比较，更可靠

        # --- 控制点：排除逻辑开始 ---
        # 规则1: 跳过测试文件自身 (本测试文件)
        if path_str == str(Path(__file__).resolve()):
            continue

        # 规则2: 跳过在 `allowed_logic_paths_or_files` 中定义的文件或路径模式
        # 这个 `any` 表达式会检查 `path_str` 是否包含列表中的任何一个关键词。
        # 例如，如果 "dtos.py" 在列表中，任何路径包含 "dtos.py" 的文件都会被跳过。
        # 如果 "stages/" 在列表中，则 "kd_tool/stages/some_stage.py" 会被跳过。
        # 调整 `allowed_logic_paths_or_files` 的内容和精确度是控制此测试行为的关键。
        if any(allowed_keyword in path_str for allowed_keyword in allowed_logic_paths_or_files):
            continue
        
        # 规则3: 跳过直接位于 kd_tool 包根目录下的文件 (如 kd_tool/__init__.py)
        # 这些文件通常只包含导入或非常简单的包级别设置。
        if file_path.parent.resolve() == kd_tool_package_dir.resolve():
            continue
        # --- 结束控制点：排除逻辑 ---

        try:
            tree = ast.parse(file_path.read_text(encoding='utf-8'))
            if not ast_body_only_has_pass_or_ellipsis_or_todo_comment(tree):
                non_compliant_files.append(str(file_path))
        except Exception as e:
            # 如果文件解析失败，也视为测试失败，因为它表明代码有问题
            pytest.fail(f"解析文件 {file_path} 以检查伪代码规范时失败: {e}")

    if non_compliant_files:
        failure_message = "以下文件似乎包含了具体实现代码，而它们没有在豁免列表中。\n" \
                          "请检查这些文件是否应该：\n" \
                          "1. 遵循伪代码规范 (方法体使用 pass/ellipsis/TODO注释)。\n" \
                          "2. 或者，如果它们按设计就应包含结构性/定义性代码，请将其路径或模式添加到\n" \
                          "   `test_pseudo_code_conformance` 函数内的 `allowed_logic_paths_or_files` 列表中进行豁免。\n" \
                          "不合规文件列表:\n"
        failure_message += "\n".join(sorted(non_compliant_files)) # 排序方便查看
        pytest.fail(failure_message)


def test_required_core_interfaces_exist():
    """
    why: 确保核心层定义的关键抽象接口存在。这是架构契约的一部分。
    what: 检查 `kd_tool.core.interfaces` 模块是否包含预期的接口名称。
    how: 动态导入并使用 `hasattr` 检查。
    """
    from kd_tool.core import interfaces as core_interfaces #

    expected_core_interfaces = (
        "StageInterface", #
        # "UoWInterface", # 如果未来在这里定义
    )
    missing_interfaces = [
        attr for attr in expected_core_interfaces if not hasattr(core_interfaces, attr)
    ]
    assert not missing_interfaces, \
        f"核心接口缺失，请在 kd_tool/core/interfaces.py 中定义: {', '.join(missing_interfaces)}"

def test_required_storage_interfaces_exist():
    """
    why: 确保存储层定义的关键抽象接口存在。这是架构契约的一部分。
    what: 检查 `kd_tool.storage.storage_interface` 模块是否包含预期的接口名称。
    how: 动态导入并使用 `hasattr` 检查。
    """
    from kd_tool.storage import storage_interface as storage_iface_module

    expected_storage_interfaces = (
        "StorageInterface",
    )
    missing_interfaces = [
        attr for attr in expected_storage_interfaces if not hasattr(storage_iface_module, attr)
    ]
    assert not missing_interfaces, \
        f"存储接口缺失，请在 kd_tool/storage/storage_interface.py 中定义: {', '.join(missing_interfaces)}"


def test_required_logger_protocol_exists():
    """
    why: 确保日志协议 LoggerProtocol 存在于 logging 层。
    what: 检查 `kd_tool.logging.protocols` 模块是否包含 LoggerProtocol。
    how: 动态导入并使用 hasattr 检查。
    """
    from kd_tool.logging import protocols as logging_protocols_module
    assert hasattr(logging_protocols_module, "LoggerProtocol"), \
        "LoggerProtocol 未在 kd_tool/logging/protocols.py 中定义"


def test_no_relative_imports_in_project():
    """
    why: 确保项目遵循"强制使用绝对导入"的架构规则，以保证导入的清晰性和可维护性。
    what: 遍历所有非测试的 .py 文件，检查是否存在相对导入。
    how: 使用 `ast` 解析文件，查找 `ImportFrom` 节点并检查其 `level` 属性。
         `level > 0` 表示相对导入。
    """
    project_root_pkg = kd_tool
    non_absolute_imports_found = {}

    # 获取 tests 目录的绝对路径，以便准确排除
    try:
        # __file__ 是当前测试文件的路径
        tests_dir_path = Path(__file__).parent.parent.resolve()
    except NameError: # 如果在某些特殊执行环境下 __file__ 未定义
        # 尝试从 kd_tool 包的位置推断 tests 目录
        # 这假设 tests 目录与 kd_tool 目录在同一父级下
        try:
            kd_tool_module_path = Path(kd_tool.__file__).parent
            tests_dir_path = kd_tool_module_path.parent / "tests"
            if not tests_dir_path.is_dir(): # 如果推断不正确，设置一个无效路径以避免错误排除
                 tests_dir_path = Path("___INVALID_TESTS_PATH_SENTINEL___")
        except (AttributeError, TypeError): # 如果 kd_tool 不是常规包
             tests_dir_path = Path("___INVALID_TESTS_PATH_SENTINEL___")


    for file_path in iter_source_files(project_root_pkg):
        # --- 控制点：排除测试文件自身 ---
        # 这个检查是为了确保我们不分析测试代码中的导入行为，
        # 因为测试代码有时为了方便组织，可能会使用相对导入其局部的辅助模块。
        # 架构规则主要针对生产代码（kd_tool 包内）。
        is_test_file = False
        try:
            if tests_dir_path.exists() and file_path.resolve().is_relative_to(tests_dir_path):
                is_test_file = True
        except Exception: 
             # 如果 tests_dir_path 无效或比较出错，保守地认为不是测试文件，继续检查
             pass
        if is_test_file:
            continue
        # --- 结束控制点 ---

        try:
            source_code = file_path.read_text(encoding='utf-8')
            tree = ast.parse(source_code)
            file_relative_imports = []
            for node in ast.walk(tree):
                # 检查 ast.ImportFrom 节点，其 level 属性 > 0 表示是相对导入
                if isinstance(node, ast.ImportFrom) and node.level is not None and node.level > 0:
                    file_relative_imports.append(
                        f"  - L{node.lineno}: from {'.' * node.level}{node.module or ''} import ..."
                    )
            
            if file_relative_imports:
                non_absolute_imports_found[str(file_path)] = file_relative_imports
        except Exception as e:
            pytest.fail(f"解析文件 {file_path} 以检查导入时失败: {e}")

    if non_absolute_imports_found:
        error_message = "发现相对导入，请改为绝对导入 (架构规则 16):\n"
        for f_path, imports in sorted(non_absolute_imports_found.items()): # 排序方便查看
            error_message += f"文件: {f_path}\n"
            error_message += "\n".join(imports) + "\n"
        pytest.fail(error_message)

def test_orm_isolation():
    """
    why: 确保 ORM 相关代码（如 SQLAlchemy Base/Session/模型）仅限于存储层，防止业务/核心层耦合数据库实现。
    what: 检查 kd_tool 除 storage 目录外的所有 .py 文件，是否导入 sqlalchemy 或自定义 ORM model。
    how: 用 AST 检查 import/from-import 语句，禁止 'sqlalchemy'、'Base'、'Session'、'models_sqlalchemy' 等关键字。
    """
    import kd_tool
    from pathlib import Path
    import ast

    root = Path(kd_tool.__file__).parent
    for py_file in root.rglob("*.py"):
        # 跳过存储层
        if "storage" in py_file.parts:
            continue
        # 跳过 __init__.py
        if py_file.name == "__init__.py":
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("sqlalchemy") or "models_sqlalchemy" in alias.name:
                        raise AssertionError(f"{py_file} 不应导入 ORM 相关模块: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module.startswith("sqlalchemy") or "models_sqlalchemy" in node.module):
                    raise AssertionError(f"{py_file} 不应 from-import ORM 相关模块: {node.module}")
                for alias in node.names:
                    if alias.name in ("Base", "Session"):
                        raise AssertionError(f"{py_file} 不应直接导入 ORM 基类/Session: {alias.name}")

def iter_py_files(root_dir: Path):
    """
    why: 递归获取指定目录下所有 .py 文件路径。
    what: 用于后续自动发现所有 DTO/配置模型定义。
    how: 递归遍历。
    """
    for path in root_dir.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        yield path


def import_module_from_path(module_path: Path):
    """
    why: 动态导入指定路径的 Python 模块。
    what: 便于后续反射获取 BaseModel 子类。
    how: 使用 importlib.util。
    """
    module_name = module_path.stem + "_archcheck"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def get_all_pydantic_models(module: types.ModuleType):
    """
    why: 获取模块内所有 Pydantic BaseModel 子类。
    what: 用于后续配置检查。
    how: 反射+issubclass。
    """
    models = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, pydantic.BaseModel) and obj is not pydantic.BaseModel:
            models.append((name, obj))
    return models


def test_pydantic_dto_config_conformance():
    """
    why: 保证所有 Pydantic DTO/配置模型 extra='forbid'、validate_assignment=True、frozen=False。
    what: 自动发现并检查所有 BaseModel 子类。
    how: 动态导入+反射+断言。
    """
    root_dir = Path(__file__).parent.parent.parent / "kd_tool"
    errors = []
    for py_file in iter_py_files(root_dir):
        module = import_module_from_path(py_file)
        if module is None:
            continue  # 跳过无法导入的模块
        for class_name, model_cls in get_all_pydantic_models(module):
            # 获取 model_config 或 Config
            config = getattr(model_cls, "model_config", None)
            if config is None and hasattr(model_cls, "Config"):
                config = getattr(model_cls.Config, "__dict__", {})
            # 检查 extra
            extra = getattr(config, "extra", config.get("extra") if config else None)
            if extra != "forbid":
                errors.append(f"{py_file}:{class_name} 必须设置 extra='forbid'")
            # 检查 validate_assignment
            validate_assignment = getattr(config, "validate_assignment", config.get("validate_assignment") if config else None)
            if validate_assignment is not True:
                errors.append(f"{py_file}:{class_name} 必须设置 validate_assignment=True")
            # 检查 frozen
            frozen = getattr(config, "frozen", config.get("frozen") if config else None)
            if frozen is True:
                errors.append(f"{py_file}:{class_name} 不允许设置 frozen=True，必须可变")
    if errors:
        pytest.fail("\n".join(errors))