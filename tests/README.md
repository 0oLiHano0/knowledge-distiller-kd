# 测试说明文档

本文件提供关于如何设置、运行和理解 `knowledge-distiller-kd` 项目测试套件的说明。

## 测试目录结构 (主要文件)


tests/
├── init.py             # 使 tests 成为一个包
├── conftest.py             # pytest 配置文件和 fixtures
├── requirements-test.txt   # (已废弃，请使用根目录的 requirements-dev.txt)
├── test_constants.py       # 测试常量定义
├── test_engine.py          # 测试核心引擎 (KnowledgeDistillerEngine)
├── test_file_storage.py    # 测试文件存储 (FileStorage)
├── test_md5_analyzer.py    # 测试 MD5 分析器
├── test_semantic_analyzer.py # 测试语义分析器
├── test_document_processor.py # 测试文档处理器
├── test_block_merger.py    # 测试代码块合并器
├── test_utils.py           # 测试通用工具函数
├── test_integration.py     # 集成测试，测试多模块协同工作
├── ui/                     # UI 相关测试
│   ├── init.py
│   └── test_cli_interface.py # 测试命令行界面交互
├── test_data_generator.py  # (若存在) 用于生成测试数据的脚本
├── run_tests.py            # (可选) 运行测试的辅助脚本
└── README.md               # 本测试说明文档

*(注意：旧的测试文件如 `test_core_functions.py` 可能已不存在或需要清理)*

## 测试环境设置

1.  **确保项目已按根目录 `README.md` 的指南安装。**

2.  **安装开发与测试依赖**:
    在项目根目录下，运行以下命令安装所有必要的开发和测试库（包括 `pytest`）：
    ```bash
    pip install -r requirements-dev.txt
    ```
    *(不再需要单独使用 `tests/requirements-test.txt`)*

3.  **Python 路径**:
    通常，如果您在项目根目录下运行 `pytest`，它会自动找到 `knowledge_distiller_kd` 包。如果遇到导入错误，您可能需要确保项目根目录在您的 `PYTHONPATH` 环境变量中，但这通常不是必需的。

## 运行测试

建议使用 `pytest` 命令直接运行测试。在项目**根目录**下执行：

1.  **运行所有测试**:
    ```bash
    pytest
    ```
    或者，为了更详细的输出：
    ```bash
    pytest -v
    ```

2.  **运行特定测试文件**:
    ```bash
    pytest tests/test_engine.py -v
    ```

3.  **运行特定测试类或函数**:
    使用 `::` 分隔符。例如，运行 `TestEngine` 类中的 `test_run_analysis` 方法：
    ```bash
    pytest tests/test_engine.py::TestEngine::test_run_analysis -v
    ```

4.  **使用标记 (Markers) 运行特定类型的测试 (如果已定义)**:
    例如，如果某些测试被标记为 `@pytest.mark.slow`：
    ```bash
    pytest -m "not slow" # 运行所有非慢速测试
    ```

## 测试报告

-   **控制台输出**: `pytest` 会在控制台直接显示测试结果（通过、失败、跳过、错误）。使用 `-v` (verbose) 或 `-vv` 可以获取更详细的输出。
-   **覆盖率报告 (如果配置)**: 如果配置了 `pytest-cov` 插件，可以生成代码覆盖率报告。通常通过以下命令运行：
    ```bash
    pytest --cov=knowledge_distiller_kd --cov-report=html
    ```
    这将在项目根目录下生成一个 `htmlcov/` 目录，包含详细的 HTML 覆盖率报告。

## 测试内容

测试套件旨在覆盖项目的主要功能和模块：

-   **核心引擎 (`test_engine.py`)**: 测试引擎的初始化、状态管理、分析流程编排。
-   **分析器 (`test_md5_analyzer.py`, `test_semantic_analyzer.py`)**: 测试 MD5 哈希计算、重复检测、语义模型加载、相似度计算。
-   **处理器 (`test_document_processor.py`, `test_block_merger.py`)**: 测试文档解析、`ContentBlock` 生成、代码块合并逻辑。
-   **存储 (`test_file_storage.py`)**: 测试决策文件的加载和保存。
-   **UI (`tests/ui/test_cli_interface.py`)**: 测试命令行界面的菜单显示、用户输入处理和与引擎的交互。
-   **工具与常量 (`test_utils.py`, `test_constants.py`)**: 测试辅助函数和常量定义。
-   **集成测试 (`test_integration.py`)**: 测试从文档输入到生成输出的端到端流程，确保各模块正确协同工作。

## 测试数据

-   测试通常使用 `conftest.py` 中定义的 fixtures 来创建模拟数据、临时文件或目录。
-   部分测试可能依赖于 `tests/fixtures/` 或类似目录下的静态测试文件。
-   如果使用了 `test_data_generator.py`，它负责生成更复杂的测试数据集。

## 注意事项

-   测试应设计为可重复执行，并在执行后清理其创建的临时文件或状态。
-   测试日志通常会输出到控制台，或根据日志配置写入特定文件（例如 `logs/test.log`）。
-   确保运行测试的环境具有执行所有测试所需的资源（内存、CPU）。