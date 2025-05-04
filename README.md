# Knowledge Distiller (KD)
知识蒸馏工具 - 一个用于检测和处理文档重复内容的智能工具。

## 项目概述
Knowledge Distiller (KD) 是一个强大的文档内容分析工具，专门用于检测和处理文档中的重复内容。它采用分层架构设计，使用先进的语义分析技术，不仅能够识别完全相同的内容，还能发现语义相似的内容片段。

## 主要特性

- **精确重复检测**： 使用 MD5 哈希算法检测完全相同的内容块（默认跳过标题块）。
- **语义相似度分析**： 使用 Sentence Transformers 进行语义相似度计算（默认跳过标题块）。
- **文档解析**： 使用 `unstructured` 库解析多种文档格式（当前主要测试 Markdown），提取内容块。
- **代码块合并**： 自动合并 Markdown 代码块的碎片（起始围栏、内容、结束围栏），以便进行更准确的分析和预览。
- **交互式决策**： 提供友好的命令行界面 (`CLI`) 进行内容处理决策。
- **持久化决策**： 将用户决策保存到 JSON 文件中。
- **去重输出**： 根据用户决策生成去重后的 Markdown 文件。
- **模块化设计**： 清晰的分层架构 (UI, Core Engine, Analysis, Processing, Storage)，易于维护和扩展。

## 安装指南

### 系统要求

- Python 3.8+
- pip 包管理器
- （推荐）libmagic 库 (用于 `unstructured` 更准确的文件类型检测，安装方式见下文)

### 安装步骤

1.  **克隆仓库**：
    ```bash
    git clone [https://github.com/yourusername/knowledge-distiller-kd.git](https://github.com/yourusername/knowledge-distiller-kd.git) # 替换为你的仓库地址
    cd knowledge-distiller-kd
    ```

2.  **创建并激活虚拟环境**：
    ```bash
    # 确保使用 Python 3.8+
    python3 -m venv venv
    source venv/bin/activate  # Linux/Mac
    # 或
    # .\venv\Scripts\activate  # Windows
    ```

3.  **更新 pip (推荐)**：
    ```bash
    pip install --upgrade pip
    ```

4.  **安装核心依赖**：
    ```bash
    pip install -r requirements.txt
    ```

5.  **安装开发与测试依赖 (若需开发或运行测试)**：
    ```bash
    pip install -r requirements-dev.txt
    ```

6.  **安装 libmagic (推荐)**:
    - macOS: `brew install libmagic`
    - Debian/Ubuntu: `sudo apt-get update && sudo apt-get install libmagic1`
    - Windows: 较复杂，可能需要下载预编译库或通过 conda 等方式安装。请参考 `unstructured` 官方文档获取最新指南。

## 使用方法

### 基本用法

1.  **准备输入文件**：
    将需要处理的文档放入 `input/` 目录（或您指定的其他目录）。

2.  **运行工具 (确保虚拟环境已激活)**:
    ```bash
    # 进入交互模式 (推荐)
    python -m knowledge_distiller_kd.cli

    # 或直接指定输入目录运行分析 (非交互，自动应用已有决策或默认行为)
    # python -m knowledge_distiller_kd.cli -i input/
    ```
    *(注意：直接指定输入目录运行的非交互模式可能需要进一步完善)*

3.  **交互式处理**：
    - 工具启动后会显示主菜单。
    - 根据菜单提示进行操作，例如：
        - 设置输入/输出/决策目录。
        - 运行分析 (`Run Analysis`)。
        - 查看 MD5 重复项 (`Review MD5 Duplicates`)。
        - 查看语义相似项 (`Review Semantic Duplicates`)。
        - 保存决策 (`Save Decisions`)。
        - 应用决策生成去重文件 (`Apply Decisions`)。
    - 在查看重复/相似项时，使用提示的命令（如 `k`, `d`, `k1d2` 等）进行决策。

### 高级配置

可以通过修改以下参数自定义工具行为（可通过交互菜单或未来可能的配置文件）：

- `similarity_threshold`: 语义相似度阈值（默认：0.8）
- `model_name`: 使用的语义模型（默认：`paraphrase-multilingual-MiniLM-L12-v2`）
- `decision_file`: 决策文件路径 (默认: `decisions/decisions.json`)
- `output_dir`: 输出目录路径 (默认: `output/`)
- `input_dir`: 输入目录路径 (默认: `input/`)
- `skip_semantic`: 是否跳过语义分析

## 开发指南

### 项目结构


knowledge-distiller-kd/
├── input/                  # 输入文件目录
├── output/                 # 输出文件目录
├── decisions/              # 决策文件目录
├── logs/                   # 日志文件目录
├── knowledge_distiller_kd/ # 项目核心代码包
│   ├── init.py
│   ├── cli.py              # 应用入口点和参数解析
│   ├── core/               # 核心逻辑与组件
│   │   ├── init.py
│   │   ├── engine.py       # 核心引擎 (KnowledgeDistillerEngine)
│   │   ├── constants.py
│   │   ├── error_handler.py
│   │   └── utils.py
│   ├── analysis/           # 去重与分析逻辑
│   │   ├── init.py
│   │   ├── md5_analyzer.py
│   │   └── semantic_analyzer.py
│   ├── processing/         # 文档预处理
│   │   ├── init.py
│   │   ├── document_processor.py
│   │   └── block_merger.py
│   ├── storage/            # 数据持久化
│   │   ├── init.py
│   │   └── file_storage.py # JSON 文件存储
│   └── ui/                 # 用户界面
│       ├── init.py
│       └── cli_interface.py # 命令行界面实现
├── tests/                  # 测试代码目录 (详见 tests/README.md)
│   ├── init.py
│   ├── conftest.py
│   ├── test_*.py           # 各模块单元测试
│   ├── test_integration.py # 集成测试
│   └── ui/
│       └── test_cli_interface.py
├── docs/                   # 项目文档目录
│   └── ...
├── .gitignore
├── LICENSE                 # <--- 建议添加许可证文件
├── README.md               # 本文件
├── requirements-dev.txt    # 开发与测试依赖
├── requirements.txt        # 核心运行依赖
└── setup.py                # 打包与安装配置


### 开发进度与下一步计划

详细内容请参考最新的项目状态文档。主要已完成和计划中的任务包括：

**已完成 (截至 2025-05-03):**

- [x] **核心重构完成:** 实现分层架构 (UI, Core, Analysis, Processing, Storage)。
- [x] **核心流程可运行:** 文档处理 -> 代码块合并 -> MD5分析 -> 语义分析 -> 决策 -> 输出。
- [x] **`unstructured` 集成:** 稳定解析文档为 `ContentBlock`。
- [x] **代码块合并:** 解决代码块碎片化问题。
- [x] **分析优化:** 跳过标题块分析。
- [x] **CLI 界面:** 实现交互式命令行操作。
- [x] **JSON 决策存储:** 实现决策加载与保存。
- [x] **测试覆盖:** 为主要模块添加单元测试和集成测试。

**短期任务 (Immediate Focus):**

- [ ] **代码清理与完善:** 移除冗余文件、清理旧测试、补充文档字符串、规范化路径处理。
- [ ] **测试增强:** 提高测试覆盖率，增加边界条件测试。
- [ ] **配置管理优化:** 考虑使用更健壮的配置方式。

**中期任务 (Core Functionality & Infrastructure):**

- [ ] **引入数据库存储:** 设计并实现数据库存储层 (如 SQLite)。
- [ ] **扩展文件格式支持:** 明确支持并测试 PDF、DOCX 等格式。

**长期任务 (Feature Enhancement & Advanced Capabilities):**

- [ ] **实现辅助功能:** 标签、分类、搜索、版本控制。
- [ ] **探索 LLM 集成:** 辅助去重、冲突检测等。
- [ ] **开发 GUI:** 图形用户界面。
- [ ] **性能优化:** 持续关注和优化。

### 测试

本项目使用 `pytest` 进行单元测试和集成测试。关于如何设置测试环境和运行测试的详细说明，请参阅 [`tests/README.md`](tests/README.md) 文件。