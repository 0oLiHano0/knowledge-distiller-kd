# Knowledge Distiller (KD)
知识蒸馏工具 - 一个用于检测和处理文档重复内容的智能工具。

## 项目概述
Knowledge Distiller (KD) 是一个强大的文档内容分析工具，采用分层架构设计(UI, Core, Analysis, Processing, Storage)，使用先进的语义分析技术，能够：
- 通过Czkawka进行文件级预过滤
- 实现MD5 → SimHash → SBERT三层内容去重
- 支持SQLite作为核心数据存储
- 提供灵活的命令行交互界面

## 主要特性

- **精确重复检测**： 使用 MD5 哈希算法检测完全相同的内容块（默认跳过标题块）。
- **语义相似度分析**： 使用 Sentence Transformers 进行语义相似度计算（默认跳过标题块）。
- **文档解析**： 使用 `unstructured` 库解析多种文档格式（当前主要测试 Markdown），提取内容块。
- **代码块合并**： 自动合并 Markdown 代码块的碎片（起始围栏、内容、结束围栏），以便进行更准确的分析和预览。
- **交互式决策**： 提供友好的命令行界面 (`CLI`) 进行内容处理决策。
- **持久化决策**： 将用户决策保存到 JSON 文件中。
- **去重输出**： 根据用户决策生成去重后的 Markdown 文件。
- **模块化设计**： 清晰的分层架构 (UI, Core Engine, Analysis, Processing, Storage)，易于维护和扩展。
- **文件级预过滤**：集成Czkawka工具检测重复文件
- **三层去重检测**：MD5(精确) → SimHash(近似) → SBERT(语义)
- **SQLite存储**：核心数据持久化方案，支持高效查询
- **决策管理**：统一的决策管理接口

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

### 项目结构 (更新后)


knowledge-distiller-kd/
├── input/ # 输入文件目录
├── output/ # 输出文件目录
├── decisions/ # 决策文件目录 (JSON导出)
├── logs/ # 日志文件目录
├── knowledge_distiller_kd/ # 项目核心代码包
│ ├── core/ # 核心逻辑与组件
│ │ ├── engine.py # 核心引擎 (KnowledgeDistillerEngine)
│ │ └── ...
│ ├── prefilter/ # 新增: 文件预过滤
│ │ ├── czkawka_adapter.py # Czkawka集成
│ ├── analysis/ # 去重与分析逻辑
│ │ ├── md5_analyzer.py
│ │ ├── simhash_analyzer.py # 新增: SimHash分析
│ │ └── semantic_analyzer.py
│ ├── storage/ # 数据持久化
│ │ ├── sqlite_storage.py # 新增: SQLite实现
│ │ └── file_storage.py # JSON文件存储(降级为辅助)
│ └── ui/                 # 用户界面
│    ├── init.py
│    └── cli_interface.py # 命令行界面实现
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


### 开发进度与下一步计划 (更新后)

**待完成 (截至 2025-06-15):**
- [ ] **三层去重架构**: MD5 → SimHash → SBERT
- [ ] **SQLite集成**: 核心数据存储方案
- [ ] **Czkawka预过滤**: 文件级重复检测
- [ ] **GUI开发**: 基于Tkinter的图形界面
- [ ] **性能优化**: 大规模文档处理效率提升

**已完成:**
- [x] **BlockMerger优化**: 提升代码块合并准确性


**长期规划:**
- [ ] **OCR支持**: 扫描文档处理
- [ ] **LLM集成**: 智能冲突解决

### 测试

本项目使用 `pytest` 进行单元测试和集成测试。关于如何设置测试环境和运行测试的详细说明，请参阅 [`tests/README.md`](tests/README.md) 文件。
