# Knowledge Distiller (KD)
知识蒸馏工具 - 一个用于检测和处理文档重复内容的智能工具。

## 项目概述
Knowledge Distiller (KD) 是一个强大的文档内容分析工具，采用分层架构设计(UI, Core, Analysis, Processing, Storage)，使用先进的语义分析技术，能够：
- 通过Czkawka进行文件级预过滤（待嵌入流程）
- 实现MD5 → SimHash（开发中） → SBERT三层内容去重
- 支持SQLite作为核心数据存储（开发中）
- 提供灵活的命令行交互界面
- 提供UI界面(开发中)

## 分层架构

Knowledge Distiller (KD) 采用清晰的六层架构设计，各层职责分明，易于维护与扩展：

1. **Prefilter（预过滤层）**  
   - 目录：`knowledge_distiller_kd/prefilter/`  
   - 功能：基于 Czkawka 对整个文档或文件集合进行快速、文件级重复检测，剔除高度重复或无效文件。  
   - 主要模块：  
     - `czkawka_adapter.py`（Czkawka 集成适配器）

2. **Processing（处理层）**  
   - 目录：`knowledge_distiller_kd/processing/`  
   - 功能：从预过滤层获取**不重复**的文档列表进行内容分割与结构化：  
     - `document_processor.py`：其中`process_file`函数调用 `unstructured` 库，将原始文档（Markdown、PDF、Office 等）切分成结构化的原始块列表，包装成ContentBlock对象。
     - `block_merger.py`：`block_merger`模块在`process_file`返回的原始块列表基础上进行合并操作（如将连续的代码块合并），避免信息碎片化带来的语义缺失。输出合并后的块列表`List[ContentBlock]`

3.  **Analysis（分析层）**  
   - 目录：`knowledge_distiller_kd/analysis/`  
   - 功能：针对同一文档的内容块执行多维度相似度评估：  
     - `md5_analyzer.py`：对精准哈希比对  
     - `simhash_analyzer.py`：近似重复识别  
     - `semantic_analyzer.py`：基于 SBERT 的语义相似度计算

4. **Core（核心引擎层）**  
   - 目录：`knowledge_distiller_kd/core/`  
   - 功能：整体流程调度与协调各层逻辑，提供统一的程序接口。  
   - 主要组件：  
     - `engine.py`（`KnowledgeDistillerEngine`）

5. **Storage（存储层）**  
   - 目录：`knowledge_distiller_kd/storage/`  
   - 功能：持久化分析结果与用户决策，支持多种存储后端。  
     - `sqlite_storage.py`：SQLite 数据库存取  
     - `file_storage.py`：JSON 文件读写  

6. **UI（用户交互层）**  
   - 目录：`knowledge_distiller_kd/ui/`  
   - 功能：提供命令行界面，供用户交互式审阅重复/相似项并做出决策。  
   - 主要模块：  
     - `cli_interface.py`

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
