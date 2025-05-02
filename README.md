# Knowledge Distiller (KD)

知识蒸馏工具 - 一个用于检测和处理文档重复内容的智能工具。

## 项目概述

Knowledge Distiller (KD) 是一个强大的文档内容分析工具，专门用于检测和处理文档中的重复内容。它使用先进的语义分析技术，不仅能够识别完全相同的内容，还能发现语义相似的内容片段。

### 主要特性

- 精确重复检测：使用 MD5 哈希算法检测完全相同的内容
- 语义相似度分析：使用 Sentence Transformers 进行语义相似度计算
- 交互式决策：提供友好的命令行界面进行内容处理
- 批量处理：支持批量处理多个文档
- 缓存机制：使用向量缓存提高性能
- 多语言支持：支持中文和英文文档

## 安装指南

### 系统要求

- Python 3.8+
- pip 包管理器
- （推荐）libmagic 库 (用于更准确的文件类型检测，安装方式见下文)

### 安装步骤

1.  克隆仓库：
    ```bash
    git clone https://github.com/yourusername/knowledge-distiller-kd.git
    cd knowledge-distiller-kd
    ```

2.  创建并激活虚拟环境：
    ```bash
    # 确保使用 Python 3.8+
    python3 -m venv venv
    source venv/bin/activate  # Linux/Mac
    # 或
    # .\venv\Scripts\activate  # Windows
    ```

3.  （可选但推荐）更新虚拟环境中的 pip：
    ```bash
    pip install --upgrade pip
    ```

4.  安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

5.  安装开发依赖（可选）：
    ```bash
    pip install -r requirements-dev.txt
    ```

6.  **安装 libmagic (推荐):**
    * **macOS:** `brew install libmagic`
    * **Debian/Ubuntu:** `sudo apt-get update && sudo apt-get install libmagic1`
    * **Windows:** 较复杂，可能需要下载预编译库或通过 conda 等方式安装。`unstructured` 文档可能有更详细说明。

## 使用方法

### 基本用法

1.  准备输入文件：
    * 将需要处理的 Markdown 文件放入 `input` 目录

2.  运行工具 (确保虚拟环境已激活):
    ```bash
    # 进入交互模式
    python3 -m knowledge_distiller_kd.core.kd_tool_CLI
    # 或直接指定输入目录运行分析
    python3 -m knowledge_distiller_kd.core.kd_tool_CLI -i input
    ```

3.  交互式处理：
    * 工具会显示主菜单或分析结果。
    * 根据菜单提示进行操作，例如设置目录、运行分析、查看重复、保存/应用决策等。
    * 在查看重复项时，使用提示的命令（如 `k`, `d`, `k1d2`等）进行决策。

### 高级配置

可以通过修改以下参数自定义工具行为（可通过交互菜单或命令行参数）：

- `similarity_threshold`: 语义相似度阈值（默认：0.8）
- `model_name`: 使用的语义模型（默认：paraphrase-multilingual-MiniLM-L12-v2）
- `decision_file`: 决策文件路径
- `output_dir`: 输出目录路径
- `skip_semantic`: 是否跳过语义分析

## 开发指南

### 项目结构
knowledge-distiller-kd/
├── input/                # 输入文件目录
├── output/               # 输出文件目录
├── decisions/            # 决策文件目录
├── logs/                 # 日志文件目录
├── knowledge_distiller_kd/ # 主要代码包
│   ├── core/             # 核心逻辑模块
│   │   ├── init.py
│   │   ├── constants.py
│   │   ├── document_processor.py # 文档处理 (使用unstructured)
│   │   ├── error_handler.py
│   │   ├── kd_tool_CLI.py     # 主控制与命令行接口
│   │   ├── md5_analyzer.py
│   │   ├── semantic_analyzer.py
│   │   └── utils.py
│   └── init.py
├── tests/                # 测试代码目录
│   ├── ...
├── docs/                 # 文档目录
│   └── ...
├── .gitignore
├── https://www.google.com/search?q=LICENSE
├── README.md
├── requirements-dev.txt  # 开发依赖
├── requirements.txt      # 运行依赖
├── setup.py              # 打包配置

### 开发进度

[x] 基础框架搭建
[x] MD5 精确重复检测
[x] 语义分析器实现
[x] 测试用例编写
[x] 文档完善 (持续)
[x] 集成 unstructured 进行文档处理 &lt;--- 新增标记
[ ] 性能优化 (特别是大规模文件处理)
[ ] 合并代码块碎片 (改进报告和预览) &lt;--- 下一步重要优化
[ ] Web 界面开发
[ ] API 接口开发
[ ] 支持更多文档格式 (需结合合并或重构 document_processor)

## 当前功能
- Markdown文件重复内容检测 (基于MD5和语义相似度)。
- 使用 unstructured 进行文档元素分割。
- 交互式命令行界面，用于审查重复项、管理决策。
- 决策的保存和加载。
- 基于决策生成去重后的输出文件。

## 下一步开发目标
- 合并代码块碎片
   目前 unstructured 会将代码块分割成多个元素（起始、内容、结束）。为了更准确地进行分析（特别是语义分析）和提供更完整的预览，需要实现一个后处理步骤，在分析前将这些碎片合并成代表整个代码块的单个 ContentBlock。

- 支持更多文档格式
   在解决了碎片问题或有了更健壮的 document_processor 之后，扩展支持 Word, PDF 等格式。需要研究 unstructured 对这些格式的处理能力以及如何最好地集成。