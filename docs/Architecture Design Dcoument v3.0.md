# 知识蒸馏工具（KD_Tool）

> **发布日期** 2025‑05‑25 | **维护人**： Hansen |  v3.2

**！！！！注意：本项目正在进行v4.0的重构，此文档仅在必要时参考！！！**

---

## 0. 文档更新记录

| 版本 | 日期 | 更新说明 |
| --- | --- | --- |
| v3.0 | 2025-05-10 | 文档初始版本，描述基础架构设计 |
| v3.1 | 2025-05-20 | 更新Storage层实现详情和数据模型 |
| v3.2 | 2025-05-25 | 更新Phase 1完成情况：使用Pydantic进行配置管理、Dependency-Injector实现依赖注入、SQLite+SQLAlchemy实现存储接口集成和Loguru日志系统 |

## 1. 项目目标
构建一款完全本地化的源信息治理工具，为个人与企业在应用知识库 / RAG 实践之前，完成知识治理，提供高质量信息。

计划实现**文件级去重→文本块级去重（MD5、语义）→知识冲突检测→通用知识过滤**。
**本工具非知识库或AI问答系统**

## 2. 分层架构

Knowledge Distiller (KD) 采用清晰的六层架构设计，各层职责分明，易于维护与扩展：

0. **主程序入口**
  - 主程序目录：`knowledge_distiller_kd/`
  - 主程序：`cli.py`，UI将后期构建
  - 第三方工具：
    - 打包一并分发与安装
    - 目录：knowledge_distiller_kd/vendor/czkawka/macos-arm64/

1. **Prefilter（预过滤层）**  
  - 目录：`knowledge_distiller_kd/prefilter/`  
  - 功能：基于 Czkawka 对整个文档或文件集合进行快速、文件级重复检测，剔除高度重复或无效文件。  
  - 主要模块：  
    - `czkawka_adapter.py`（Czkawka 集成适配器）
      - 已实现功能:
        * 支持多平台（Windows/macOS/Linux）的Czkawka二进制调用 (通过 core.utils.get_bundled_czkawka_path)
        * 自动检测系统架构并加载对应版本 (通过 core.utils.get_bundled_czkawka_path)
        * 支持JSON格式结果解析并转换为DTO (DuplicateFileGroupDTO)
        * 提供文件扫描、重复文件组获取、唯一文件过滤等核心API
        * 内置扫描结果缓存和执行超时机制
  - 输入：
    - `file_path: Path` - 待扫描目录路径
    - `config: PrefilterConfig` - 配置参数（使用dataclass封装）
      * `similarity_threshold: float = 0.9` - 文件相似度阈值
      * `scan_depth: int = 5` - 目录扫描深度
      * `file_types: List[str] = ["pdf", "docx", "md"]` - 支持的文件类型
  - 输出：
    - `unique_files: List[Path]` - 唯一文件列表
    - `duplicate_groups: List[DuplicateFileGroup]` - 重复文件组信息
      * 包含重复文件路径列表和相似度分数
    - `scan_stats: Dict[str, int]` - 扫描统计信息
      * 包含总文件数、重复文件数等
  - 进度：
    - 已完成核心功能开发并通过单元测试
    - 已集成到Engine主流程 (待通过Engine代码确认)
    - 已包含扫描缓存和超时机制，持续关注大规模目录性能。

2. **Processing（处理层）**  
  - 目录：`knowledge_distiller_kd/processing/`  
  - 功能：从预过滤层获取**不重复**的文档列表进行结构化分割和智能重组，为后续分析层准备高质量内容块
  - 主要模块： 
     - `document_processor.py`：核心处理模块，包含：
       * `process_file()` - 主要使用 `unstructured` 库解析 Markdown 文档，其他格式支持依赖 `unstructured` 的能力但需在 `process_file` 中扩展调用逻辑。
       * `ContentBlock` 类 - 在 `document_processor.py` 中定义，用于封装 `unstructured` 的元素，并进行类型推断和文本规范化。注意其与 `core.models.ContentBlock` (即 `BlockDTO`) 的区别和联系。
     - `block_merger.py`：智能合并模块，实现：
       * 代码块合并（基于行数阈值）
       * 段落合并（基于语义连续性）
       * 表格/图片保持独立（不合并）
  - 输入：
    - `process_file()` 参数：
      * `file_path: Path` - 必选，待处理文件路径
      * `config: ProcessingConfig` - 可选，处理配置（使用dataclass封装）
        - `code_block_threshold: int = 10` - 代码块合并最小行数
        - `merge_paragraphs: bool = True` - 是否启用段落合并
        - `languages: List[str] = ["zh", "en"]` - 支持的语言列表
    - `merge_blocks()` 参数：
      * `blocks: List[ContentBlock]` - 必选，原始内容块列表
      * `rules: MergeRules` - 可选，合并规则配置
        - `max_merged_size: int = 2000` - 合并后最大字符数
        - `preserve_formats: List[str] = ["table", "image"]` - 保持原样的格式
  - 输出：
    - `process_file()` 返回：
      * `processed_blocks: List[ContentBlock]` - 结构化内容块
      * `metadata: ProcessingMetadata` - 处理统计信息
        - `file_size: int` - 文件大小(bytes)
        - `processing_time: float` - 处理耗时(秒)
        - `block_counts: Dict[str, int]` - 各类型块数量
    - `merge_blocks()` 返回：
      * `merged_blocks: List[ContentBlock]` - 合并后内容块
      * `merge_logs: List[MergeLog]` - 合并操作记录
        - 包含操作类型、涉及块ID、时间戳等
  - 进度：
    - 已实现基于 `unstructured` 的文档解析框架，当前 `process_file` 函数明确支持 Markdown 文件，其他格式支持有待在该函数中扩展。
    - 完成 Markdown 代码块的合并逻辑；段落合并及表格/图片等其他类型内容的合并/处理逻辑待实现。
    - 通过200+测试用例验证
  - 待优化：
    - 大文件处理性能优化（>50MB）
    - 复杂表格的解析精度提升

3.  **Analysis（分析层）**  
  - 目录：`knowledge_distiller_kd/analysis/`  
  - 功能：对处理后的内容块进行多级相似度分析（精确匹配→近似匹配→语义匹配）
  - 主要模块：
     - `md5_analyzer.py`：精确内容匹配（基于MD5哈希）
       * 已完成：基于MD5的精确重复块检测，支持跳过特定类型和已决策的块。
       * 待关注：大规模数据下的哈希计算性能。
     - `simhash_analyzer.py`：内容相似度检测（基于SimHash）
       * 文件状态：待开始。
       * 进度：待创建和实现。
     - `semantic_analyzer.py`：深度语义分析（基于SBERT模型）
       * 已完成：通过SentenceTransformers集成语义模型，支持向量缓存。
       * 待优化：GPU加速支持。代码中ContentBlock DTO的统一性待检查。
  - 输入规范：
    - 统一输入：`List[ContentBlock]`（来自processing层）
    - 配置来源：
      * 硬编码默认值：`core/constants.py`
      * 运行时配置：用户输入/配置文件
  - 输出规范：
    - 统一输出格式：`AnalysisResult` 数据类（定义在`core/models.py`）
      * 包含：匹配块对、相似度分数、分析元数据
    - 错误处理：统一使用`AnalysisError`异常
  - 处理流程：
    1. MD5精确匹配（快速过滤完全相同的块）
    2. SimHash近似匹配（检测内容相似的块）
    3. SBERT语义匹配（最终相似度确认）
  - 性能指标：
    * 平均处理速度：≥1000块/秒（CPU）
    * 内存占用：≤2GB（处理10万内容块时）
  - 进度：
    * 已完成：MD5分析器、语义分析器基础版
    * 进行中：SimHash分析器开发（预计3天）
    * 待规划：多进程加速、结果缓存机制

4. **Core（核心引擎层）**  
  - 目录：`knowledge_distiller_kd/core/`  
  - 功能：作为系统中枢，负责协调预处理、分析和存储各层工作流，提供统一API接口。  
  - 主要模块：  
     - `engine.py` - 主引擎类`KnowledgeDistillerEngine`，实现以下核心方法：
       * `run_pipeline()` - 执行完整处理流水线
       * `validate_inputs()` - 输入参数校验
       * `generate_report()` - 生成分析报告
       * **依赖注入** - 引擎通过构造函数注入 `storage`, `config` 和 `logger`，实现解耦
       * **存储接口集成** - 引擎所有数据持久化操作均通过`self.storage`接口完成，不直接执行文件IO或数据库操作
       * **重要方法**：
         - `save_results()` - 将分析结果和决策保存到存储层
         - `load_decisions()` - 从存储层加载用户决策
         - `save_decisions()` - 将用户决策保存到存储层
         - `apply_decisions()` - 应用决策生成输出内容
         - `run_analysis()` - 运行完整分析流程
     - `config.py` - 配置管理:
       * 基于Pydantic的分层配置体系：`StorageConfig`, `LoggingConfig`, `EngineConfig`, `AppConfig`
       * 支持从环境变量、.env文件和默认值加载配置
       * 提供类型安全和验证机制
     - `factories.py` - 依赖管理工厂:
       * `create_app_config()` - 创建应用配置实例
       * `create_storage()` - 使用配置创建存储实例
       * `create_logger()` - 创建配置好的Loguru日志器实例
         - 基于`LoggingConfig`配置日志级别、文件路径等
         - 移除默认处理器并添加文件、控制台处理器
         - 文件处理器支持JSON格式、日志轮转和保留策略
         - 确保日志目录存在，便于部署
       * `create_engine()` - 创建引擎实例，注入配置、存储和日志器依赖
     - `constants.py` - 集中管理：
       * 默认配置参数（相似度阈值=0.85）
       * 模型路径常量
       * 性能调优参数
     - `error_handler.py` - 提供：
       * 异常统一封装（`KDCoreError`）
       * 错误代码体系（`ERROR_CODE`枚举）
       * 错误日志标准化
     - `models.py` - 定义：
       * `ContentBlock` (@dataclass): 内容块的详细数据模型，包含丰富元数据，用于引擎内部处理。
       * `BlockDTO` (Pydantic): 内容块的轻量级数据传输对象，用于模块间（如存储、合并）交互。
         * (注: `processing.document_processor` 中存在旧版`ContentBlock`定义，项目正逐步统一至`core.models`)
       * `AnalysisResult` (@dataclass): 分析结果容器。
       * `UserDecision` (@dataclass): 用户决策容器。
       * `FileRecord` (@dataclass): 文件记录DTO。
       * `EngineConfig`: (待创建/未在core.models中找到；引擎当前通过构造函数参数接收配置)。
       * (注: `core/models.py` 中也包含 SQLAlchemy ORM 实体的重复定义，建议整合到存储层以避免重复并保持DTO文件的纯粹性)。
     - `utils.py` - 工具集：
       * 性能监控装饰器
       * 类型检查工具
       * 路径处理辅助方法
  - 引擎与存储层集成：
     - **完成状态**: ✅ 已完成 (KD-INTEGRATE-001)
     - **实现方式**: 
       * 引擎通过依赖注入接收`StorageInterface`实例
       * 所有数据持久化操作通过接口方法调用完成
       * 不再直接执行文件IO或数据库操作
     - **主要接口调用**:
       * `storage.register_file()` - 注册文件记录
       * `storage.save_blocks()` - 保存内容块
       * `storage.get_blocks_for_analysis()` - 获取用于分析的内容块
       * `storage.get_block()` - 获取单个内容块
       * `storage.save_analysis_result()` - 保存分析结果
       * `storage.save_user_decision()` - 保存用户决策
       * `storage.list_files()` - 获取所有文件记录
       * `storage.get_blocks_by_file()` - 获取特定文件的所有内容块
     - **数据传输**: 
       * 统一使用`core.models`中定义的DTO对象进行数据传输
       * 解决了`processing.ContentBlock`与`core.models.ContentBlock`的不一致问题
  - 存储生命周期管理和错误处理：
     - **完成状态**: ✅ 已完成 (KD-ARCH-003)
     - **生命周期管理**:
       * 应用启动时：在`factories.py`的`create_storage()`中调用`storage.initialize()`
       * 应用退出时：通过`atexit`注册`_cleanup_storage()`函数调用`storage.finalize()`
       * 全局存储实例引用：使用`_storage_instance`保存引用以便清理
     - **错误处理机制**:
       * 新增`KDStorageError`类：封装存储层错误，继承自`KDError`基类
       * 工厂中的错误处理：捕获`SQLAlchemyError`等存储异常，转换为`KDStorageError`
       * 引擎中的错误处理：
         - 所有对`self.storage.*`的调用均使用`try-except`块保护
         - 捕获`SQLAlchemyError`等数据库异常
         - 使用`self.logger.exception()`记录详细错误和堆栈
         - 适当处理错误：静默处理、返回空值或False、或封装重抛
     - **测试覆盖**:
       * 单元测试：完整测试生命周期管理和错误处理逻辑
       * 测试文件：`tests/core/test_storage_lifecycle_errors.py`
       * 测试范围：
         - 工厂函数中initialize/finalize的调用
         - 各种存储错误场景的正确处理
  - 输入规范：
    - 文件输入：支持`str/Path/List[Path]`多种格式
    - 配置输入：优先使用用户配置，回退到`constants.py`默认值
    - 预处理输入：必须经过`prefilter`层过滤
  - 输出规范：
    - 分析结果：统一为`AnalysisResult`数据类，包含：
      * 匹配块对列表
      * 相似度分数矩阵
      * 分析元数据（时间戳、配置参数等）
    - 报告输出：
      * JSON格式：用于调试和结果导出
      * SQLite格式：用于持久化存储和查询
      * 支持增量更新和批量导入
    - 日志输出：
      * 采用结构化JSON格式
      * 包含时间戳、日志级别、模块名
      * 支持错误追踪和性能监控
  - 处理流程：
    1. 初始化引擎配置
    2. 执行预处理流水线
    3. 运行多级分析
    4. 生成最终报告
  - 性能要求：
    * 单次分析内存峰值 ≤1.5GB
    * 10万内容块处理时间 ≤30秒
  - 进度：
    * 已完成：引擎基础框架、配置管理系统
    * 已完成：SQLite存储集成（已实现基础CRUD操作）
    * 已完成：引擎与存储层完全解耦（通过存储接口）
    * 待优化：异步任务调度机制
  - 日志系统（Loguru集成）：
     - **完成状态**: ✅ 已完成 (KD-LOGGING-001, KD-LOGGING-002, KD-LOGGING-003)
     - **实现方式**:
       * 在`factories.py`中实现`create_logger()`函数，基于AppConfig配置日志系统
       * 日志配置通过`LoggingConfig`管理，包含级别、文件路径、轮转策略等
       * 使用`logger.remove()`和`logger.add()`配置Loguru的处理器
     - **主要特性**:
       * 文件日志：JSON格式、支持轮转和保留策略
       * 控制台日志：美化格式、便于开发调试
       * 自动创建日志目录，增强部署便利性
       * 完全通过配置控制日志行为，无需代码修改
     - **集成点**:
       * 引擎通过构造函数接收logger实例
       * 所有模块统一使用Loguru记录日志
     - **旧日志代码移除** (KD-LOGGING-003):
       * 移除了所有对Python标准库`logging`模块的配置代码:
         - 删除了`logging.basicConfig`调用和相关配置
         - 移除了`logging.getLogger()`、`addHandler()`和`setLevel()`等代码
         - 清理了创建`FileHandler`和`StreamHandler`的冗余代码
       * 替换所有`import logging`为`from loguru import logger`
       * 完善了单元测试，增强了测试覆盖率:
         - 添加了`test_logger_actual_output`测试实际日志输出
         - 添加了`test_logger_structured_json_output`测试JSON格式日志
         - 添加了`test_logger_exception_capturing`验证异常捕获功能
         - 添加了`test_logger_intercept_filter`测试日志过滤功能
         - 添加了`test_logger_custom_sink_and_format`测试自定义格式
       * 保证了迁移的完整性，所有代码和测试都已完全使用Loguru

5. **Storage（存储层）**  
  - 目录：`knowledge_distiller_kd/storage/`  
  - 功能：持久化分析结果与用户决策，支持多种存储后端。 
  - 主要模块： 
     - `storage_interface.py`：定义存储层抽象接口，提供统一的存储操作方法。Core引擎通过此接口与存储层交互，实现与具体存储实现的解耦。包含初始化、文件注册、块存储、分析结果管理和用户决策处理等完整操作集。
     - `orm_storage.py`：实现基于SQLAlchemy的ORM模型存储，是`StorageInterface`的主要具体实现。处理数据库连接、事务管理、异常处理和日志记录，确保数据持久化安全可靠。
     - `sqlite_storage.py`：实现SQLite数据库连接和会话管理功能，包含数据库URL生成和目录创建  
     - `models_sqlalchemy.py`：定义所有数据库表的ORM模型，包括Document、Block、Analysis和Decision 
     - `file_storage.py`：提供JSON文件读写功能，是`StorageInterface`的另一种实现，主要用于调试和结果导出  
  - 输入：
    - `orm_storage.py` 输入:
      * `file_metadata: Dict[str, Any]` - 文件元数据，来自:
        - `prefilter/czkawka_adapter.py` 的扫描结果
      * `content_blocks: List[ContentBlock]` - 内容块，来自:
        - `processing/document_processor.py` 的处理结果
      * `analysis_results: List[AnalysisResult]` - 分析结果，来自:
        - 各分析器（MD5/SimHash/Semantic）的分析结果
      * `user_decisions: List[UserDecision]` - 用户决策，来自:
        - UI层收集的用户决策

  - 输出：
    - `orm_storage.py` 输出:
      * `content_blocks: List[ContentBlock]` - 获取的内容块，返回给:
        - `core/engine.py` 的分析流程
      * `analysis_results: List[AnalysisResult]` - 查询的分析结果，返回给:
        - `ui` 层的结果展示
      * `file_metadata: List[Dict]` - 文件元数据，返回给:
        - `core/engine.py` 的文件处理流程
    - `file_storage.py` 输出:
      * `export_status: bool` - 导出状态，返回给:
        - `core/engine.py` 的导出日志
  - 进度：存储层已基本完成，`storage_interface.py` 已定义并实现，`orm_storage.py` 实现了完整的存储接口，`sqlite_storage.py` 提供了数据库会话管理，`models_sqlalchemy.py` 定义了完整的数据模型。`file_storage.py` 仅保留用于调试和导出功能。所有相关测试已通过。

6. **UI（用户交互层）**  
  - 目录：`knowledge_distiller_kd/ui/`  
  - 功能：提供命令行界面，供用户交互式审阅重复/相似项并做出决策。  
  - 主要模块：  
     - `cli_interface.py`: 实现交互式命令行主循环、菜单、状态显示，并调用引擎执行核心操作。包含MD5和语义重复项的交互式审阅和初步决策收集逻辑。
     - `progress_display.py`: (文件未找到) 计划实现进度显示 (如tqdm) 和状态更新。
     - `decision_collector.py`: (文件未找到) 计划中独立的决策收集模块。当前 `cli_interface.py` 已包含基本的交互式决策输入功能。
  - 输入：
    - `cli_interface.py` 输入：
      * `user_args: Dict[str, Any]` - 用户命令行参数，来自:
        - `argparse` 解析的用户输入
      * `engine_results: Dict[str, Any]` - 引擎分析结果，来自:
        - `core/engine.py` 的 `run_analysis()` 方法
      * `storage_data: List[Dict]` - 存储查询结果，来自:
        - `storage/sqlite_storage.py` 的 `query_results()` 方法
    - `progress_display.py` 输入：
      * `analysis_status: Dict[str, Any]` - 分析状态信息，来自:
        - `core/engine.py` 的 `get_analysis_status()` 方法
    - `decision_collector.py` 输入：
      * `duplicate_groups: List[Dict]` - 重复内容组，来自:
        - `core/engine.py` 的 `get_duplicate_groups()` 方法
  - 输出：
    - `cli_interface.py` 输出：
      * `user_decisions: List[Dict]` - 用户决策数据，传递给:
        - `core/engine.py` 的 `apply_decisions()` 方法
      * `export_path: Path` - 导出路径，传递给:
        - `storage/file_storage.py` 的 `export_to_json()` 方法
      * `status_messages: List[str]` - 状态信息，传递给:
        - 终端显示和日志记录
    - `progress_display.py` 输出：
      * `display_status: Dict[str, Any]` - 显示状态，传递给:
        - 终端显示和日志记录
    - `decision_collector.py` 输出：
      * `collected_decisions: List[Dict]` - 收集的决策，传递给:
        - `cli_interface.py` 的 `process_decisions()` 方法
  - 进度：
    * 已完成：基础CLI框架 (`cli.py`)、参数解析 (`argparse` in `cli.py`)、交互式命令行界面 (`cli_interface.py` 主体功能)。
    * 待开发：独立的进度显示模块 (`progress_display.py` 未找到)。
    * 待开发：独立的决策收集模块 (`decision_collector.py` 未找到，但 `cli_interface.py` 已有部分功能)；错误处理机制已部分实现，可进一步完善。
    * 计划中：交互式UI优化


---

## 3. 技术栈与依赖

* **Python 3.12.10** | Poetry 1.8.2 | Pytest 8.1.1 + Coverage 7.4.4 | Black 24.3.0 / isort 5.13.2 / mypy 1.9.0
* **文档解析**：unstructured 0.13.1 (Apache‑2.0)
* **文件去重**：Czkawka 9.0.0 (MIT) – **捆绑CLI与应用一同分发**
* **语义分析**：SentenceTransformers 4.1.0
* **SimHash**：`simhash 1.0.0` (by 1e0ng, MIT License) Python Simhash实现 （待集成）
* **数据库**：SQLite 3.45.1 (Phase 3核心存储) 
* **GUI 预选**：Tkinter 8.6 (Python内置) → PySide 6.7.0 （待实现）
* **日志记录**：loguru 0.7.2 (结构化日志记录)
* **进度显示**：tqdm 4.66.2 (进度条可视化)
* **测试框架**：pytest 8.1.1 + pytest-mock 3.14.0 (单元测试和模拟)
* **代码质量**：
  - black 24.3.0 (代码格式化)
  - isort 5.13.2 (导入排序)
  - mypy 1.9.0 (静态类型检查)
* **数据处理**：
  - pandas 2.2.1 (数据分析)
  - numpy 1.26.4 (数值计算)
* **文件操作**：
  - python-magic 0.4.27 (文件类型检测)
  - pathlib (跨平台路径处理)
* **并发处理**：
  - concurrent.futures (线程池/进程池)
* **配置管理**：
  - pydantic 2.6.4 (配置验证)
  - pydantic-settings 2.9.1 (基于Pydantic的.env和环境变量配置)
* **错误处理**：
  - retrying 1.3.4 (重试机制)

## 3.1 配置管理

项目使用基于Pydantic的分层配置系统，实现类型安全和结构化的配置管理。配置从环境变量、.env文件和默认值中加载，确保灵活性和可维护性。

### 配置组件

1. **StorageConfig**
   - 管理存储相关配置，如数据库URL、目录和文件名
   - 示例属性：`database_url`, `db_dir`, `db_name`

2. **LoggingConfig**
   - 管理日志相关配置，如日志文件路径、级别、轮转和JSON序列化
   - 包含日志级别验证，确保值有效
   - 示例属性：`log_file_path`, `log_level`, `log_rotation`, `log_retention`, `log_serialize_json`

3. **EngineConfig**
   - 管理引擎相关配置，如相似度阈值、语义模型和缓存设置
   - 示例属性：`similarity_threshold`, `semantic_model`, `batch_size`, `cache_dir`

4. **AppConfig**
   - 主配置类，聚合上述所有配置
   - 使用`pydantic-settings`从环境变量和.env文件加载配置
   - 通过属性方法提供对子配置的访问

### 配置加载优先级

1. 环境变量（最高优先级）
2. .env文件中的变量
3. 代码中定义的默认值（最低优先级）

### 配置访问方式

项目采用工厂模式管理配置实例，在`core/factories.py`中实现：

```python
def create_app_config() -> AppConfig:
    """创建并返回应用配置实例"""
    return get_config()  # 单例模式
```

引擎和其他组件通过依赖注入接收配置，确保解耦和可测试性。

## 3.2 日志系统

项目使用Loguru作为统一的日志框架，提供结构化、易于配置和功能丰富的日志记录能力。

### 日志系统特点

1. **结构化日志**
   - 支持JSON格式输出，便于后期分析和处理
   - 包含时间戳、级别、模块等元数据

2. **可配置性**
   - 通过`LoggingConfig`集中管理日志配置
   - 支持设置日志级别、文件路径、轮转策略等
   - 配置从环境变量、.env文件或默认值加载

3. **多目标输出**
   - 文件日志：JSON格式，支持大小轮转和保留策略
   - 控制台日志：美化格式，便于开发调试

4. **异常捕获增强**
   - 自动捕获和格式化异常跟踪信息
   - 支持上下文绑定，丰富日志数据

### 日志初始化流程

1. 在`core/factories.py`中实现`create_logger()`函数，负责:
   - 读取`LoggingConfig`配置
   - 移除默认处理器，避免重复
   - 添加文件和控制台处理器
   - 确保日志目录存在
   - 配置级别、格式和轮转策略

2. 全局初始化一次，通过依赖注入提供给各组件:
   ```python
   # 在应用入口处
   config = create_app_config()
   logger = create_logger(config)
   storage = create_storage(config)
   engine = create_engine(storage, config, logger)
   ```

3. 在需要日志记录的类中接收logger实例:
   ```python
   def __init__(self, logger, ...):
       self.logger = logger
       self.logger.info("组件初始化完成")
   ```

### 日志使用示例

```python
# 简单记录
logger.info("处理文件开始")

# 带上下文信息
logger.bind(file_id="1234", size=1024).info("文件已加载")

# 异常记录
try:
    # 业务代码
except Exception as e:
    logger.exception(f"处理失败: {e}")

# 不同级别记录
logger.debug("详细调试信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误")
```

---

## 4. 数据模型与存储演进

| 表 / 功能         | SQLite - 核心存储                                    | 说明                                                                      |
| ---------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `files`      | id (PK, Integer, AutoInc), file_id (String, UK, NotNull, BusinessUUID), path (String, UK, NotNull), file_hash (String, NotNull), type (String), size (Integer), ctime (DateTime), mtime (DateTime), ingest_time (DateTime, DefaultNow), status (String) | 存储已扫描/处理的文件元数据。`id` 为主键，`file_id` 为业务唯一标识。status可反映是否为Czkawka识别的重复文件等。存储接口已实现并通过测试。        |
| `blocks`         | id (PK, Integer, AutoInc), file_id (Integer, FK to files.id, NotNull), block_id (String(64), NotNull, FromContentHash), content_hash (String(64), NotNull), simhash (String(64)), text (Text, NotNull), block_type (String(32), NotNull), processing_status (String(32), DefaultProcessed), meta_data (JSON) | 存储内容块信息。`file_id` 关联 `files` 表的 `id` 主键。`block_id` 通常使用 `content_hash`。包括其处理状态、各类指纹、原始元素信息及自定义元数据。存储接口已实现并通过测试。     |
| `analysis_results`       | id (PK, Integer), result_id (String(64), UK, NotNull), block_id_1 (String(64), NotNull), block_id_2 (String(64), NotNull), analysis_type (String(32), NotNull), score (JSON, NotNull), details (JSON), block_id (Integer, FK to blocks.id, Historical) | 存储各分析阶段产生的对比结果。`block_id_1` 和 `block_id_2` 引用相关块的 `block_id`。`block_id` 为历史兼容字段，关联 `blocks.id`。存储接口已实现并通过测试。                                   |
| `user_decisions`      | id (PK, Integer, AutoInc), decision_id (String(64), UK, NotNull), result_id (String(64), FK to analysis_results.result_id, NotNull), decision_type (String, NotNull), timestamp (DateTime, DefaultNow), comment (Text), block_id (Integer, FK to blocks.id, Nullable, Historical), duplicate_of_block_id (Integer, FK to blocks.id, Nullable, Historical) | 存储用户或系统自动做出的决策。`result_id` 关联 `analysis_results` 表的 `result_id`。`block_id` 和 `duplicate_of_block_id` 为历史兼容字段，关联 `blocks.id`。存储接口已实现并通过测试。                                              |
| **JSON导出** | 按需从SQLite导出为JSON文件                                         | 用于调试、数据检查或排错场景，不再作为实时数据存储。JSON导出功能已实现。                     |

> **存储策略变更**：
> - 直接使用SQLite作为核心数据存储。
> - 不考虑原存储方式的兼容、迁移。
> - 不再进行JSON文件的实时写入或作为主要数据源。
> - 提供从SQLite导出数据到JSON格式的功能,以便开发阶段查验。

---


## 5. 设计原则与约束 (Guiding Principles & Constraints)

- **5.1 核心原则:**
    - **本地化优先 (Local-First & Offline-Capable):** 核心功能和数据处理完全在本地进行。
    - **模块化与可维护性:** 采用分层架构，各层职责清晰，低耦合，便于独立开发、测试和维护。
    - **分阶段演进:** 功能按计划分阶段实现，特别是对LLM等复杂技术采取谨慎、逐步引入的策略。
    - **用户为中心:** (GUI阶段) 需考虑非技术用户的易用性，提供清晰的结果展示和交互审核机制。
    - **性能与资源效率:** 优化本地计算和存储资源消耗。
- **5.2 主要约束:**
    - **部署模型:** 严格本地化桌面应用，不支持云部署或SaaS模式。
    - **数据隐私:** 最高级别，用户数据不离本地，不收集任何用户行为数据。
    - **LLM使用策略:** 遵循分阶段计划，**当前版本未使用LLM**。后续（Phase 4）仅考虑**本地、辅助型**LLM，结果需用户确认。
    - **性能约束:** 单机运行，需优化内存和CPU使用，处理10万+文件时内存占用不超过2GB。
    - **兼容性:** 支持Windows/macOS/Linux主流操作系统，Python 3.10+环境。
    - **依赖管理:** 仅使用开源许可的第三方库，避免GPL等传染性协议。
    - **安全要求:** 所有文件操作需进行权限检查，防止任意文件读取/写入漏洞。
    - **用户交互:** CLI优先，后续GUI需保持非技术用户友好性。

- **5.3 技术栈:** 
    - * **开发语言:** Python 3.12.10
- **5.4 核心库 (已使用/计划使用):**
    - 文件解析 : `unstructured` (已使用)
    - 语义分析: `SimHashAnalyzer` 、(`1e0ng/simhash` MIT库）（计划使用，待实现）、`Sentence-BERT` (计划使用)
    - 数据存储: SQLite (计划使用，待实现)
    - CLI: `argparse` (已使用), 标准输入输出 (CLI交互)。
    - UI：未定 (待实现)
    - 数据库: SQLite （计划使用，待实现）

## 6. 核心工作流程（core Workflows）
  - 6.1 文档处理与分析流程 (注：此为目标工作流程，部分组件如Czkawka完整集成、SimHash分析、SQLite存储待实现)

  ```mermaid
  sequenceDiagram
    participant User
    participant CLI
    participant Engine
    participant CzkawkaAdapter
    participant PreFilter
    participant BlockProcessor
    participant BlockMerger
    participant MD5Analyzer
    participant SimHashAnalyzer
    participant SBERTAnalyzer
    participant SQLiteStorage
    participant DecisionManager

    User->>CLI: 执行扫描命令(scan-files --dir PATH)
    CLI->>Engine: 启动文件扫描流程
    Engine->>CzkawkaAdapter: 扫描目录查找重复文件
    CzkawkaAdapter-->>Engine: 返回重复文件组列表
    Engine->>PreFilter: 过滤唯一文件
    PreFilter-->>Engine: 返回待处理文件列表
    Engine->>SQLiteStorage: 保存扫描结果到数据库
    Engine->>BlockProcessor: 处理文件内容为原始块
    BlockProcessor-->>Engine: 返回原始内容块列表
    Engine->>BlockMerger: 合并相似代码块
    BlockMerger-->>Engine: 返回合并后内容块列表
    Engine->>MD5Analyzer: 计算块级MD5指纹
    MD5Analyzer-->>Engine: 返回MD5重复块对
    Engine->>SimHashAnalyzer: 计算SimHash指纹
    SimHashAnalyzer-->>Engine: 返回SimHash相似块对
    Engine->>SBERTAnalyzer: 计算语义相似度
    SBERTAnalyzer-->>Engine: 返回语义相似块对
    Engine->>DecisionManager: 汇总分析结果
    DecisionManager->>SQLiteStorage: 持久化分析结果
    Engine-->>CLI: 返回分析完成状态
    CLI-->>User: 显示分析结果摘要
  ```

  - 6.2 决策应用流程（待更新）

  ```mermaid
  sequenceDiagram
    participant User
    participant CLI
    participant Engine
    participant Storage

    User->>CLI: Review Duplicates/Similarities
    CLI->>Engine: get_md5_duplicates() / get_semantic_duplicates()
    Engine-->>CLI: pairs_to_review
    CLI-->>User: Display Pairs
    User->>CLI: Make Decision (e.g., keep_1 for pair X)
    CLI->>Engine: update_decision(pair_key, decision)
    User->>CLI: Save Decisions
    CLI->>Engine: save_decisions()
    Engine->>Storage: save_decisions(filepath, self.block_decisions)
    User->>CLI: Apply Decisions (Generate dedup file)
    CLI->>Engine: apply_decisions()
    Engine->>Storage: load_decisions() # Load fresh decisions if needed
    Engine->>Engine: Generate Output Content based on decisions and blocks_data
    Engine-->>CLI: Output file generated
    CLI-->>User: Confirmation message
```

## 7. API规范（Internal API Specifications）

Python内部接口调用规范，分为核心引擎API和各模块接口。

### 7.1 核心引擎API (KnowledgeDistillerEngine)
- **引擎控制与状态**:
  - `__init__(storage: StorageInterface, config: AppConfig, logger: Any, input_dir: Optional[Union[str, Path]] = None, decision_file: Optional[Union[str, Path]] = None, output_dir: Optional[Union[str, Path]] = None, skip_semantic: bool = False, skip_prefilter: bool = False, similarity_threshold: Optional[float] = None) -> None`: 初始化引擎，注入存储接口、应用配置和日志器。
  - `set_input_dir(input_dir: Union[str, Path]) -> bool`: 设置输入目录并重置分析状态。(已实现)
  - `get_status_summary() -> Dict[str, Any]`: 获取当前引擎状态和分析摘要。(已实现)
  - `_reset_state() -> None`: (内部方法) 重置分析运行相关的内部状态。(已实现)

- **分析流程控制**:
  - `run_prefilter_only() -> Tuple[int, List[Path], List[List[Path]]]`: 仅运行预过滤步骤 (Czkawka)。(已实现)
  - `run_analysis() -> bool`: 启动完整分析流程 (包括预处理、合并、MD5分析、语义分析)。(主要流程已实现，各阶段集成和健壮性持续完善中)
  - `_gather_input_files(input_dir: Path) -> List[Path]`: (内部方法) 收集输入文件。(已实现)
  - `_process_documents(files_to_process: Optional[List[Path]] = None) -> bool`: (内部方法) 调用文档处理器处理文件。(已实现)
  - `_merge_blocks_step() -> bool`: (内部方法) 调用内容块合并逻辑。(已实现)
  - `_initialize_decisions() -> bool`: (内部方法) 初始化决策数据结构。(已实现)
  - `_model_loaded_successfully() -> bool`: (内部方法) 检查语义模型是否成功加载。(已实现)
  - `_filter_blocks_for_semantic() -> List[ContentBlockDTO]`: (内部方法) 过滤用于语义分析的块。(已实现)
  - `stop_analysis()`: (未实现/代码中未找到)

- **决策管理**:
  - `load_decisions() -> bool`: 从存储加载决策。(已实现)
  - `save_decisions() -> bool`: 保存当前决策到存储。(已实现)
  - `update_decision(block_key: str, decision: str) -> bool`: 更新单个块的决策状态。(已实现)
  - `_update_decisions_from_md5(suggested_decisions: Dict[str, str])`: (内部方法) 根据MD5分析结果更新决策。(已实现)
  - `bulk_update_decisions(decisions: Dict[str, DecisionType]) -> None`: (未实现/代码中未找到)
  - `get_pending_decisions() -> List[DecisionPair]`: (未实现/代码中未找到) 

- **结果查询与获取**:
  - `get_md5_duplicates() -> List[List[ContentBlockDTO]]`: 获取MD5精确重复块列表。(已实现)
  - `get_semantic_duplicates() -> List[Tuple[ContentBlockDTO, ContentBlockDTO, float]]`: 获取语义相似块对列表。(已实现)
  - `get_merged_blocks() -> List[MergedBlock]`: (未找到直接API，引擎内部有合并步骤)

- **配置管理**:
  - 配置通过 `

## 8. 接口契约测试

为确保架构的一致性和健壮性，项目实现了接口契约测试，重点验证存储实现类是否严格遵守接口定义。

### 8.1 接口契约的重要性

在以接口为中心的架构中，确保实现类严格遵守接口定义至关重要，这包括：

- 方法名称必须完全匹配
- 参数名称、数量、类型和默认值必须匹配
- 返回类型必须匹配
- 文档字符串应保持一致

任何偏差都可能导致运行时错误，特别是在依赖注入场景下，这些错误可能难以调试。

### 8.2 实现方式

项目使用Python的`inspect`模块动态验证接口契约：

```python
def test_method_signatures_match_interface(self):
    """测试ORMStorage的方法签名是否与StorageInterface一致"""
    interface_methods = self._get_public_methods(StorageInterface)
    implementation_methods = self._get_public_methods(ORMStorage)
    
    # 检查实现类是否包含接口定义的所有方法
    for method_name, interface_method in interface_methods.items():
        assert method_name in implementation_methods, f"方法 {method_name} 未在ORMStorage中实现"
        
        impl_method = implementation_methods[method_name]
        interface_sig = inspect.signature(interface_method)
        impl_sig = inspect.signature(impl_method)
        
        # 检查参数数量和名称
        assert len(interface_sig.parameters) == len(impl_sig.parameters), \
            f"方法 {method_name} 参数数量不匹配: 接口 {len(interface_sig.parameters)} vs 实现 {len(impl_sig.parameters)}"

        # 检查参数名称、类型注解和默认值
        for param_name, interface_param in interface_sig.parameters.items():
            assert param_name in impl_sig.parameters, \
                f"方法 {method_name} 缺少参数 {param_name}"
            
            impl_param = impl_sig.parameters[param_name]
            assert interface_param.kind == impl_param.kind, \
                f"方法 {method_name} 参数 {param_name} 种类不匹配"
            
            # 检查类型注解（如果有）
            if interface_param.annotation != inspect.Parameter.empty:
                assert impl_param.annotation == interface_param.annotation, \
                    f"方法 {method_name} 参数 {param_name} 类型注解不匹配"
            
            # 检查默认值（如果有）
            if interface_param.default != inspect.Parameter.empty:
                assert impl_param.default == interface_param.default, \
                    f"方法 {method_name} 参数 {param_name} 默认值不匹配"
        
        # 检查返回类型
        if interface_sig.return_annotation != inspect.Signature.empty:
            assert impl_sig.return_annotation == interface_sig.return_annotation, \
                f"方法 {method_name} 返回类型不匹配"
```

### 8.3 测试覆盖范围

- 文件位置：`tests/storage/test_storage_contracts.py`
- 测试对象：
  - `StorageInterface` 与 `ORMStorage` 之间的接口契约
  - `StorageInterface` 与 `FileStorage` 之间的接口契约（用于全面覆盖）
- 运行时机：本地开发测试和CI/CD流程中

通过这种方式，项目能够及早发现接口变更带来的影响，确保架构一致性和代码质量。

## 9. 质量保障与测试策略 (Quality Assurance & Testing Strategy)

为确保代码质量、模块间交互的正确性以及架构的健壮性，项目采用多层次的测试策略。

// ... existing code ...