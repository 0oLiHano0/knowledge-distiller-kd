# 路线图：Phase 1 - 存储集成、日志迁移与架构基础优化

**目标:** 完成存储层与核心引擎的集成，将日志系统迁移至 Loguru，并引入依赖管理工厂、分层配置和接口契约测试，为后续 `engine.py` 的解耦和项目整体的健壮性、可维护性打下坚实基础。

**核心原则:**

*   **解耦优先:** 通过接口和依赖注入，减少模块间（特别是 Engine 与其他部分）的直接耦合。
*   **职责清晰:** 让各模块（Engine, Factory, Storage, Config）承担明确、单一的职责。
*   **一致性:** 建立统一的依赖获取、配置管理和日志记录模式。
*   **可测试性:** 结构化设计，方便单元测试和集成测试，尤其易于 Mock 依赖。
*   **可配置性:** 将易变配置（路径、阈值、连接参数）与代码分离。
*   **健壮性:** 通过类型检查、配置验证和接口契约测试提升代码质量。

---

## 阶段一：奠定基础 - 配置管理、依赖工厂与引擎解耦

*   **总目标:** 建立集中的配置管理和依赖注入机制，并将引擎对存储的依赖通过接口和工厂进行解耦。

*   **任务 1.1: 实现分层配置管理 (Pydantic)** [✅已完成]
    *   **目标:** 使用 Pydantic 集中管理、验证并加载应用配置。
    *   **内容:**
        *   ✅ 创建 `knowledge_distiller_kd/core/config.py`。
        *   ✅ 定义 Pydantic 模型：`StorageConfig` (包含数据库 URL/路径, 可选的超时/池大小等), `LoggingConfig` (日志文件路径, 级别, 轮转设置, JSON格式开关), `EngineConfig` (相似度阈值, Czkawka路径等, 可按需细化), 以及一个顶层 `AppConfig` 聚合这些模型。
        *   ✅ 实现配置加载逻辑：优先从环境变量或 `.env` 文件加载，若无则使用合理的默认值。
        *   ✅ 添加必要的验证规则到 Pydantic 模型中。
    *   **产出:** 
        *   ✅ 可靠、类型安全的配置加载与访问机制
        *   ✅ 完整的测试覆盖
        *   ✅ 配置模板文件 (.env.example)
        *   ✅ 更新的架构设计文档
    *   **遇到的问题:**
        *   `.env.example`文件创建时遇到权限问题，通过使用shell命令成功解决
        *   从`constants.py`迁移配置项时需要调整配置模型结构，已完成迁移

*   **任务 1.2: 实现依赖管理工厂 (`core/factories.py`)** [✅已完成]
    *   **目标:** 集中创建和提供核心依赖项（配置、存储实例、日志器）。
    *   **内容:**
        *   ✅ 创建 `knowledge_distiller_kd/core/factories.py`。
        *   ✅ 实现一个工厂类或一组函数，用于：
            *   ✅ 加载 `AppConfig` (调用 Task 1.1 的逻辑)。
            *   ✅ 根据 `StorageConfig` 创建并返回配置好的 `StorageInterface` 实现实例 (例如 `ORMStorage`)。
            *   ✅ (准备阶段) 提供获取配置好的 `logger` 实例的方法 (将在 Phase 2 完成具体配置)。
            *   ✅ (准备阶段) 提供创建 `KnowledgeDistillerEngine` 实例的方法，并将工厂管理的其他依赖（storage, config, logger）注入。
    *   **产出:** ✅ 统一的依赖获取入口，简化应用启动和测试设置。
    *   **遇到的问题:**
        *   无

*   **任务 1.3: 重构引擎初始化与依赖注入** [✅已完成]
    *   **目标:** 使 `Engine` 通过构造函数接收其所有外部依赖，并由工厂提供。
    *   **内容:**
        *   ✅ 修改 `knowledge_distiller_kd/core/engine.py` 中 `KnowledgeDistillerEngine.__init__` 方法，使其接受 `storage: StorageInterface`, `config: AppConfig` (或其子配置), 以及 `logger` (Loguru logger instance) 参数。移除引擎内部创建这些依赖的逻辑。
        *   ✅ 更新 `knowledge_distiller_kd/cli.py` (或主程序入口)，使用 `core/factories.py` 来创建 `Engine` 实例及其依赖项。
    *   **产出:** ✅ `Engine` 与其依赖项创建过程解耦，提高了可测试性和灵活性。
    *   **遇到的问题:**
        *   修改 Engine 初始化后导致多个测试失败，需要更新测试用例
        *   实现了配置和日志器的依赖注入，使 Engine 更加可测试，实现了目标
        *   创建了通用测试夹具，简化测试代码

*   **任务 1.4: 在引擎中集成 `StorageInterface` 调用**
    *   **目标:** 将 `Engine` 内所有直接的数据持久化操作替换为对注入的 `storage` 对象的调用。
    *   **内容:**
        *   全面扫描 `engine.py`，识别所有数据库交互或文件系统读写（用于持久化）的代码。
        *   替换为 `self.storage.method_name(...)` 调用，例如 `save_blocks`, `get_analysis_results`, `register_file` 等。
        *   确保在引擎和存储层之间传递的数据使用的是 `core.models.py` 中定义的统一 DTO 或 SQLAlchemy 模型。特别关注并解决任何模型不一致问题（如 `processing.ContentBlock` vs `core.models.ContentBlock`）。
    *   **产出:** `Engine` 不再关心存储的具体实现细节，只通过接口交互。

*   **任务 1.5: 集成存储生命周期与错误处理**
    *   **目标:** 确保存储资源被正确管理，并且存储层错误得到健壮处理。
    *   **内容:**
        *   确保存储初始化逻辑（如 `storage.initialize()`）在应用启动时通过工厂或主程序调用。
        *   考虑实现并调用 `storage.finalize()`（如果需要，例如显式关闭连接池）在应用退出时。
        *   在 `engine.py` 中，为所有 `self.storage.*` 调用添加 `try...except` 块。捕获可能由存储层抛出的特定异常（如 `SQLAlchemyError`），记录详细错误信息（使用 logger），并根据需要将其封装为更高层次的领域特定异常（如自定义的 `KDStorageError` 或 `KDCoreError`）后重新抛出或进行处理。
    *   **产出:** 更可靠的存储操作和更清晰的错误反馈路径。

---

## 阶段二：统一日志 - 迁移至 Loguru

*   **总目标:** 将项目日志系统完全切换到 Loguru，利用其结构化日志和便捷配置。

*   **任务 2.1: 通过工厂和配置完成 Loguru 设置**
    *   **目标:** 使用集中配置初始化 Loguru，并通过工厂提供 logger 实例。
    *   **内容:**
        *   在 `core/factories.py` 中添加创建和配置 `logger` 实例的逻辑。
        *   该逻辑应读取 `LoggingConfig` (来自 Task 1.1) 中的设置。
        *   使用 `logger.add()` 配置 sinks：
            *   一个文件 sink，写入 `LoggingConfig` 指定的路径，使用 JSON 格式 (`serialize=True`)，并配置轮转 (`rotation=`, `retention=`) 和级别 (`level=`)。
            *   一个控制台 sink (`sys.stderr`)，使用更易读的格式 (`serialize=False`)，级别可单独配置。
        *   确保工厂能返回配置好的 `logger` 对象。
    *   **产出:** 可配置、结构化的日志系统实例。

*   **任务 2.2: 全局迁移日志调用**
    *   **目标:** 替换代码库中所有标准 `logging` 模块的使用。
    *   **内容:**
        *   搜索整个 `knowledge_distiller_kd` 目录，将 `import logging`, `logging.getLogger(...)`, 以及 `logger.*` (标准库 logger) 的调用替换为：
            *   `from loguru import logger` (在模块顶部)
            *   直接使用 `logger.*` 调用 (Loguru 的 logger)。
        *   鼓励使用 `logger.bind(key=value)` 添加上下文信息到日志记录中。
        *   对于需要显式传递 logger 的类或函数（如 `Engine`），确保从工厂获取并注入。
    *   **产出:** 整个项目使用统一、功能更强的 Loguru 进行日志记录。

*   **任务 2.3: 移除旧日志配置代码**
    *   **目标:** 清理遗留代码，避免配置冲突。
    *   **内容:**
        *   删除所有 `logging.basicConfig()` 调用以及任何其他标准库 `logging` 的配置代码。
    *   **产出:** 清洁的日志系统实现。

---

## 阶段三：质量保障与文档同步

*   **总目标:** 通过自动化测试和文档更新，确保本阶段所有变更的质量和可维护性。

*   **任务 3.1: 编写接口契约测试 (本地优先)**
    *   **目标:** **在本地测试环境中**验证存储实现是否严格遵守 `StorageInterface` 定义，为后续 CI 集成打好基础。
    *   **内容:**
        *   创建新的测试文件，例如 `tests/storage/test_storage_contracts.py`。
        *   编写测试用例，使用 Python 的 `inspect` 模块或相关工具，检查 `ORMStorage`（以及未来可能的其他实现）的所有公开方法签名（方法名、参数数量、参数类型注解、返回类型注解）是否与 `StorageInterface` 中定义的完全一致。
        *   确保这些测试可以在本地通过 `pytest` 运行。
        *   **(延后任务)** 当 CI/CD 流程配置完成后，将此测试套件添加到 CI 配置文件（如 `.github/workflows/ci.yml`）中。
    *   **产出:** 可在本地运行的接口契约测试，提供即时反馈；为自动化接口保障奠定基础。

*   **任务 3.2: 全面测试覆盖**
    *   **目标:** 确保所有新引入和修改的代码都有充分的单元测试和集成测试。
    *   **内容:**
        *   **单元测试:**
            *   为 `core/config.py` 编写测试，验证配置加载和验证逻辑。
            *   为 `core/factories.py` 编写测试，验证依赖项是否正确创建和配置（可使用 Mock 验证配置是否传递）。
            *   更新 `Engine` 的单元测试，使用 Mock `