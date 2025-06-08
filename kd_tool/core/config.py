"""
====================开发指引======================
kd_tool/core/config.py - v4.7
=================================================

**【文件定位】**  
- 所属包结构：kd_tool.core
- 所属模块/层次：核心服务/配置管理
- 该文件为 KD-Tool 应用的顶层配置入口，聚合所有子模块（如存储、日志、各处理阶段等）的配置 DTO，供工厂、服务、编排器等依赖注入。

**【模块职责（SRP）】**  
- 唯一职责：定义并管理 KD-Tool 应用的全局配置模型 AppConfig，集中化聚合所有子模块的配置对象。

**【依赖关系与注入】**  
- 依赖外部服务/DTO：
    - StorageSettingsDTO（kd_tool.storage.settings_models）
    - OrchestratorSettings（kd_tool.core.core_settings_models）
    - LoggingConfigDTO（kd_tool.logging.settings）
    - 各阶段配置 DTO（如 PrefilterStageSettings、MD5AnalysisStageSettings 等，均在各自模块 settings_models 中定义）
- 依赖注入方式：所有依赖通过构造器注入（如工厂/服务/编排器），禁止直接实例化依赖。
- Mock/替换需求：如需测试可通过 Pydantic 的 Config 或自定义 Mock DTO 实现。

**【输入输出规范】**  
- AppConfig 字段：
    - project_name: str
    - project_version: str
    - storage: StorageSettingsDTO
    - orchestrator: OrchestratorSettings
    - logging: LoggingConfigDTO
    - prefilter: PrefilterStageSettings
    - document_processing: DocumentProcessingStageSettings
    - block_merging: BlockMergerStageSettings
    - md5_analysis: MD5AnalysisStageSettings
    - simhash_analysis: SimHashAnalysisStageSettings
    - semantic_analysis: SemanticAnalysisStageSettings
    - decision: DecisionStageSettings
    - cleanup: CleanupStageSettings
- 字段类型：全部为 Pydantic DTO，强制类型注解
- 输出结果：AppConfig 实例，供工厂/服务/编排器依赖注入
- 异常类型：配置加载/校验失败时抛出自定义异常（如 ConfigLoadError，需继承自 KDToolError）
- DTO/ORM/原始数据边界：仅允许 Pydantic DTO 作为配置载体，禁止直接传递 dict/ORM

**【核心架构约束】**  
- 禁止业务逻辑与配置耦合
- 禁止直接实例化依赖，必须依赖注入
- 必须类型注解
- 日志/异常处理需符合全局规范
- 重要类/函数需添加三段式注释（WHY/WHAT/HOW）
- 禁止相对导入，强制绝对导入
- 配置模型字段需通过 default_factory 注入默认值，确保无参初始化可用

**【接口与DTO规范】**  
- 仅暴露 AppConfig 顶层配置类
- DTO 必须在各自模块定义，接口与实现分离
- 自定义异常类需继承自 KDToolError
- 字段需包含类型注解、默认值、描述信息

**【日志与安全】**  
- 日志记录点：配置加载、校验、异常
- 日志级别：INFO（加载成功）、ERROR（加载/校验失败）
- 敏感信息处理：如有密钥/凭证，日志输出需脱敏，禁止明文输出敏感字段

**【任务清单】**  
1. [✔] 明确各子模块配置 DTO 的导入路径与依赖关系，确保全部为绝对导入
2. [✔] 规范 AppConfig 字段类型、默认值、描述、类型注解
3. [✔] 实现自定义异常类（如 ConfigLoadError），并在配置加载/校验失败时抛出（当前文件职责范围内无需实现，建议在加载逻辑中补充）
4. [✔] 为 AppConfig 及其关键方法添加三段式注释（WHY/WHAT/HOW）
5. [✖] 编写单元测试（tests/core/test_config.py），覆盖正常与异常场景（建议后续补充）
6. [✔] 检查并修正所有导入为绝对路径
7. [✔] 日志记录点与敏感信息处理规范化（本文件为模型定义，日志点应在加载/校验环节实现）

**【其他说明】**  
- 若未来有新阶段模块，仅需在本文件新增对应配置字段及导入，无需修改现有逻辑
- 历史遗留配置项需逐步迁移至各自子模块 DTO
- TODO：后续可考虑支持多环境配置加载（如 dev/prod）

"""

from pydantic import BaseModel, Field, ConfigDict, ValidationError
from kd_tool.core.core_settings_models import OrchestratorSettings, OrchestratorSettingsError
from kd_tool.logging.settings import (
    LoggingConfigDTO,
)  # kd_tool/logging/settings.py 日志配置
from kd_tool.storage.settings_models import StorageSettingsDTO, StorageBackend
from kd_tool.stages.prefilter.settings_models import PrefilterStageSettings
from kd_tool.stages.docprocessing.settings_models import DocumentProcessingStageSettings
from kd_tool.stages.blockmerging.settings_models import BlockMergerStageSettings
from kd_tool.stages.md5analysis.settings_models import MD5AnalysisStageSettings
from kd_tool.stages.simhash_analysis.settings_models import SimHashAnalysisStageSettings
from kd_tool.stages.semantic_analysis.settings_models import (
    SemanticAnalysisStageSettings,
)
from kd_tool.stages.decision.settings_models import DecisionStageSettings
from kd_tool.stages.cleanup.settings_models import CleanupStageSettings


class AppConfig(BaseModel):
    """
    WHY: 统一管理 KD_Tool 应用的所有全局配置，确保各子模块配置集中、可控，便于依赖注入和后续扩展。
    WHAT: 作为应用程序配置的唯一入口点，聚合所有子模块（如存储、日志、各处理阶段等）的 Pydantic 配置对象，定义顶层配置模型 AppConfig。
    HOW: 由 ApplicationBuilder 加载本配置模型，并将各子模块配置通过依赖注入分发给工厂、服务、编排器等，所有字段类型均为各自模块的 Pydantic DTO，字段通过 default_factory 注入默认值，确保无参初始化可用。
    """

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, arbitrary_types_allowed=True
    )
    project_name: str = Field(
        default="Knowledge Distiller KD-Tool",
        description="项目的名称，可能用于日志、报告或界面显示。",
    )
    project_version: str = Field(default="4.6.0-dev", description="项目的当前版本号。")
    storage: StorageSettingsDTO = Field(
        default_factory=lambda: StorageSettingsDTO(backend=StorageBackend.SQLITE),
        description="存储服务配置。"
    )
    orchestrator: OrchestratorSettings = Field(
        default_factory=OrchestratorSettings, description="编排器配置。"
    )
    logging: LoggingConfigDTO = Field(
        default_factory=LoggingConfigDTO,  # kd_tool/logging/settings.py 日志配置
        description="日志系统配置。",
    )
    prefilter: PrefilterStageSettings = Field(
        default_factory=PrefilterStageSettings, description="P02 预过滤阶段配置。"
    )
    document_processing: DocumentProcessingStageSettings = Field(
        default_factory=DocumentProcessingStageSettings,
        description="P03 文档处理阶段配置。",
    )
    block_merging: BlockMergerStageSettings = Field(
        default_factory=BlockMergerStageSettings, description="P04 块合并阶段配置。"
    )
    md5_analysis: MD5AnalysisStageSettings = Field(
        default_factory=MD5AnalysisStageSettings, description="P05 MD5 分析阶段配置。"
    )
    simhash_analysis: SimHashAnalysisStageSettings = Field(
        default_factory=SimHashAnalysisStageSettings,
        description="P06 SimHash 分析阶段配置。",
    )
    semantic_analysis: SemanticAnalysisStageSettings = Field(
        default_factory=SemanticAnalysisStageSettings,
        description="P07 语义分析阶段配置。",
    )
    decision: DecisionStageSettings = Field(
        default_factory=DecisionStageSettings, description="P08 决策阶段配置。"
    )
    cleanup: CleanupStageSettings = Field(
        default_factory=CleanupStageSettings, description="P09 清理阶段配置。"
    )

def load_orchestrator_settings_from_dict(config_dict: dict) -> OrchestratorSettings:
    """
    WHY: 统一入口加载OrchestratorSettings，确保类型安全和异常结构化。
    WHAT: 从dict加载配置，校验失败时抛出OrchestratorSettingsError。
    HOW: 捕获Pydantic ValidationError并转换。
    """
    try:
        return OrchestratorSettings(**config_dict)
    except ValidationError as ve:
        raise OrchestratorSettingsError("OrchestratorSettings 配置校验失败", ve)
