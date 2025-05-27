"""
=================================================
config.py - KD_Tool 应用程序顶层配置 (v4.6)
=================================================

**模块功能**:

- **核心职责**: 定义应用程序的顶层配置模型 `AppConfig`。
- **技术实现**: `AppConfig` 聚合了所有子模块或组件的配置设置，
              这些设置的Pydantic模型定义已分散到各自的模块/层级中。
- **v4.6 核心变更**:
    - **[架构指令]** `AppConfig` 内部的字段类型 **必须** 更新为从新的、分散的位置导入对应的 Settings Models。
    - **[架构指令]** 不再使用 `AnalysisSettings` 聚合模型，而是直接在 `AppConfig` 中定义并引用
                   各个分析阶段 (MD5, SimHash, Semantic) 的具体配置模型。
    - **[架构指令]** 确保所有导入路径正确无误。

---
"""
from pydantic import BaseModel, Field
from kd_tool.core.core_settings_models import OrchestratorSettings
from kd_tool.core.logging.logging_settings_models import LoggingSettings
from kd_tool.storage.settings_models import StorageSettingsDTO
from kd_tool.stages.prefilter.settings_models import PrefilterStageSettings
from kd_tool.stages.docprocessing.settings_models import DocumentProcessingStageSettings
from kd_tool.stages.blockmerging.settings_models import BlockMergerStageSettings
from kd_tool.stages.md5analysis.settings_models import MD5AnalysisStageSettings
from kd_tool.stages.simhash_analysis.settings_models import SimHashAnalysisStageSettings
from kd_tool.stages.semantic_analysis.settings_models import SemanticAnalysisStageSettings
from kd_tool.stages.decision.settings_models import DecisionStageSettings
from kd_tool.stages.cleanup.settings_models import CleanupStageSettings


class AppConfig(BaseModel):
    """
    KD_Tool v4.0 应用程序的顶层配置模型。
    **规范**: 这是应用程序配置的唯一入口点。
             它聚合了所有子模块的配置。
    **编码要求**: `ApplicationBuilder` 将负责加载此配置并分发给各个工厂。
    """
    project_name: str = Field(default='Knowledge Distiller KD-Tool',
        description='项目的名称，可能用于日志、报告或界面显示。')
    project_version: str = Field(default='4.6.0-dev', description='项目的当前版本号。')
    storage: StorageSettingsDTO = Field(default_factory=StorageSettingsDTO,
        description='存储服务配置。')
    orchestrator: OrchestratorSettings = Field(default_factory=
        OrchestratorSettings, description='编排器配置。')
    logging: LoggingSettings = Field(default_factory=LoggingSettings,
        description='日志系统配置。')
    prefilter: PrefilterStageSettings = Field(default_factory=
        PrefilterStageSettings, description='P02 预过滤阶段配置。')
    document_processing: DocumentProcessingStageSettings = Field(
        default_factory=DocumentProcessingStageSettings, description=
        'P03 文档处理阶段配置。')
    block_merging: BlockMergerStageSettings = Field(default_factory=
        BlockMergerStageSettings, description='P04 块合并阶段配置。')
    md5_analysis: MD5AnalysisStageSettings = Field(default_factory=
        MD5AnalysisStageSettings, description='P05 MD5 分析阶段配置。')
    simhash_analysis: SimHashAnalysisStageSettings = Field(default_factory=
        SimHashAnalysisStageSettings, description='P06 SimHash 分析阶段配置。')
    semantic_analysis: SemanticAnalysisStageSettings = Field(default_factory
        =SemanticAnalysisStageSettings, description='P07 语义分析阶段配置。')
    decision: DecisionStageSettings = Field(default_factory=
        DecisionStageSettings, description='P08 决策阶段配置。')
    cleanup: CleanupStageSettings = Field(default_factory=
        CleanupStageSettings, description='P09 清理阶段配置。')


    class Config:
        extra = 'forbid'
        validate_assignment = True
        arbitrary_types_allowed = True
