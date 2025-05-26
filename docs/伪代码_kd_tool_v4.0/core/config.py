# kd_tool/core/config.py (v4.6 - Schema 路径更新版)
# -*- coding: utf-8 -*-

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

# --- Pydantic 导入 ---
from pydantic import BaseModel, Field
# [指令] 不再需要从此处导入 DirectoryPath, FilePath, List, Optional，因为这些主要在子配置模型中使用

# --- 从新的、分散的位置导入所有需要的配置模型 ---

# 1. 核心层配置导入
from kd_tool.core.settings_models import OrchestratorSettings # [指令] 从 core/settings_models.py 导入
from kd_tool.core.logging.settings_models import LoggingSettings # [指令] 从 core/logging/settings_models.py 导入

# 2. 存储层配置导入
from kd_tool.storage.settings_models import StorageSettingsDTO # [指令] 从 storage/settings_models.py 导入

# 3. 各个 Stage 的配置导入
from kd_tool.stages.prefilter.settings_models import PrefilterStageSettings
from kd_tool.stages.docprocessing.settings_models import DocumentProcessingStageSettings
from kd_tool.stages.blockmerging.settings_models import BlockMergerStageSettings
from kd_tool.stages.md5analysis.settings_models import MD5AnalysisStageSettings
from kd_tool.stages.simhash_analysis.settings_models import SimHashAnalysisStageSettings
from kd_tool.stages.semantic_analysis.settings_models import SemanticAnalysisStageSettings
from kd_tool.stages.decision.settings_models import DecisionStageSettings
from kd_tool.stages.cleanup.settings_models import CleanupStageSettings


# ==============================================================================
# 应用主配置模型 (AppConfig)
# ==============================================================================
# [架构师说明]: AppConfig 是应用程序配置的唯一入口点 (Single Source of Truth)。
#               它聚合了所有子模块的配置。

class AppConfig(BaseModel):
    """
    KD_Tool v4.0 应用程序的顶层配置模型。
    **规范**: 这是应用程序配置的唯一入口点。
             它聚合了所有子模块的配置。
    **编码要求**: `ApplicationBuilder` 将负责加载此配置并分发给各个工厂。
    """
    project_name: str = Field(
        default="Knowledge Distiller KD-Tool",
        description="项目的名称，可能用于日志、报告或界面显示。"
    )
    project_version: str = Field(
        default="4.6.0-dev", # <-- [指令] 更新版本号以反映当前的重构状态
        description="项目的当前版本号。"
    )

    # --- 基础服务配置 ---
    # [指令] 类型注解必须更新为从新位置导入的模型
    storage: StorageSettingsDTO = Field(
        default_factory=StorageSettingsDTO,
        description="存储服务配置。"
    )
    orchestrator: OrchestratorSettings = Field(
        default_factory=OrchestratorSettings,
        description="编排器配置。"
    )
    logging: LoggingSettings = Field(
        default_factory=LoggingSettings,
        description="日志系统配置。"
    )

    # --- 各个阶段的配置 ---
    # [指令] 类型注解必须更新为从新位置导入的模型
    prefilter: PrefilterStageSettings = Field(
        default_factory=PrefilterStageSettings,
        description="P02 预过滤阶段配置。"
    )
    document_processing: DocumentProcessingStageSettings = Field(
        default_factory=DocumentProcessingStageSettings,
        description="P03 文档处理阶段配置。"
    )
    block_merging: BlockMergerStageSettings = Field(
        default_factory=BlockMergerStageSettings,
        description="P04 块合并阶段配置。"
    )

    # [架构指令 v4.6] 不再使用 AnalysisSettings 聚合模型。
    # AppConfig 直接包含各个分析阶段的配置。
    md5_analysis: MD5AnalysisStageSettings = Field(
        default_factory=MD5AnalysisStageSettings,
        description="P05 MD5 分析阶段配置。"
    )
    simhash_analysis: SimHashAnalysisStageSettings = Field(
        default_factory=SimHashAnalysisStageSettings,
        description="P06 SimHash 分析阶段配置。"
    )
    semantic_analysis: SemanticAnalysisStageSettings = Field(
        default_factory=SemanticAnalysisStageSettings,
        description="P07 语义分析阶段配置。"
    )
    # --- 结束分析阶段配置 ---

    decision: DecisionStageSettings = Field(
        default_factory=DecisionStageSettings,
        description="P08 决策阶段配置。"
    )
    cleanup: CleanupStageSettings = Field(
        default_factory=CleanupStageSettings,
        description="P09 清理阶段配置。"
    )

    class Config:
        extra = 'forbid' # **规范**: 顶层配置也禁止未知字段。
        validate_assignment = True
        arbitrary_types_allowed = True # 仍然需要，因为子模型可能包含Path等

# --- 配置加载函数 (示例性，实际加载逻辑会更复杂) ---
# [架构师说明]: load_config 函数的具体实现超出了当前 Schema 重构的范围，
#               但 ApplicationBuilder 依赖它。在编码阶段需要实现此函数，
#               可以考虑使用 pydantic-settings 等库。
# def load_app_config(config_file: Optional[Path] = None, **override_kwargs) -> AppConfig:
#     """
#     (占位符) 从多种来源加载 AppConfig。
#     实际实现需要解析 YAML/JSON 等文件，并可能合并环境变量。
#     """
#     # coding 阶段: 实现从文件加载 AppConfig 的逻辑。
#     # 示例:
#     # if config_file and config_file.exists():
#     #     from pydantic_settings import SettingsConfigDict
#     #     from pydantic_settings.sources import DotEnvSettingsSource, YamlSettingsSource
#     #     # class ConfigurableAppConfig(AppConfig):
#     #     #     model_config = SettingsConfigDict(
#     #     #         yaml_file=config_file,
#     #     #         env_file='.env',
#     #     #         env_nested_delimiter='__',
#     #     #         extra='ignore' # or 'forbid'
#     #     #     )
#     #     # return ConfigurableAppConfig()
#     # else:
#     #     # 返回一个包含所有默认值的 AppConfig 实例
#     #     return AppConfig()
#     pass # 占位