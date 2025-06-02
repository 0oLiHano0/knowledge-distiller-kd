"""
=================================================
factory.py - SemanticAnalysisStage 工厂 (v4.7)
=================================================

**模块功能**:

- 负责创建和配置 `SemanticAnalysisStage` 实例。
- 与 Storage 解耦，仅依赖于 context 和 settings。

"""
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.stages.semantic_analysis.settings_models import SemanticAnalysisStageSettings
from kd_tool.stages.semantic_analysis.semantic_analysis_stage import SemanticAnalysisStage
from kd_tool.stages.semantic_analysis.adapter_interface import SemanticAdapterInterface
from kd_tool.stages.semantic_analysis.sentence_transformer_adapter import SentenceTransformerAdapter
from typing import Optional

class SemanticAnalysisStageFactory:
    """
    创建 `SemanticAnalysisStage` 实例的工厂。
    """

    def __init__(self, logger: LoggerProtocol):
        """工厂构造函数。"""
        self._logger = logger.bind(factory='SemanticAnalysisStageFactory')

    def create(self, settings:
        SemanticAnalysisStageSettings, adapter: Optional[
        SemanticAdapterInterface]=None) ->SemanticAnalysisStage:
        """
        创建并返回一个配置好的 `SemanticAnalysisStage` 实例。

        **参数**:
            storage (StorageInterface): 存储服务实例。
            settings (SemanticAnalysisStageSettings): 语义分析阶段的配置。
            adapter (Optional[SemanticAdapterInterface]): (可选) 自定义的语义分析适配器。
                                                        如果为 None，则创建默认的 `SentenceTransformerAdapter`。

        **返回**:
            SemanticAnalysisStage: 配置好的语义分析阶段实例。
        """
        self._logger.debug('开始创建 SemanticAnalysisStage 实例...')
        if adapter is None:
            self._logger.debug('未提供语义分析适配器，创建默认的 SentenceTransformerAdapter...'
                )
            adapter = SentenceTransformerAdapter()
            self._logger.debug('默认 SentenceTransformerAdapter 创建成功。')
        stage_instance = SemanticAnalysisStage(logger=self._logger, settings=settings, adapter=adapter)
        self._logger.success('SemanticAnalysisStage 实例创建成功。')
        return stage_instance
