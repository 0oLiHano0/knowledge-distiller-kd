"""
=================================================
factory.py - SimHash 分析阶段工厂 (v4.7)
=================================================

**模块功能**:

- 负责创建和配置 `SimHashAnalysisStage` 实例。
- 与 Storage 解耦，仅依赖于 context 和 settings。

"""

from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.stages.simhash_analysis.simhash_analysis_stage import SimHashAnalysisStage
from kd_tool.stages.simhash_analysis.adapter_interface import SimHashAdapterInterface
from kd_tool.stages.simhash_analysis.simhash_adapter import SimhashLibAdapter
from kd_tool.stages.simhash_analysis.settings_models import SimHashAnalysisStageSettings
from typing import Optional


class SimHashAnalysisStageFactory:
    """
    创建 `SimHashAnalysisStage` 实例的工厂。
    """

    def __init__(self, logger: LoggerProtocol):
        """
        **规范**: 工厂自身依赖 (如 Logger) **必须**通过构造函数注入。
        """
        self._logger = logger.bind(factory="SimHashAnalysisStageFactory")

    def create(
        self,
        settings: SimHashAnalysisStageSettings,
        adapter: Optional[SimHashAdapterInterface] = None,
    ) -> SimHashAnalysisStage:
        """
        创建并返回一个配置好的 `SimHashAnalysisStage` 实例。

        **参数**:
            storage (StorageInterface): 存储服务实例。
            settings (SimHashAnalysisStageSettings): SimHash 阶段的配置。
            adapter (Optional[SimHashAdapterInterface]): (可选) 自定义的 SimHash 适配器。
                                                        如果为 None，则创建默认的 `SimhashLibAdapter`。

        **返回**:
            SimHashAnalysisStage: 配置好的 SimHash 分析阶段实例。
        """
        self._logger.debug("开始创建 SimHashAnalysisStage 实例...")
        if adapter is None:
            self._logger.debug("未提供 SimHash 适配器，创建默认的 SimhashLibAdapter...")
            adapter = SimhashLibAdapter()
            self._logger.debug("默认 SimhashLibAdapter 创建成功。")
        stage_instance = SimHashAnalysisStage(
            logger=self._logger, settings=settings, adapter=adapter
        )
        self._logger.success("SimHashAnalysisStage 实例创建成功。")
        return stage_instance
