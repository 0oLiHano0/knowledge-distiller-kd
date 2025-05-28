"""
=================================================
factory.py.md - MD5AnalysisStage 工厂 (v4.6)
=================================================

**模块功能**:

- 负责创建并组装 MD5AnalysisStage 实例及其依赖。
- 遵循方案二，将工厂置于其对应的 Stage 目录中。

---
"""
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.core.interfaces import StageInterface, StorageInterface
from kd_tool.stages.md5analysis.md5_analysis_stage import MD5AnalysisStage
from kd_tool.stages.md5analysis.settings_models import MD5AnalysisStageSettings


class MD5AnalysisStageFactory:
    """
    负责创建 MD5AnalysisStage 实例。
    """

    def __init__(self, logger: LoggerProtocol):
        """
        初始化 MD5 分析阶段工厂。
        """
        self._logger = logger.bind(factory_name='MD5AnalysisStageFactory')
        self._logger.info('MD5AnalysisStageFactory initialized.')

    def create(self, settings: MD5AnalysisStageSettings, storage:
        StorageInterface) ->StageInterface:
        """
        创建并返回一个配置好的 MD5AnalysisStage 实例。
        """
        self._logger.info(f'Creating MD5AnalysisStage instance...')
        stage_instance = MD5AnalysisStage(logger=self._logger.bind(
            stage_name='MD5Analysis'), settings=settings, storage=storage)
        self._logger.success('MD5AnalysisStage instance created successfully.')
        return stage_instance
