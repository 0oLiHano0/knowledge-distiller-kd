"""
=================================================
factory.py - MD5AnalysisStage 工厂 (v4.7)
=================================================
**模块功能**:

- 负责创建和配置 `MD5AnalysisStage` 实例。
- 与 Storage 解耦，仅依赖于 context 和 settings。

"""
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.core.interfaces import StageInterface
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
        self._logger.info('MD5AnalysisStageFactory 初始化完成.')

    def create(self, settings: MD5AnalysisStageSettings) ->StageInterface:
        """
        创建并返回一个配置好的 MD5AnalysisStage 实例。
        """
        self._logger.info(f'创建 MD5AnalysisStage 实例...')
        stage_instance = MD5AnalysisStage(logger=self._logger.bind(
            stage_name='MD5Analysis'), settings=settings)
        self._logger.success('MD5AnalysisStage 实例创建成功.')
        return stage_instance
