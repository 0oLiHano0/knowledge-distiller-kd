"""
=================================================
factory.py - DecisionStage 工厂 (v4.7)
=================================================

**模块功能**:

- 负责创建和配置 `DecisionStage` 实例。
- **规范**: 遵循类式工厂模式。
- 与 Storage 解耦，仅依赖于 context 和 settings。

---
"""
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.stages.decision.decision_stage import DecisionStage
from kd_tool.stages.decision.settings_models import DecisionStageSettings


class DecisionStageFactory:
    """
    创建 `DecisionStage` 实例的工厂。
    """

    def __init__(self, logger: LoggerProtocol):
        """工厂构造函数。"""
        self._logger = logger.bind(factory='DecisionStageFactory')

    def create(self, settings: DecisionStageSettings
        ) ->DecisionStage:
        """
        创建并返回一个配置好的 `DecisionStage` 实例。
        """
        self._logger.debug('开始创建 DecisionStage 实例...')
        stage_instance = DecisionStage(logger=self._logger, settings=settings)
        self._logger.success('DecisionStage 实例创建成功。')
        return stage_instance
