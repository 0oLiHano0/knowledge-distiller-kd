```python

# kd_tool/stages/decision/factory.py
# -*- coding: utf-8 -*-

"""
=================================================
factory.py - P08 决策阶段工厂 (v4.6)
=================================================

**模块功能**:

- 负责创建和配置 `DecisionStage` 实例。
- **规范**: 遵循类式工厂模式。

---
"""

from loguru import Logger

from ...core.interfaces import StorageInterface
from ...schemas.settings_models import DecisionStageSettings
from .decision_stage import DecisionStage

class DecisionStageFactory:
    """
    创建 `DecisionStage` 实例的工厂。
    """

    def __init__(self, logger: Logger):
        """工厂构造函数。"""
        self._logger = logger.bind(factory="DecisionStageFactory")

    def create(self,
               storage: StorageInterface,
               settings: DecisionStageSettings
               ) -> DecisionStage:
        """
        创建并返回一个配置好的 `DecisionStage` 实例。

        **参数**:
            storage (StorageInterface): 存储服务实例。
            settings (DecisionStageSettings): 决策阶段的配置。

        **返回**:
            DecisionStage: 配置好的决策阶段实例。
        """
        self._logger.debug("开始创建 DecisionStage 实例...")

        stage_instance = DecisionStage(
            logger=self._logger, 
            storage=storage,
            settings=settings
        )
        self._logger.success("DecisionStage 实例创建成功。")

        return stage_instance

```