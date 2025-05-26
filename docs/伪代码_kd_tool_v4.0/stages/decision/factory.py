# kd_tool/stages/decision/factory.py (v4.6 - Schema 路径更新版)
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

# --- 核心模块导入 ---
# [指令] StorageInterface 从 core 层导入
from ....core.interfaces import StorageInterface # 路径相对于 kd_tool/stages/decision/

# --- Stage 实现导入 (本地导入) ---
from .decision_stage import DecisionStage

# --- [指令] 更新 DecisionStageSettings 的导入路径 ---
from .settings_models import DecisionStageSettings # <-- [指令] 已更新为本地导入


class DecisionStageFactory:
    """
    创建 `DecisionStage` 实例的工厂。
    """

    def __init__(self, logger: Logger): #
        """工厂构造函数。"""
        self._logger = logger.bind(factory="DecisionStageFactory") #

    def create(self,
               storage: StorageInterface,
               settings: DecisionStageSettings # <-- [指令] 类型已更新
               ) -> DecisionStage: #
        """
        创建并返回一个配置好的 `DecisionStage` 实例。
        """
        self._logger.debug("开始创建 DecisionStage 实例...") #

        stage_instance = DecisionStage( #
            logger=self._logger, #
            storage=storage, #
            settings=settings #
        )
        self._logger.success("DecisionStage 实例创建成功。") #

        return stage_instance #