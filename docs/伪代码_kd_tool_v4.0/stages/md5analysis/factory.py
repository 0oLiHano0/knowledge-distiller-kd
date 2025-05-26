# kd_tool/stages/md5analysis/factory.py (v4.6 - Schema 路径更新版)
# -*- coding: utf-8 -*-


"""
=================================================
factory.py.md - MD5AnalysisStage 工厂 (v4.6)
=================================================

**模块功能**:

- 负责创建并组装 MD5AnalysisStage 实例及其依赖。
- 遵循方案二，将工厂置于其对应的 Stage 目录中。

---
"""


from loguru import Logger

# --- 核心模块导入 ---
# [指令] StageInterface 和 StorageInterface 从 core 层导入
from ....core.interfaces import StageInterface, StorageInterface # 路径相对于 kd_tool/stages/md5analysis/

# --- Stage 实现导入 (本地导入) ---
from .md5_analysis_stage import MD5AnalysisStage # 原文件名 st.md5.01.md5_analysis_stage.py.md

# --- [指令] 更新 MD5AnalysisStageSettings 的导入路径 ---
from .settings_models import MD5AnalysisStageSettings # <-- [指令] 已更新为本地导入


# --- 类式工厂定义 ---
class MD5AnalysisStageFactory:
    """
    负责创建 MD5AnalysisStage 实例。
    """

    def __init__(self, logger: Logger): #
        """
        初始化 MD5 分析阶段工厂。
        """
        self._logger = logger.bind(factory_name="MD5AnalysisStageFactory") #
        self._logger.info("MD5AnalysisStageFactory initialized.") #

    def create(
        self,
        settings: MD5AnalysisStageSettings, # <-- [指令] 类型已更新为本地导入的模型
        storage: StorageInterface,
    ) -> StageInterface: #
        """
        创建并返回一个配置好的 MD5AnalysisStage 实例。
        """
        self._logger.info(f"Creating MD5AnalysisStage instance...") #

        stage_instance = MD5AnalysisStage( #
            logger=self._logger.bind(stage_name="MD5Analysis"), #
            settings=settings, #
            storage=storage, #
        )

        self._logger.success("MD5AnalysisStage instance created successfully.") #
        return stage_instance #

