# kd_tool/stages/blockmerging/factory.py (v4.6 - Schema 路径更新版)
# -*- coding: utf-8 -*-

"""
=================================================
factory.py.md - BlockMergerStage 工厂 (v4.6)
=================================================
... (模块注释保持不变) ...
---
"""

from loguru import Logger

# --- 核心模块导入 ---
# [指令] StageInterface 和 StorageInterface 从 core 层导入
from ....core.interfaces import StageInterface, StorageInterface # 路径相对于 kd_tool/stages/blockmerging/

# --- Stage 实现导入 (本地导入) ---
from .block_merging_stage import BlockMergerStage

# --- [指令] 更新 BlockMergerStageSettings 的导入路径 ---
from .settings_models import BlockMergerStageSettings # <-- [指令] 已更新为本地导入


# --- 类式工厂定义 ---
class BlockMergerStageFactory:
    """
    负责创建 BlockMergerStage 实例。
    """

    def __init__(self, logger: Logger): #
        """
        初始化块合并阶段工厂。
        """
        self._logger = logger.bind(factory_name="BlockMergerStageFactory") #
        self._logger.info("BlockMergerStageFactory initialized.") #

    def create(
        self,
        settings: BlockMergerStageSettings, # <-- [指令] 类型已更新为本地导入的模型
        storage: StorageInterface,
    ) -> StageInterface: #
        """
        创建并返回一个配置好的 BlockMergerStage 实例。
        """
        self._logger.info(f"Creating BlockMergerStage instance...") #

        stage_instance = BlockMergerStage( #
            logger=self._logger.bind(stage_name="BlockMerger"), #
            settings=settings, #
            storage=storage, #
        )

        self._logger.success("BlockMergerStage instance created successfully.") #
        return stage_instance #