# kd_tool/stages/cleanup/factory.py (v4.6 - Schema 路径更新版)
# -*- coding: utf-8 -*-

"""
=================================================
factory.py - P09 清理阶段工厂 (v4.6)
=================================================
**模块功能**:

- 负责创建和配置 `CleanupStage` 实例。
- **规范**: 负责创建文件系统适配器 (如果未提供)。

---
"""

from loguru import Logger
from typing import Optional

from ....core.interfaces import StorageInterface # 路径相对于 kd_tool/stages/cleanup/
from .settings_models import CleanupStageSettings # <-- [指令] 已更新为本地导入
from .cleanup_stage import CleanupStage 
from .adapter_interface import FileSystemAdapterInterface
from .real_file_system_adapter import RealFileSystemAdapter # 默认适配器


class CleanupStageFactory:
    """
    创建 `CleanupStage` 实例的工厂。
    """

    def __init__(self, logger: Logger): #
        """工厂构造函数。"""
        self._logger = logger.bind(factory="CleanupStageFactory") #

    def create(self,
               storage: StorageInterface,
               settings: CleanupStageSettings, # <-- [指令] 类型已更新
               adapter: Optional[FileSystemAdapterInterface] = None
               ) -> CleanupStage: #
        """
        创建并返回一个配置好的 `CleanupStage` 实例。
        """
        self._logger.debug("开始创建 CleanupStage 实例...") #

        if adapter is None: #
            self._logger.debug("未提供文件系统适配器，创建默认的 RealFileSystemAdapter...") #
            adapter = RealFileSystemAdapter(logger=self._logger.bind(component="FileSystemAdapter")) #
            self._logger.debug("默认 RealFileSystemAdapter 创建成功。") #

        stage_instance = CleanupStage( #
            logger=self._logger, #
            storage=storage, #
            settings=settings, #
            fs_adapter=adapter #
        )
        self._logger.success("CleanupStage 实例创建成功。") #

        return stage_instance #