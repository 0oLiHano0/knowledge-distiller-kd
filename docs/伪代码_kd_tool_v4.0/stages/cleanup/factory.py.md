```python

# kd_tool/stages/cleanup/factory.py
# -*- coding: utf-8 -*-

"""
=================================================
factory.py - P09 清理阶段工厂 (v4.7)
=================================================

**模块功能**:

- 负责创建和配置 `CleanupStage` 实例。
- **规范**: 负责创建文件系统适配器 (如果未提供)。

---
"""

from loguru import Logger
from typing import Optional

from ...core.interfaces import StorageInterface
from ...schemas.settings_models import CleanupStageSettings
from .cleanup_stage import CleanupStage
from .adapter_interface import FileSystemAdapterInterface
from .real_file_system_adapter import RealFileSystemAdapter # 默认适配器

class CleanupStageFactory:
    """
    创建 `CleanupStage` 实例的工厂。
    """

    def __init__(self, logger: Logger):
        """工厂构造函数。"""
        self._logger = logger.bind(factory="CleanupStageFactory")

    def create(self,
               storage: StorageInterface,
               settings: CleanupStageSettings,
               fs_adapter: Optional[FileSystemAdapterInterface] = None
               ) -> CleanupStage:
        """
        创建并返回一个配置好的 `CleanupStage` 实例。

        **参数**:
            storage (StorageInterface): 存储服务实例。
            settings (CleanupStageSettings): 清理阶段的配置。
            fs_adapter (Optional[FileSystemAdapterInterface]): (可选) 自定义的文件系统适配器。

        **返回**:
            CleanupStage: 配置好的清理阶段实例。
        """
        self._logger.debug("开始创建 CleanupStage 实例...")

        if fs_adapter is None:
            self._logger.debug("未提供文件系统适配器，创建默认的 RealFileSystemAdapter...")
            fs_adapter = RealFileSystemAdapter() 
            self._logger.debug("默认 RealFileSystemAdapter 创建成功。")

        stage_instance = CleanupStage(
            logger=self._logger, 
            storage=storage,
            settings=settings,
            fs_adapter=fs_adapter
        )
        self._logger.success("CleanupStage 实例创建成功。")

        return stage_instance

```