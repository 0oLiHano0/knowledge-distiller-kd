"""
=================================================
factory.py - P09 清理阶段工厂 (v4.6)
=================================================
**模块功能**:

- 负责创建和配置 `CleanupStage` 实例。
- **规范**: 负责创建文件系统适配器 (如果未提供)。

---
"""
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.core.interfaces import StorageInterface
from typing import Optional
from kd_tool.stages.cleanup.settings_models import CleanupStageSettings
from kd_tool.stages.cleanup.cleanup_stage import CleanupStage
from kd_tool.stages.cleanup.adapter_interface import FileSystemAdapterInterface
from kd_tool.stages.cleanup.real_file_system_adapter import RealFileSystemAdapter


class CleanupStageFactory:
    """
    创建 `CleanupStage` 实例的工厂。
    """

    def __init__(self, logger: LoggerProtocol):
        """工厂构造函数。"""
        self._logger = logger.bind(factory='CleanupStageFactory')

    def create(self, storage: StorageInterface, settings:
        CleanupStageSettings, adapter: Optional[FileSystemAdapterInterface]
        =None) ->CleanupStage:
        """
        创建并返回一个配置好的 `CleanupStage` 实例。
        """
        self._logger.debug('开始创建 CleanupStage 实例...')
        if adapter is None:
            self._logger.debug('未提供文件系统适配器，创建默认的 RealFileSystemAdapter...')
            adapter = RealFileSystemAdapter(logger=self._logger.bind(
                component='FileSystemAdapter'))
            self._logger.debug('默认 RealFileSystemAdapter 创建成功。')
        stage_instance = CleanupStage(logger=self._logger, storage=storage,
            settings=settings, fs_adapter=adapter)
        self._logger.success('CleanupStage 实例创建成功。')
        return stage_instance
