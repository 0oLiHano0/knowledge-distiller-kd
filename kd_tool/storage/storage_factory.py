# kd_tool/storage/storage_factory.py
"""
WHY  : 提供统一入口，屏蔽不同存储后端创建细节。
WHAT : 根据 StorageSettingsDTO 生产 StorageInterface 实例。
HOW  : 工厂模式 + 依赖注入，避免直接耦合。
"""
from __future__ import annotations

from kd_tool.storage.errors import StorageInitializationError
from kd_tool.storage.settings_models import StorageBackend, StorageSettingsDTO
from kd_tool.storage.sqlite_storage import SQLiteStorage
from kd_tool.storage.storage_interface import StorageInterface
from kd_tool.logging.protocols import LoggerProtocol


class FactoryError(StorageInitializationError):
    """工厂通用错误。"""


class StorageFactory:
    """
    WHY: 统一存储工厂。
    WHAT: 根据配置创建StorageInterface实例。
    HOW: 依赖注入logger 和 settings。
    """

    def __init__(self, logger: LoggerProtocol, settings: StorageSettingsDTO):
        self._logger = logger
        self._settings = settings

    def create(self) -> StorageInterface:
        """根据 backend 创建对应实现并初始化。"""
        backend = self._settings.backend

        if backend is StorageBackend.SQLITE:
            storage: StorageInterface = SQLiteStorage(self._settings, self._logger)
        else:
            raise FactoryError(f"未知后端: {backend}")

        self._logger.debug(f"开始初始化存储后端: {backend}")
        storage.initialize()
        self._logger.success(f"后端 {backend} 初始化完成")  # Loguru success

        return storage
