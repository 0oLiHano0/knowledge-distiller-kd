"""
此模块定义了 StorageFactory，负责根据配置创建和初始化存储服务实例。
... (模块注释保持不变) ...
"""
from loguru import Logger
from kd_tool.storage.settings_models import StorageSettingsDTO
from kd_tool.storage.storage_interface import StorageInterface
from kd_tool.storage.sqlite_storage import SQLiteStorage
from kd_tool.core.errors import KDToolError


class FactoryError(KDToolError):
    """与工厂操作相关的基本异常。"""
    pass


class UnsupportedStorageBackendError(FactoryError):
    """当请求不支持的存储后端类型时抛出。"""

    def __init__(self, backend_type: str):
        super().__init__(message=f'不支持的存储后端类型: {backend_type}')
        self.backend_type = backend_type


class StorageInitializationError(FactoryError):
    """当存储后端初始化失败时抛出。"""

    def __init__(self, backend_type: str, original_exception: Exception):
        full_message = f"存储后端 '{backend_type}' 初始化失败。"
        super().__init__(message=full_message, original_exception=
            original_exception)
        self.backend_type = backend_type


class StorageFactory:
    """
    一个类式工厂，负责创建和初始化存储服务实例。
    """

    def __init__(self, logger: Logger):
        self._logger = logger.bind(factory='StorageFactory')
        self._logger.info('StorageFactory 实例已创建。')

    def create(self, settings: StorageSettingsDTO) ->StorageInterface:
        """
        根据提供的设置创建并初始化一个存储服务实例。
        """
        backend_type = settings.backend_type
        self._logger.info(f"尝试创建存储服务，后端类型: '{backend_type}'")
        storage_instance: StorageInterface
        if backend_type == 'sqlite':
            self._logger.debug(f"为 'sqlite' 后端实例化 SQLiteStorage...")
            storage_instance = SQLiteStorage(settings=settings, logger=self
                ._logger.bind(component='SQLiteStorage'))
        else:
            self._logger.error(f"在工厂中遇到不支持的存储后端类型: '{backend_type}'")
            raise UnsupportedStorageBackendError(backend_type=backend_type)
        try:
            self._logger.debug(f"正在初始化存储后端: '{backend_type}'...")
            storage_instance.initialize()
            self._logger.success(f"存储后端 '{backend_type}' 已成功初始化。")
        except Exception as e:
            self._logger.exception(f"存储后端 '{backend_type}' 初始化失败。")
            raise StorageInitializationError(backend_type=backend_type,
                original_exception=e) from e
        return storage_instance
