"""
此模块定义了 StorageFactory，负责根据配置创建和初始化存储服务实例。
它现在使用 LoggerProtocol 进行日志记录。
"""
# 从新的日志层导入 LoggerProtocol
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.storage.settings_models import StorageSettingsDTO #
from kd_tool.storage.sqlite_storage import SQLiteStorage #
from kd_tool.core.errors import KDToolError #
from kd_tool.storage.storage_interface import StorageInterface


class FactoryError(KDToolError): #
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
    它通过构造函数接收一个 LoggerProtocol 实例。
    """

    def __init__(self, logger: LoggerProtocol):
        """
        初始化 StorageFactory。

        参数:
            logger (LoggerProtocol): 用于日志记录的、符合 LoggerProtocol 接口的实例。
        """
        self._logger = logger.bind(factory='StorageFactory')
        self._logger.info('StorageFactory 实例已创建。') #

    def create(self, settings: StorageSettingsDTO) -> StorageInterface: #
        """
        根据提供的设置创建并初始化一个存储服务实例。
        会将自身的 LoggerProtocol 实例（已绑定上下文）传递给创建的存储实例。

        参数:
            settings (StorageSettingsDTO): 存储服务的配置设置。

        返回:
            StorageInterface: 一个配置好并已初始化的存储服务实例。

        可能抛出的异常:
            UnsupportedStorageBackendError: 如果配置的后端类型不被支持。
            StorageInitializationError: 如果存储后端在初始化过程中失败。
        """
        backend_type = settings.backend_type
        self._logger.info(f"尝试创建存储服务，后端类型: '{backend_type}'") #
        storage_instance: StorageInterface #

        if backend_type == 'sqlite':
            self._logger.debug(f"为 'sqlite' 后端实例化 SQLiteStorage...") #
            # 将绑定了新上下文的 LoggerProtocol 实例传递给 SQLiteStorage
            sqlite_logger = self._logger.bind(component='SQLiteStorage')
            storage_instance = SQLiteStorage(settings=settings, logger=sqlite_logger) #
        else:
            self._logger.error(f"在工厂中遇到不支持的存储后端类型: '{backend_type}'") #
            raise UnsupportedStorageBackendError(backend_type=backend_type)

        try:
            self._logger.debug(f"正在初始化存储后端: '{backend_type}'...") #
            storage_instance.initialize()
            # 假设 LoggerProtocol 有 success 方法，如果无，则用 info
            try:
                self._logger.success(f"存储后端 '{backend_type}' 已成功初始化。") # type: ignore
            except AttributeError:
                self._logger.info(f"存储后端 '{backend_type}' 已成功初始化。") #
        except Exception as e:
            self._logger.exception(f"存储后端 '{backend_type}' 初始化失败。") #
            raise StorageInitializationError(backend_type=backend_type,
                original_exception=e) from e

        return storage_instance