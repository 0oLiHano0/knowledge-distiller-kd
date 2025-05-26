# kd_tool/storage/factory.py (导入更新)
"""
此模块定义了 StorageFactory，负责根据配置创建和初始化存储服务实例。
... (模块注释保持不变) ...
"""

from loguru import Logger

# --- 从 schemas, storage, core 模块导入核心定义 ---
# [指令] 更新 StorageSettingsDTO 的导入路径
from .settings_models import StorageSettingsDTO  # <-- [指令] 已更新
from .storage_interface import StorageInterface # [指令] 保持相对导入 (如果适用)
from .sqlite_storage import SQLiteStorage     # [指令] 保持相对导入 (如果适用)

# [指令] 从 core.errors 导入基础错误
from ..core.errors import KDToolError #


# --- 工厂相关的自定义异常 ---
class FactoryError(KDToolError): #
    """与工厂操作相关的基本异常。"""
    pass

class UnsupportedStorageBackendError(FactoryError): #
    """当请求不支持的存储后端类型时抛出。"""
    def __init__(self, backend_type: str):
        super().__init__(message=f"不支持的存储后端类型: {backend_type}")
        self.backend_type = backend_type

class StorageInitializationError(FactoryError): #
    """当存储后端初始化失败时抛出。"""
    def __init__(self, backend_type: str, original_exception: Exception):
        full_message = f"存储后端 '{backend_type}' 初始化失败。"
        super().__init__(message=full_message, original_exception=original_exception)
        self.backend_type = backend_type


# --- 类式工厂定义 ---
class StorageFactory:
    """
    一个类式工厂，负责创建和初始化存储服务实例。
    """
    # ... (构造函数保持不变) ...
    def __init__(self, logger: Logger): #
        self._logger = logger.bind(factory="StorageFactory")
        self._logger.info("StorageFactory 实例已创建。")


    # [指令] create 方法的 settings 参数类型现在来自 .settings_models
    def create(self, settings: StorageSettingsDTO) -> StorageInterface: #
        """
        根据提供的设置创建并初始化一个存储服务实例。
        """
        # ... (方法内部逻辑保持不变) ...
        backend_type = settings.backend_type #
        self._logger.info(f"尝试创建存储服务，后端类型: '{backend_type}'") #

        storage_instance: StorageInterface #

        if backend_type == "sqlite": #
            self._logger.debug(f"为 'sqlite' 后端实例化 SQLiteStorage...") #
            storage_instance = SQLiteStorage( #
                settings=settings, #
                logger=self._logger.bind(component="SQLiteStorage") #
            )
        # ...
        else: #
            self._logger.error(f"在工厂中遇到不支持的存储后端类型: '{backend_type}'") #
            raise UnsupportedStorageBackendError(backend_type=backend_type) #

        try: #
            self._logger.debug(f"正在初始化存储后端: '{backend_type}'...") #
            storage_instance.initialize() #
            self._logger.success(f"存储后端 '{backend_type}' 已成功初始化。") #
        except Exception as e: #
            self._logger.exception(f"存储后端 '{backend_type}' 初始化失败。") #
            raise StorageInitializationError(backend_type=backend_type, original_exception=e) from e #

        return storage_instance #