"""
=================================================
orchestrator_factory.py - Orchestrator 工厂 (v4.1)
=================================================

**模块功能**:

- **核心职责**: 创建 `Orchestrator` 实例。
- **v4.1 核心变更**:
    - **[指令] 必须** 更新 `create` 方法签名以接收 `default_stage_order`。
    - **[指令] 必须** 确保将所有必需参数传递给 `Orchestrator` 构造函数。

---
"""

from typing import Dict, List
from kd_tool.logging.protocols import (
    LoggerProtocol,
)  # kd_tool/logging/protocols.py 日志协议
from kd_tool.core.core_settings_models import OrchestratorSettings
from kd_tool.core.interfaces import StageInterface
from kd_tool.core.orchestrator import Orchestrator
from kd_tool.core.errors import KDToolError
from kd_tool.storage.storage_interface import StorageInterface


class FactoryError(KDToolError):
    """与工厂操作相关的基本异常。"""

    pass


class OrchestratorCreationError(FactoryError):
    """当 Orchestrator 创建失败时抛出。"""

    def __init__(self, message: str, original_exception: Exception):
        full_message = f"Orchestrator 创建失败: {message}"
        super().__init__(message=full_message, original_exception=original_exception)


class OrchestratorFactory:
    """
    一个类式工厂，负责创建和配置 Orchestrator 实例。
    """

    def __init__(self, logger: LoggerProtocol):
        """
        初始化工厂。
        """
        self._logger = logger.bind(factory="OrchestratorFactory")
        self._logger.info("OrchestratorFactory 实例已创建。")

    def create(
        self,
        stage_modules: Dict[str, StageInterface],
        settings: OrchestratorSettings,
        storage: StorageInterface,
    ) -> Orchestrator:
        """
        创建并返回一个配置好的 `Orchestrator` 实例。

        参数:
            stage_modules: 已实例化的 StageInterface 字典。
            settings: 经 Pydantic 校验的 OrchestratorSettings 实例。
            storage: StorageInterface 实例，用于流水线持久化。

        返回:
            Orchestrator: 完整配置好的编排器实例。
        """
        self._logger.info("尝试创建 Orchestrator 实例...")
        try:
            orchestrator_instance = Orchestrator(
                stage_modules=stage_modules,
                default_stage_order=settings.default_stage_order,
                settings=settings,
                logger=self._logger,
                storage=storage,
            )
            self._logger.success("Orchestrator 实例已成功创建。")
            return orchestrator_instance
        except Exception as e:
            self._logger.exception("创建 Orchestrator 实例时发生未预料的错误。")
            raise OrchestratorCreationError(message=str(e), original_exception=e) from e
