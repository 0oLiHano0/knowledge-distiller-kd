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
from kd_tool.core.core_settings_models import OrchestratorSettings, OrchestratorSettingsError
from kd_tool.core.interfaces import StageInterface
from kd_tool.core.orchestrator import Orchestrator
from kd_tool.core.errors import KDToolError
from kd_tool.core.config import load_orchestrator_settings_from_dict


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
        config_dict: dict,
    ) -> Orchestrator:
        """
        **[指令]** 创建并返回一个配置好的 `Orchestrator` 实例。
        **必须** 通过config.py加载配置，捕获并转换异常。

        **参数**:
            stage_modules (Dict[str, StageInterface]): 阶段模块字典。
            config_dict (dict): OrchestratorSettings配置字典。

        **返回**:
            Orchestrator: 一个准备好使用的 Orchestrator 实例。
        """
        self._logger.info("尝试创建 Orchestrator 实例...")
        try:
            settings = load_orchestrator_settings_from_dict(config_dict)
            default_stage_order = settings.default_stage_order
            self._logger.debug(f"使用的 OrchestratorSettings: {settings}")
            orchestrator_instance = Orchestrator(
                stage_modules=stage_modules,
                default_stage_order=default_stage_order,
                settings=settings,
                logger=self._logger,
            )
            self._logger.success("Orchestrator 实例已成功创建。")
            return orchestrator_instance
        except OrchestratorSettingsError as ose:
            self._logger.error(f"配置校验失败: {ose}")
            raise OrchestratorCreationError(message=str(ose), original_exception=ose)
        except Exception as e:
            self._logger.exception("创建 Orchestrator 实例时发生未预料的错误。")
            raise OrchestratorCreationError(message=str(e), original_exception=e) from e
