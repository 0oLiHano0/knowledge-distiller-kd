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
        default_stage_order: List[str],
        settings: OrchestratorSettings,
    ) -> Orchestrator:
        """
        **[指令]** 创建并返回一个配置好的 `Orchestrator` 实例。
        **必须** 确保所有参数都传递给 `Orchestrator` 的构造函数。

        **参数**:
            stage_modules (Dict[str, StageInterface]): 阶段模块字典。
            default_stage_order (List[str]): 默认阶段顺序列表。
            settings (OrchestratorSettings): Orchestrator 配置。

        **返回**:
            Orchestrator: 一个准备好使用的 Orchestrator 实例。
        """
        self._logger.info("尝试创建 Orchestrator 实例...")
        self._logger.debug(
            f"接收到 {len(stage_modules)} 个阶段模块，默认顺序: {default_stage_order}"
        )
        self._logger.debug(f"使用的 OrchestratorSettings: {settings}")
        try:
            orchestrator_instance = Orchestrator(
                stage_modules=stage_modules,
                default_stage_order=default_stage_order,
                settings=settings,
                logger=self._logger,
            )
            self._logger.success("Orchestrator 实例已成功创建。")
            return orchestrator_instance
        except Exception as e:
            self._logger.exception("创建 Orchestrator 实例时发生未预料的错误。")
            raise OrchestratorCreationError(message=str(e), original_exception=e) from e
