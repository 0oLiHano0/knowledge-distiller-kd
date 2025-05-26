# kd_tool/stages/prefilter/prefilter_factory.py (v4.6 - Schema 路径更新版)
# -*- coding: utf-8 -*-

"""
=================================================
prefilter_factory.py.md - PrefilterStage 工厂 (v4.6)
=================================================

**模块功能**:

- 负责创建并组装 PrefilterStage 实例及其依赖。
- **v4.6 核心变更**:
    - **[架构指令]** `PrefilterStageSettings` 的导入路径已更新为指向本地的 `settings_models.py`。
    - **[架构指令]** 核心接口和错误类的导入路径已更新，以反映其在项目中的标准位置。

---
"""

from loguru import Logger

# --- 核心模块导入 ---
# [指令] 核心接口、存储接口、核心错误的导入路径应为绝对路径或相对于项目根的正确相对路径。
# 假设 prefilter stage 位于 kd_tool/stages/prefilter/
from ....core.interfaces import StageInterface, StorageInterface
from ....core.errors import KDToolError

# --- Stage 和 Adapter 导入 (这些已经是正确的本地相对导入) ---
from .prefilter_stage import PrefilterStage
from .adapter_interface import CzkawkaAdapterInterface
from .czkawka_adapter import CzkawkaAdapter

# --- [指令] 更新 PrefilterStageSettings 的导入路径 ---
from .settings_models import PrefilterStageSettings # <-- [指令] 已更新为本地导入

# --- 自定义异常 ---
# [架构师说明]: 工厂自身的错误可以定义在此，也可以考虑移至更通用的工厂错误模块。
#               当前保持在此处。
class FactoryConfigurationError(KDToolError): #
    """当工厂遇到配置问题时抛出。"""
    def __init__(self, message: str, **kwargs): # 保持与 KDToolError 一致
        super().__init__(message, module="PrefilterStageFactory", **kwargs)


# --- 类式工厂定义 ---
class PrefilterStageFactory:
    """
    负责创建 PrefilterStage 实例。
    **规范**: 严格遵循依赖注入。
    """

    def __init__(self, logger: Logger):
        """
        初始化预过滤阶段工厂。
        **参数**:
            logger (Logger): **[必须]** 日志记录器实例。
        """
        self._logger = logger.bind(factory_name="PrefilterStageFactory")
        self._logger.info("PrefilterStageFactory initialized.")

    def create(self,
                 settings: PrefilterStageSettings, # <-- [指令] 类型已更新为本地导入的模型
                 storage: StorageInterface
                 ) -> StageInterface:
        """
        创建并返回一个配置好的 PrefilterStage 实例。
        **参数**:
            settings (PrefilterStageSettings): **[必须]** Prefilter 阶段的配置 DTO。
            storage (StorageInterface): **[必须]** 存储服务接口实例。
        **返回**:
            StageInterface: 一个实现了 `StageInterface` 的 `PrefilterStage` 实例。
        """
        self._logger.info("Creating PrefilterStage instance...") #

        # [指令] 必须校验 Czkawka 配置的完整性 (如果选用 Czkawka 工具)
        if settings.tool == "czkawka" and not settings.czkawka: #
            self._logger.error("Czkawka settings are missing, but it's the selected tool. Cannot create CzkawkaAdapter.") #
            raise FactoryConfigurationError( #
                message="PrefilterStage 配置错误：工具设置为 'czkawka'，但 'czkawka' 配置块缺失。" #
            )

        # [指令] 根据配置创建相应的适配器实例
        czkawka_adapter_instance: CzkawkaAdapterInterface
        if settings.tool == "czkawka": #
             # [指令] CzkawkaSettings 现在从 settings.czkawka 获取，其类型也应正确
             czkawka_adapter_instance = CzkawkaAdapter( #
                settings=settings.czkawka, # type: ignore # 假设 settings.czkawka 是 CzkawkaSettings 类型
                logger=self._logger.bind(component="CzkawkaAdapter") #
             )
        else: #
             # [指令] 如果未来支持其他工具，在此处添加逻辑
             self._logger.error(f"Unsupported prefilter tool: {settings.tool}") #
             raise FactoryConfigurationError(message=f"不支持的预过滤工具: {settings.tool}") #

        # [指令] 创建 PrefilterStage 实例并注入所有依赖
        stage_instance = PrefilterStage( #
            logger=self._logger.bind(stage_name="Prefilter"), # 为 Stage 绑定特定的日志上下文
            settings=settings, #
            storage=storage, #
            czkawka_adapter=czkawka_adapter_instance #
        )

        self._logger.success("PrefilterStage instance created successfully.") #
        return stage_instance #