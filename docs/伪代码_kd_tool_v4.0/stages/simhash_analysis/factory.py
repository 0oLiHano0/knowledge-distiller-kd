# kd_tool/stages/simhash_analysis/factory.py(v4.6 - Schema 路径更新版)
# -*- coding: utf-8 -*-

"""
=================================================
factory.py - P06 SimHash 分析阶段工厂 (v4.6)
=================================================

**模块功能**:

- 负责创建和配置 `SimHashAnalysisStage` 实例。
- 遵循类式工厂模式。
- **规范**:
    - **必须**通过构造函数注入 `Logger`。
    - `create` 方法**必须**接收 `StorageInterface`、`SimHashAnalysisStageSettings` 和可选的 `SimHashAdapterInterface`。
    - 如果未提供 `SimHashAdapterInterface`，工厂**必须**负责创建默认的适配器实例 (例如 `SimhashLibAdapter`)。
    - **必须**将所有依赖项注入到 `SimHashAnalysisStage` 实例中。

---
"""

from loguru import Logger
from typing import Optional

# --- 核心模块导入 ---
# [指令] StorageInterface 从 core 层导入
from ....core.interfaces import StorageInterface # 路径相对于 kd_tool/stages/simhash_analysis/

# --- Stage 实现与适配器导入 (本地导入) ---
from .simhash_analysis_stage import SimHashAnalysisStage
from .adapter_interface import SimHashAdapterInterface
from .simhash_adapter import SimhashLibAdapter # 默认适配器

# --- [指令] 更新 SimHashAnalysisStageSettings 的导入路径 ---
from .settings_models import SimHashAnalysisStageSettings # <-- [指令] 已更新为本地导入

class SimHashAnalysisStageFactory:
    """
    创建 `SimHashAnalysisStage` 实例的工厂。
    """

    def __init__(self, logger: Logger):
        """
        **规范**: 工厂自身依赖 (如 Logger) **必须**通过构造函数注入。
        """
        self._logger = logger.bind(factory="SimHashAnalysisStageFactory")

    def create(self,
               storage: StorageInterface,
               settings: SimHashAnalysisStageSettings, # <-- [指令] 类型已更新
               adapter: Optional[SimHashAdapterInterface] = None
               ) -> SimHashAnalysisStage:
        """
        创建并返回一个配置好的 `SimHashAnalysisStage` 实例。

        **参数**:
            storage (StorageInterface): 存储服务实例。
            settings (SimHashAnalysisStageSettings): SimHash 阶段的配置。
            adapter (Optional[SimHashAdapterInterface]): (可选) 自定义的 SimHash 适配器。
                                                        如果为 None，则创建默认的 `SimhashLibAdapter`。

        **返回**:
            SimHashAnalysisStage: 配置好的 SimHash 分析阶段实例。
        """
        self._logger.debug("开始创建 SimHashAnalysisStage 实例...")

        if adapter is None:
            self._logger.debug("未提供 SimHash 适配器，创建默认的 SimhashLibAdapter...")
            # **注意**: 如果 SimhashLibAdapter 需要自己的配置或 Logger，
            #           也应该在这里进行注入或创建。
            adapter = SimhashLibAdapter() 
            self._logger.debug("默认 SimhashLibAdapter 创建成功。")

        stage_instance = SimHashAnalysisStage(
            logger=self._logger, # 将工厂的 logger 传递下去，Stage 会进一步 bind
            storage=storage,
            settings=settings,
            adapter=adapter
        )
        self._logger.success("SimHashAnalysisStage 实例创建成功。")

        return stage_instance
