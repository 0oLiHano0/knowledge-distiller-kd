# kd_tool/stages/semantic_analysis/factory.py (v4.6 - Schema 路径更新版)
# -*- coding: utf-8 -*-

"""
=================================================
factory.py - P07 语义分析阶段工厂 (v4.6)
=================================================

**模块功能**:

- 负责创建和配置 `SemanticAnalysisStage` 实例。
- **规范**: 遵循类式工厂模式，处理适配器的创建。

---
"""

from loguru import Logger
from typing import Optional

from ....core.interfaces import StorageInterface # 调整相对路径层级
# [指令] 必须从同级目录下的 settings_models.py 导入 SemanticAnalysisStageSettings
from .settings_models import SemanticAnalysisStageSettings
from .semantic_analysis_stage import SemanticAnalysisStage
from .adapter_interface import SemanticAdapterInterface
from .sentence_transformer_adapter import SentenceTransformerAdapter # 默认适配器

class SemanticAnalysisStageFactory:
    """
    创建 `SemanticAnalysisStage` 实例的工厂。
    """

    def __init__(self, logger: Logger):
        """工厂构造函数。"""
        self._logger = logger.bind(factory="SemanticAnalysisStageFactory")

    def create(self,
               storage: StorageInterface,
               settings: SemanticAnalysisStageSettings, # <-- [指令] 类型已更新
               adapter: Optional[SemanticAdapterInterface] = None
               ) -> SemanticAnalysisStage:
        """
        创建并返回一个配置好的 `SemanticAnalysisStage` 实例。

        **参数**:
            storage (StorageInterface): 存储服务实例。
            settings (SemanticAnalysisStageSettings): 语义分析阶段的配置。
            adapter (Optional[SemanticAdapterInterface]): (可选) 自定义的语义分析适配器。
                                                        如果为 None，则创建默认的 `SentenceTransformerAdapter`。

        **返回**:
            SemanticAnalysisStage: 配置好的语义分析阶段实例。
        """
        self._logger.debug("开始创建 SemanticAnalysisStage 实例...")

        if adapter is None:
            self._logger.debug("未提供语义分析适配器，创建默认的 SentenceTransformerAdapter...")
            # **注意**: 适配器本身是无状态的，但 Stage 会调用其 load_model。
            adapter = SentenceTransformerAdapter() 
            self._logger.debug("默认 SentenceTransformerAdapter 创建成功。")

        stage_instance = SemanticAnalysisStage(
            logger=self._logger, 
            storage=storage,
            settings=settings,
            adapter=adapter
        )
        self._logger.success("SemanticAnalysisStage 实例创建成功。")

        return stage_instance