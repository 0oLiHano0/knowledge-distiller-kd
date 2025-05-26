
# docs/伪代码_kd_tool_v4.0/stages/docprocessing/factory.py

# -*- coding: utf-8 -*-

"""
=================================================
factory.py.md - DocumentProcessingStage 工厂伪代码
=================================================

**模块功能**:

- 负责创建并组装 DocumentProcessingStage 实例及其依赖。
- 遵循方案二，将工厂置于其对应的 Stage 目录中。

---
"""

# =============================================================================
# Imports (伪代码，仅示意)
# =============================================================================
from loguru import Logger

# --- 核心模块导入 ---
# [指令] StageInterface 和 StorageInterface 从 core 层导入
from ....core.interfaces import StageInterface, StorageInterface # 路径相对于 kd_tool/stages/docprocessing/

# --- Stage 实现导入 (本地导入) ---
from .document_processing_stage import DocumentProcessingStage

# --- [指令] 更新 DocumentProcessingStageSettings 的导入路径 ---
from .settings_models import DocumentProcessingStageSettings # <-- [指令] 已更新为本地导入


# =============================================================================
# DocumentProcessingStageFactory
# =============================================================================

class DocumentProcessingStageFactory:
    """
    负责创建 DocumentProcessingStage 实例。
    """

    def __init__(self, logger: Logger):
        """
        初始化文档处理阶段工厂。

        Args:
            logger: 日志记录器实例。
        """
        self._logger = logger.bind(factory_name="DocumentProcessingStageFactory")
        self._logger.info("DocumentProcessingStageFactory initialized.")

    def create(
        self,
        settings: DocumentProcessingStageSettings, # <-- [指令] 类型已更新为本地导入的模型
        storage: StorageInterface,
    ) -> StageInterface:
        """
        创建并返回一个配置好的 DocumentProcessingStage 实例。

        Args:
            settings: 文档处理阶段的配置 DTO。
            storage: 存储服务接口实例。

        Returns:
            一个实现了 StageInterface 的 DocumentProcessingStage 实例。
        """
        self._logger.info(f"Creating DocumentProcessingStage instance...")

        stage_instance = DocumentProcessingStage(
            logger=self._logger.bind(stage_name="DocumentProcessing"),
            settings=settings,
            storage=storage,
        )

        self._logger.success("DocumentProcessingStage instance created successfully.")
        return stage_instance
