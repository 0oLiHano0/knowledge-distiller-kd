```python

# docs/伪代码_kd_tool_v4.0/stages/blockmerging/factory.py

# -*- coding: utf-8 -*-

"""
=================================================
factory.py.md - BlockMergerStage 工厂伪代码
=================================================

**模块功能**:

- 负责创建并组装 BlockMergerStage 实例及其依赖。
- 遵循方案二，将工厂置于其对应的 Stage 目录中。

---
"""

# =============================================================================
# Imports (伪代码，仅示意)
# =============================================================================
from loguru import Logger

from kd_tool.core.interfaces import StageInterface, StorageInterface
from kd_tool.schemas.settings_models import BlockMergerStageSettings # 假设存在
from kd_tool.stages.blockmerging.block_merger_stage import BlockMergerStage # 假设存在

# =============================================================================
# BlockMergerStageFactory
# =============================================================================

class BlockMergerStageFactory:
    """
    负责创建 BlockMergerStage 实例。
    """

    def __init__(self, logger: Logger):
        """
        初始化块合并阶段工厂。

        Args:
            logger: 日志记录器实例。
        """
        self._logger = logger.bind(factory_name="BlockMergerStageFactory")
        self._logger.info("BlockMergerStageFactory initialized.")

    def create(
        self,
        settings: BlockMergerStageSettings,
        storage: StorageInterface,
    ) -> StageInterface:
        """
        创建并返回一个配置好的 BlockMergerStage 实例。

        Args:
            settings: 块合并阶段的配置 DTO。
            storage: 存储服务接口实例。

        Returns:
            一个实现了 StageInterface 的 BlockMergerStage 实例。
        """
        self._logger.info(f"Creating BlockMergerStage instance...")

        stage_instance = BlockMergerStage(
            logger=self._logger.bind(stage_name="BlockMerger"),
            settings=settings,
            storage=storage,
        )

        self._logger.success("BlockMergerStage instance created successfully.")
        return stage_instance

```