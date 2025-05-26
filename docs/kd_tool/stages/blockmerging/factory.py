"""
=================================================
factory.py.md - BlockMergerStage 工厂 (v4.6)
=================================================
... (模块注释保持不变) ...
---
"""
from loguru import Logger
from ....core.interfaces import StageInterface, StorageInterface
from kd_tool.stages.blockmerging.block_merging_stage import BlockMergerStage
from kd_tool.stages.blockmerging.settings_models import BlockMergerStageSettings


class BlockMergerStageFactory:
    """
    负责创建 BlockMergerStage 实例。
    """

    def __init__(self, logger: Logger):
        """
        初始化块合并阶段工厂。
        """
        self._logger = logger.bind(factory_name='BlockMergerStageFactory')
        self._logger.info('BlockMergerStageFactory initialized.')

    def create(self, settings: BlockMergerStageSettings, storage:
        StorageInterface) ->StageInterface:
        """
        创建并返回一个配置好的 BlockMergerStage 实例。
        """
        self._logger.info(f'Creating BlockMergerStage instance...')
        stage_instance = BlockMergerStage(logger=self._logger.bind(
            stage_name='BlockMerger'), settings=settings, storage=storage)
        self._logger.success('BlockMergerStage instance created successfully.')
        return stage_instance
