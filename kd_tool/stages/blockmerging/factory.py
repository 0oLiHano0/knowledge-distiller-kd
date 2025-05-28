"""
=================================================
factory.py - BlockMergerStage 工厂 (v4.6)
=================================================
---
"""
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.core.interfaces import StageInterface, StorageInterface
from kd_tool.stages.blockmerging.block_merging_stage import BlockMergerStage
from kd_tool.stages.blockmerging.settings_models import BlockMergerStageSettings


class BlockMergerStageFactory:
    """
    负责创建 BlockMergerStage 实例。
    """

    def __init__(self, logger: LoggerProtocol):
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
