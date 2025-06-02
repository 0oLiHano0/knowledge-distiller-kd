"""
=================================================
factory.py - BlockMergerStage 工厂 (v4.7)
=================================================

**模块功能**:

- 负责创建和配置 `BlockMergingStage` 实例。
- 与 Storage 解耦，仅依赖于 context 和 settings。

"""
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.core.interfaces import StageInterface
from kd_tool.stages.blockmerging.block_merging_stage import BlockMergingStage
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
        self._logger.info('BlockMergerStageFactory 初始化完成.')

    def create(self, settings: BlockMergerStageSettings) ->StageInterface:
        """
        创建并返回一个配置好的 BlockMergerStage 实例。
        """
        self._logger.info(f'创建 BlockMergerStage 实例...')
        stage_instance = BlockMergingStage(logger=self._logger.bind(
            stage_name='BlockMerger'), settings=settings)
        self._logger.success('BlockMergerStage 实例创建成功.')
        return stage_instance
