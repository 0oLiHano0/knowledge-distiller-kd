"""
=================================================
factory.py.md - DocumentProcessingStage 工厂伪代码
=================================================

**模块功能**:

- 负责创建并组装 DocumentProcessingStage 实例及其依赖。
- 遵循方案二，将工厂置于其对应的 Stage 目录中。

---
"""
from loguru import Logger
from ....core.interfaces import StageInterface, StorageInterface
from kd_tool.stages.docprocessing.document_processing_stage import DocumentProcessingStage
from kd_tool.stages.docprocessing.settings_models import DocumentProcessingStageSettings


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
        self._logger = logger.bind(factory_name=
            'DocumentProcessingStageFactory')
        self._logger.info('DocumentProcessingStageFactory initialized.')

    def create(self, settings: DocumentProcessingStageSettings, storage:
        StorageInterface) ->StageInterface:
        """
        创建并返回一个配置好的 DocumentProcessingStage 实例。

        Args:
            settings: 文档处理阶段的配置 DTO。
            storage: 存储服务接口实例。

        Returns:
            一个实现了 StageInterface 的 DocumentProcessingStage 实例。
        """
        self._logger.info(f'Creating DocumentProcessingStage instance...')
        stage_instance = DocumentProcessingStage(logger=self._logger.bind(
            stage_name='DocumentProcessing'), settings=settings, storage=
            storage)
        self._logger.success(
            'DocumentProcessingStage instance created successfully.')
        return stage_instance
