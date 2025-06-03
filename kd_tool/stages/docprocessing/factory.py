"""
=================================================
factory.py - DocumentProcessingStage 工厂 (v4.7)
=================================================

**模块功能**:

- 负责创建并组装 DocumentProcessingStage 实例及其依赖。
- 与 Storage 解耦，仅依赖于 context 和 settings。

"""
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.core.interfaces import StageInterface
from kd_tool.stages.docprocessing.document_processing_stage import DocumentProcessingStage
from kd_tool.stages.docprocessing.settings_models import DocumentProcessingStageSettings
from kd_tool.stages.docprocessing.adapter_interface import ParserAdapterInterface
from kd_tool.stages.docprocessing.unstructured_adapter import UnstructuredParserAdapter

class DocumentProcessingStageFactory:
    """
    负责创建 DocumentProcessingStage 实例。
    """

    def __init__(self, logger: LoggerProtocol):
        """
        初始化文档处理阶段工厂。

        Args:
            logger: 日志记录器实例。
        """
        self._logger = logger.bind(factory_name=
            'DocumentProcessingStageFactory')
        self._logger.info('DocumentProcessingStageFactory 初始化完成.')

    def _build_parser_adapter(self, parser_type: str, parsing_strategy: str) -> ParserAdapterInterface:
        """
        WHY: 工厂方法，按parser_type组装解析器
        WHAT: 返回ParserAdapterInterface实例
        HOW: 可扩展多种解析器
        """
        if parser_type == "unstructured":
            return UnstructuredParserAdapter(self._logger.bind(component="UnstructuredParserAdapter"))
        elif parser_type == "pdfplumber":
            # TODO: 实现 PDFPlumberParserAdapter 并注入
            raise NotImplementedError("PDFPlumberParserAdapter 尚未实现")
        raise ValueError(f"未知解析器类型: {parser_type}")

    def create(self, settings: DocumentProcessingStageSettings) -> StageInterface:
        """
        创建并返回一个配置好的 DocumentProcessingStage 实例。

        Args:
            settings: 文档处理阶段的配置 DTO。
            storage: 存储服务接口实例。

        Returns:
            一个实现了 StageInterface 的 DocumentProcessingStage 实例。
        """
        self._logger.info(f'创建 DocumentProcessingStage 实例...')
        parser_adapter = self._build_parser_adapter(settings.parser_type, settings.parsing_strategy)
        stage_instance = DocumentProcessingStage(
            logger=self._logger.bind(stage_name='DocumentProcessing'),
            settings=settings,
            parser_adapter=parser_adapter
        )
        self._logger.success('DocumentProcessingStage 实例创建成功.')
        return stage_instance
