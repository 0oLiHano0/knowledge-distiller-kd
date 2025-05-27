"""
=================================================
c02.application_builder.py.md - 应用程序构建器 (v4.5.1 + Context 修复)
=================================================

**模块功能**:

- **核心职责**: 作为组合根，创建和组装所有核心组件。
- **v4.x 核心变更**:
    - **[指令] 必须** 正确调用 `OrchestratorFactory.create`。
    - **[指令] 必须** 定义 `Application` 类以处理 `Orchestrator.run` 返回的 `PipelineContextDTO`。
    - **[指令] 必须** 实现从字符串路径到 `Path` 对象的转换。
    - **[指令] 必须** 处理 `PipelineContextDTO` 中的错误并提供适当的日志反馈。

---
"""
import traceback
from loguru import Logger
from typing import Dict, List, Optional
from pathlib import Path
from kd_tool.core.interfaces import StorageInterface, StageInterface
from kd_tool.core.config import AppConfig, load_config
from kd_tool.core.orchestrator import Orchestrator
from kd_tool.core.logging_setup import setup_logging
from kd_tool.core.errors import KDToolError
from kd_tool.schemas.dtos import PipelineContextDTO
from kd_tool.storage.storage_factory import StorageFactory
from kd_tool.stages.prefilter.prefilter_factory import PrefilterStageFactory
from kd_tool.stages.docprocessing.factory import DocumentProcessingStageFactory
from kd_tool.stages.blockmerging.factory import BlockMergerStageFactory
from kd_tool.stages.md5analysis.factory import MD5AnalysisStageFactory
from kd_tool.stages.simhash_analysis.factory import SimHashAnalysisStageFactory
from kd_tool.stages.semantic_analysis.factory import SemanticAnalysisStageFactory
from kd_tool.stages.decision.factory import DecisionStageFactory
from kd_tool.stages.cleanup.factory import CleanupStageFactory
from kd_tool.core.orchestrator_factory import OrchestratorFactory


class Application:
    """
    **[规范]** 代表可运行的 KD_Tool 应用程序实例。
    **必须** 包含 `Orchestrator` 和 `Logger`。
    **必须** 提供运行默认流水线的方法。
    """

    def __init__(self, orchestrator: Orchestrator, logger: Logger):
        """
        **[指令]** 构造函数。**必须** 接收 `Orchestrator` 和 `Logger`。
        """
        self.orchestrator = orchestrator
        self.logger = logger

    def run_default_pipeline(self, input_paths_str: List[str]) ->None:
        """
        **[指令]** 运行默认配置的流水线。
        **必须** 接收字符串路径列表，并将其转换为 `Path` 对象。
        **必须** 调用 `Orchestrator.run` 并接收 `PipelineContextDTO`。
        **必须** 检查 `PipelineContextDTO.errors` 并记录日志。
        **必须** 在发生严重错误时抛出异常。

        **参数**:
            input_paths_str (List[str]): **[必须]** 来自 CLI 的输入路径字符串列表。

        **可能抛出的异常**:
            KDToolError 或其子类: 如果流水线执行中发生无法恢复的错误。
        """
        self.logger.info(f'🚀 应用程序启动，开始运行默认流水线，输入路径: {input_paths_str}')
        try:
            input_paths = [Path(p).resolve() for p in input_paths_str]
            self.logger.debug(f'解析后的输入路径: {[str(p) for p in input_paths]}')
            context: PipelineContextDTO = self.orchestrator.run(input_paths)
            if context.errors:
                self.logger.error(f'⚠️ 流水线执行完成，但检测到 {len(context.errors)} 个错误：'
                    )
                for i, error in enumerate(context.errors):
                    self.logger.error(
                        f'  {i + 1}. [{type(error).__name__}] {error}')
                    if error.original_exception:
                        self.logger.opt(exception=error.original_exception
                            ).debug(f'     原始异常详情 (Error {i + 1}):')
                raise KDToolError(
                    f'流水线执行完成，但包含 {len(context.errors)} 个错误。详情请查看日志。')
            else:
                self.logger.success('✅ 流水线执行成功完成，未发现错误！')
        except KDToolError as e:
            self.logger.critical(f'❌ 流水线执行期间发生受控错误: {e}', exc_info=True)
            raise
        except Exception as e:
            self.logger.critical(f'�� 流水线执行期间发生未捕获的严重错误: {e}', exc_info=True)
            raise KDToolError(f'未捕获的严重错误: {e}', original_exception=e) from e


class ApplicationBuilder:
    """
    **[规范]** 负责构建 `Application` 实例。这是应用程序的组合根。
    """

    def __init__(self, config_path: str):
        """
        **[指令]** 初始化构建器。**必须** 加载配置并设置日志系统。
        """
        try:
            self._config: AppConfig = load_config(config_path)
            self._logger: Logger = setup_logging(self._config.logging)
            self._logger.info(f'ApplicationBuilder 初始化完成 (配置: {config_path})。')
        except Exception as e:
            print(f'CRITICAL: ApplicationBuilder 初始化失败: {e}')
            traceback.print_exc()
            raise KDToolError(f'ApplicationBuilder 初始化失败: {e}',
                original_exception=e) from e
        self._storage_instance: Optional[StorageInterface] = None

    def _get_storage_instance(self) ->StorageInterface:
        """
        **[指令]** 创建或获取存储服务实例。**必须** 保证单例。
        """
        if self._storage_instance is None:
            self._logger.debug('正在创建存储服务实例...')
            storage_factory = StorageFactory(self._logger)
            self._storage_instance = storage_factory.create(self._config.
                storage)
            self._logger.info(
                f'存储服务实例 ({self._config.storage.backend_type}) 创建成功。')
        return self._storage_instance

    def _create_stages(self) ->Dict[str, StageInterface]:
        """
        **[指令]** 使用各自的工厂创建所有已配置的 Stage 实例。
        **必须** 根据配置中的 `enabled` 标志决定是否创建 Stage。
        **必须** 使用配置中定义的键名 (`prefilter`, `document_processing` 等)。
        """
        self._logger.debug('开始创建所有 Stage 实例...')
        storage = self._get_storage_instance()
        stages: Dict[str, StageInterface] = {}
        if self._config.prefilter.enabled:
            stages['prefilter'] = PrefilterStageFactory(self._logger).create(
                self._config.prefilter, storage)
        if self._config.document_processing.enabled:
            stages['document_processing'] = DocumentProcessingStageFactory(self
                ._logger).create(self._config.document_processing, storage)
        if self._config.block_merging.enabled:
            stages['block_merging'] = BlockMergerStageFactory(self._logger
                ).create(self._config.block_merging, storage)
        if self._config.analysis.md5.enabled:
            stages['md5_analysis'] = MD5AnalysisStageFactory(self._logger
                ).create(self._config.analysis.md5, storage)
        if self._config.analysis.simhash.enabled:
            stages['simhash_analysis'] = SimHashAnalysisStageFactory(self.
                _logger).create(storage, self._config.analysis.simhash)
        if self._config.analysis.semantic.enabled:
            stages['semantic_analysis'] = SemanticAnalysisStageFactory(self
                ._logger).create(storage, self._config.analysis.semantic)
        if self._config.decision.enabled:
            stages['decision'] = DecisionStageFactory(self._logger).create(
                storage, self._config.decision)
        if self._config.cleanup.enabled:
            stages['cleanup'] = CleanupStageFactory(self._logger).create(
                storage, self._config.cleanup)
        self._logger.info(
            f'成功创建 {len(stages)} 个 Stage 实例: {list(stages.keys())}。')
        return stages

    def build(self) ->Application:
        """
        **[指令]** 构建并返回完整的 `Application` 实例。
        **必须** 调用 `OrchestratorFactory` 并传递正确的参数。
        """
        self._logger.info('开始构建 Application 实例...')
        all_stages = self._create_stages()
        orchestrator_factory = OrchestratorFactory(self._logger)
        orchestrator = orchestrator_factory.create(stage_modules=all_stages,
            default_stage_order=self._config.orchestrator.
            default_stage_order, settings=self._config.orchestrator)
        self._logger.info('Orchestrator 实例创建成功。')
        app = Application(orchestrator, self._logger)
        self._logger.success('🏆 Application 实例构建成功！准备就绪。')
        return app
