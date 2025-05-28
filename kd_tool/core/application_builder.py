"""
=================================================
application_builder.py - 应用程序构建器 (v4.6 - 新日志层集成)
=================================================

**模块功能**:

- **核心职责**: 作为组合根 (Composition Root)，创建和组装所有核心组件，包括使用新的日志层。
- **v4.6 核心变更**:
    - **[指令] 必须** 移除对 `core.logging_setup` 的引用。
    - **[指令] 必须** 引入并使用 `kd_tool.logging.LoggerFactory`。
    - **[指令] 必须** 使用 `kd_tool.logging.LoggerProtocol` 作为日志记录器的类型。
    - **[指令] 必须** 将 `LoggerProtocol` 实例正确注入到所有依赖项中。
    - **[指令] 必须** 确保所有注释为中文。

---
"""
import traceback
from typing import Dict, List, Optional
from pathlib import Path

# 新日志层导入
from kd_tool.logging import LoggerProtocol, LoggerFactory

# 核心接口和组件导入
from kd_tool.core.interfaces import StorageInterface, StageInterface
from kd_tool.core.config import AppConfig
from kd_tool.core.orchestrator import Orchestrator
from kd_tool.core.errors import KDToolError
from kd_tool.core.orchestrator_factory import OrchestratorFactory

# DTOs 导入
from kd_tool.schemas.dtos import PipelineContextDTO # 如果Application中需要，保留

# 存储工厂导入
from kd_tool.storage.storage_factory import StorageFactory

# 各阶段工厂导入
from kd_tool.stages.prefilter.prefilter_factory import PrefilterStageFactory
from kd_tool.stages.docprocessing.factory import DocumentProcessingStageFactory
from kd_tool.stages.blockmerging.factory import BlockMergerStageFactory
from kd_tool.stages.md5analysis.factory import MD5AnalysisStageFactory
from kd_tool.stages.simhash_analysis.factory import SimHashAnalysisStageFactory
from kd_tool.stages.semantic_analysis.factory import SemanticAnalysisStageFactory
from kd_tool.stages.decision.factory import DecisionStageFactory
from kd_tool.stages.cleanup.factory import CleanupStageFactory


# 占位符：实现或导入真实的配置加载逻辑
def load_config(path: str) -> AppConfig:
    """
    加载应用程序配置。
    【注意】这是一个占位符实现。您需要替换为从 YAML 或其他来源加载配置的真实逻辑。
    """
    print(f"警告：正在使用占位符 'load_config'。请实现从 {path} 加载配置的真实逻辑。")
    # 这里返回一个默认配置，仅用于演示结构
    return AppConfig()


class Application:
    """
    **[规范]** 代表可运行的 KD_Tool 应用程序实例。
    **必须** 包含 `Orchestrator` 和 `LoggerProtocol`。
    **必须** 提供运行默认流水线的方法。
    """

    def __init__(self, orchestrator: Orchestrator, logger: LoggerProtocol):
        """
        **[指令]** 构造函数。**必须** 接收 `Orchestrator` 和 `LoggerProtocol`。
        """
        self.orchestrator = orchestrator
        self.logger = logger

    def run_default_pipeline(self, input_paths_str: List[str]) -> None:
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
        self.logger.info(f'🚀 应用程序启动，开始运行默认流水线，输入路径: {input_paths_str}') #
        try:
            input_paths = [Path(p).resolve() for p in input_paths_str]
            self.logger.debug(f'解析后的输入路径: {[str(p) for p in input_paths]}') #

            context: PipelineContextDTO = self.orchestrator.run(input_paths)

            if context.errors:
                self.logger.error(f'⚠️ 流水线执行完成，但检测到 {len(context.errors)} 个错误：') #
                for i, error in enumerate(context.errors):
                    self.logger.error(f'  {i + 1}. [{type(error).__name__}] {error}') #
                    if error.original_exception:
                        # Loguru 的 opt(exception=...) 很好，但 Protocol 中没有。
                        # 我们使用 exception() 方法，如果存在的话，或者退回到 error()。
                        try:
                           # 假设 LoggerProtocol 有 exception 方法
                           self.logger.exception(f'     原始异常详情 (Error {i + 1}):') #
                        except AttributeError:
                           self.logger.error(f'     原始异常详情 (Error {i + 1}): {error.original_exception}') #

                raise KDToolError(
                    f'流水线执行完成，但包含 {len(context.errors)} 个错误。详情请查看日志。') #
            else:
                self.logger.info('✅ 流水线执行成功完成，未发现错误！') # 使用 info 或 success (如果 protocol 支持)
        except KDToolError as e:
            self.logger.error(f'❌ 流水线执行期间发生受控错误: {e}') #
            raise
        except Exception as e:
            self.logger.error(f'💥 流水线执行期间发生未捕获的严重错误: {e}') #
            raise KDToolError(f'未捕获的严重错误: {e}', original_exception=e) from e #


class ApplicationBuilder:
    """
    **[规范]** 负责构建 `Application` 实例。这是应用程序的组合根。
    """

    def __init__(self, config_path: str):
        """
        **[指令]** 初始化构建器。**必须** 加载配置并设置**新的**日志系统。
        """
        try:
            self._config: AppConfig = load_config(config_path)
            # --- 新的日志系统初始化 ---
            self._logger_factory = LoggerFactory(self._config.logging) #
            self._logger: LoggerProtocol = self._logger_factory.get_logger() #
            # --------------------------
            self._logger.info(f'ApplicationBuilder 初始化完成 (配置: {config_path})。') #
        except Exception as e:
            # 在日志系统完全可用前，使用 print 输出关键错误
            print(f'严重错误: ApplicationBuilder 初始化失败: {e}')
            traceback.print_exc()
            raise KDToolError(f'ApplicationBuilder 初始化失败: {e}',
                original_exception=e) from e #

        self._storage_instance: Optional[StorageInterface] = None

    def _get_storage_instance(self) -> StorageInterface:
        """
        **[指令]** 创建或获取存储服务实例。**必须** 保证单例。
        **必须** 注入 `LoggerProtocol`。
        """
        if self._storage_instance is None:
            self._logger.debug('正在创建存储服务实例...') #
            # [注意] StorageFactory 必须被修改以接受 LoggerProtocol
            storage_factory = StorageFactory(self._logger)
            self._storage_instance = storage_factory.create(self._config.storage)
            self._logger.info( #
                f'存储服务实例 ({self._config.storage.backend_type}) 创建成功。')
        return self._storage_instance

    def _create_stages(self) -> Dict[str, StageInterface]:
        """
        **[指令]** 使用各自的工厂创建所有已配置的 Stage 实例。
        **必须** 根据配置中的 `enabled` 标志决定是否创建 Stage。
        **必须** 使用配置中定义的键名。
        **必须** 注入 `LoggerProtocol`。
        """
        self._logger.debug('开始创建所有 Stage 实例...') #
        storage = self._get_storage_instance()
        stages: Dict[str, StageInterface] = {}

        # [注意] 所有 StageFactory 都必须被修改以接受 LoggerProtocol
        if self._config.prefilter.enabled:
            stages['prefilter'] = PrefilterStageFactory(self._logger).create(
                self._config.prefilter, storage)
        if self._config.document_processing.enabled:
            stages['document_processing'] = DocumentProcessingStageFactory(self
                ._logger).create(self._config.document_processing, storage)
        if self._config.block_merging.enabled:
            stages['block_merging'] = BlockMergerStageFactory(self._logger
                ).create(self._config.block_merging, storage)

        # 检查并使用正确的分析配置路径
        if self._config.md5_analysis.enabled:
             stages['md5_analysis'] = MD5AnalysisStageFactory(self._logger
                 ).create(self._config.md5_analysis, storage)
        if self._config.simhash_analysis.enabled:
             stages['simhash_analysis'] = SimHashAnalysisStageFactory(self.
                 _logger).create(storage, self._config.simhash_analysis)
        if self._config.semantic_analysis.enabled:
             stages['semantic_analysis'] = SemanticAnalysisStageFactory(self
                 ._logger).create(storage, self._config.semantic_analysis)

        if self._config.decision.enabled:
            stages['decision'] = DecisionStageFactory(self._logger).create(
                storage, self._config.decision)
        if self._config.cleanup.enabled:
            stages['cleanup'] = CleanupStageFactory(self._logger).create(
                storage, self._config.cleanup)

        self._logger.info( #
            f'成功创建 {len(stages)} 个 Stage 实例: {list(stages.keys())}。')
        return stages

    def build(self) -> Application:
        """
        **[指令]** 构建并返回完整的 `Application` 实例。
        **必须** 调用 `OrchestratorFactory` 并传递正确的参数（包括 `LoggerProtocol`）。
        """
        self._logger.info('开始构建 Application 实例...') #
        all_stages = self._create_stages()
        orchestrator_factory = OrchestratorFactory(self._logger)
        orchestrator = orchestrator_factory.create(stage_modules=all_stages,
            default_stage_order=self._config.orchestrator.default_stage_order, #
            settings=self._config.orchestrator)
        self._logger.info('Orchestrator 实例创建成功。') #
        app = Application(orchestrator, self._logger)
        self._logger.info('🏆 Application 实例构建成功！准备就绪。') #
        return app