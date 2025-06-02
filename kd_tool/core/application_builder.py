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
from pydantic import BaseModel

# 新日志层导入
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.logging.factory import LoggerFactory

# 核心接口和组件导入
from kd_tool.core.interfaces import StageInterface
from kd_tool.core.config import AppConfig
from kd_tool.core.orchestrator import Orchestrator
from kd_tool.core.errors import KDToolError
from kd_tool.core.orchestrator_factory import OrchestratorFactory

# DTOs 导入
from kd_tool.core.core_dtos import PipelineContextDTO # 如果Application中需要，保留

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

from kd_tool.storage.storage_interface import StorageInterface

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
    负责构建 Application 实例，所有依赖均通过注入或工厂模式创建。
    """

    def __init__(
        self,
        config_path: str,
        config: AppConfig,
        logger_factory: LoggerFactory,
        logger: LoggerProtocol,
        storage_factory: StorageFactory,
        orchestrator_factory: OrchestratorFactory,
        # 可扩展更多工厂
    ):
        """
        why: 支持所有依赖注入，便于测试和扩展。
        what: 允许外部传入 config、logger、factory。
        how: 通过工厂和依赖注入实现解耦。
        """
        self._config = config
        self._logger_factory = logger_factory
        self._logger = logger
        self._storage_factory = storage_factory
        self._orchestrator_factory = orchestrator_factory
        # ...其他工厂同理
        self._storage_instance: Optional[StorageInterface] = None

    def _get_storage_instance(self) -> StorageInterface:
        """
        why: 保证存储服务单例，依赖注入。
        what: 通过工厂创建 StorageInterface。
        how: 只在首次调用时创建。
        """
        if self._storage_instance is None:
            self._storage_instance = self._storage_factory.create(self._config.storage)
        return self._storage_instance

    def _create_stages(self) -> Dict[str, StageInterface]:
        """
        why: 通过工厂创建所有阶段，禁止直接实例化依赖。
        what: 返回所有已启用的阶段实例。
        how: 依次通过各自工厂创建。
        """
        storage = self._get_storage_instance()
        stages: Dict[str, StageInterface]
        stages = {}
        if self._config.prefilter.enabled:
            stages['prefilter'] = PrefilterStageFactory(self._logger).create(self._config.prefilter, storage)
        if self._config.document_processing.enabled:
            stages['document_processing'] = DocumentProcessingStageFactory(self._logger).create(self._config.document_processing, storage)
        if self._config.block_merging.enabled:
            stages['block_merging'] = BlockMergerStageFactory(self._logger).create(self._config.block_merging, storage)
        if self._config.md5_analysis.enabled:
            stages['md5_analysis'] = MD5AnalysisStageFactory(self._logger).create(self._config.md5_analysis, storage)
        if self._config.simhash_analysis.enabled:
            stages['simhash_analysis'] = SimHashAnalysisStageFactory(self._logger).create(storage, self._config.simhash_analysis)
        if self._config.semantic_analysis.enabled:
            stages['semantic_analysis'] = SemanticAnalysisStageFactory(self._logger).create(storage, self._config.semantic_analysis)
        if self._config.decision.enabled:
            stages['decision'] = DecisionStageFactory(self._logger).create(storage, self._config.decision)
        if self._config.cleanup.enabled:
            stages['cleanup'] = CleanupStageFactory(self._logger).create(storage, self._config.cleanup)
        return stages

    def build(self) -> Application:
        """
        why: 组装 Application，所有依赖均通过工厂和注入获得。
        what: 返回完整 Application 实例。
        how: 组装 orchestrator 和 logger。
        """
        stages = self._create_stages()
        orchestrator = self._orchestrator_factory.create(
            stage_modules=stages,
            default_stage_order=self._config.orchestrator.default_stage_order,
            settings=self._config.orchestrator
        )
        return Application(orchestrator, self._logger)