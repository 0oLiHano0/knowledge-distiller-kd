# kd_tool/core/application_builder.py (v4.5.1 + Context 修复版)
# -*- coding: utf-8 -*-

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

from loguru import Logger
from typing import Dict, List
from pathlib import Path  # <-- [指令] 必须 导入 Path

# 导入核心接口和组件
from .interfaces import StorageInterface, StageInterface
from .config import AppConfig, load_config
from .orchestrator import Orchestrator
from .logging_setup import setup_logging
from .errors import KDToolError # <-- [指令] 导入 KDToolError

# [指令] 必须 从 schemas 导入 PipelineContextDTO
from kd_tool.schemas.dtos import PipelineContextDTO

# 导入存储工厂
from kd_tool.storage.f03.storage_factory import StorageFactory

# [指令] 必须 导入所有需要的 Stage 工厂
from kd_tool.stages.prefilter.p04.prefilter_factory import PrefilterStageFactory
from kd_tool.stages.docprocessing.factory import DocumentProcessingStageFactory
from kd_tool.stages.blockmerging.factory import BlockMergerStageFactory
from kd_tool.stages.md5analysis.factory import MD5AnalysisStageFactory
from kd_tool.stages.simhash_analysis.factory import SimHashAnalysisStageFactory
from kd_tool.stages.semantic_analysis.factory import SemanticAnalysisStageFactory
from kd_tool.stages.decision.factory import DecisionStageFactory
from kd_tool.stages.cleanup.factory import CleanupStageFactory

# [指令] 必须 导入 OrchestratorFactory
from .f02.orchestrator_factory import OrchestratorFactory

# ==============================
# == Application 类定义 ==
# ==============================

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
        self.logger.info(f"🚀 应用程序启动，开始运行默认流水线，输入路径: {input_paths_str}")
        try:
            # [指令] 必须 将字符串路径转换为 Path 对象。
            input_paths = [Path(p).resolve() for p in input_paths_str] # 使用 resolve 获取绝对路径
            self.logger.debug(f"解析后的输入路径: {[str(p) for p in input_paths]}")

            # [指令] 必须 调用 orchestrator.run 并传递 Path 对象。
            context: PipelineContextDTO = self.orchestrator.run(input_paths)

            # [指令] 必须 检查返回的 context 中的错误。
            if context.errors:
                self.logger.error(f"⚠️ 流水线执行完成，但检测到 {len(context.errors)} 个错误：")
                for i, error in enumerate(context.errors):
                    self.logger.error(f"  {i+1}. [{type(error).__name__}] {error}")
                    # [指令] 必须 记录原始异常的堆栈信息 (如果存在且日志级别允许)。
                    if error.original_exception:
                         # 使用 exception 方法记录完整的堆栈信息
                         self.logger.opt(exception=error.original_exception).debug(
                             f"     原始异常详情 (Error {i+1}):"
                         )
                # [指令] 即使有错误，也认为 *流程* 走完了，但可以抛出特定异常或返回状态。
                # 当前决策：记录错误，但不抛出，让 CLI 判断为"完成但有错"。
                # 如果需要 CLI 明确知道失败，可以在这里抛出一个 'PipelineWithErrorsError'。
                # self.logger.warning("流水线执行已完成，但包含错误，请检查日志。")
                # **修改**: 为了让 CLI 更明确，如果存在错误，我们抛出一个异常。
                raise KDToolError(f"流水线执行完成，但包含 {len(context.errors)} 个错误。详情请查看日志。")
            else:
                self.logger.success("✅ 流水线执行成功完成，未发现错误！")

        except KDToolError as e:
            # [指令] 捕获我们自己的错误，记录并重新抛出，让 CLI 处理。
            self.logger.critical(f"❌ 流水线执行期间发生受控错误: {e}", exc_info=True)
            raise # 重新抛出给 CLI
        except Exception as e:
            # [指令] 捕获所有其他未知错误，记录并包装后抛出。
            self.logger.critical(f"💥 流水线执行期间发生未捕获的严重错误: {e}", exc_info=True)
            raise KDToolError(f"未捕获的严重错误: {e}", original_exception=e) from e

# ==============================
# == ApplicationBuilder 类定义 ==
# ==============================

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
            self._logger.info(f"ApplicationBuilder 初始化完成 (配置: {config_path})。")
        except Exception as e:
            # [指令] 在日志系统设置或配置加载失败时，必须处理并抛出。
            print(f"CRITICAL: ApplicationBuilder 初始化失败: {e}")
            traceback.print_exc()
            raise KDToolError(f"ApplicationBuilder 初始化失败: {e}", original_exception=e) from e

        self._storage_instance: Optional[StorageInterface] = None

    def _get_storage_instance(self) -> StorageInterface:
        """
        **[指令]** 创建或获取存储服务实例。**必须** 保证单例。
        """
        if self._storage_instance is None:
            self._logger.debug("正在创建存储服务实例...")
            storage_factory = StorageFactory(self._logger)
            self._storage_instance = storage_factory.create(self._config.storage)
            self._logger.info(f"存储服务实例 ({self._config.storage.backend_type}) 创建成功。")
        return self._storage_instance

    def _create_stages(self) -> Dict[str, StageInterface]:
        """
        **[指令]** 使用各自的工厂创建所有已配置的 Stage 实例。
        **必须** 根据配置中的 `enabled` 标志决定是否创建 Stage。
        **必须** 使用配置中定义的键名 (`prefilter`, `document_processing` 等)。
        """
        self._logger.debug("开始创建所有 Stage 实例...")
        storage = self._get_storage_instance()
        stages: Dict[str, StageInterface] = {}

        # [指令] 必须 按顺序检查并创建每个 Stage (如果启用)。
        # Prefilter Stage
        if self._config.prefilter.enabled:
            stages["prefilter"] = PrefilterStageFactory(self._logger).create(
                self._config.prefilter, storage
            )
        # Document Processing Stage
        if self._config.document_processing.enabled:
            stages["document_processing"] = DocumentProcessingStageFactory(self._logger).create(
                self._config.document_processing, storage
            )
        # Block Merging Stage
        if self._config.block_merging.enabled:
             stages["block_merging"] = BlockMergerStageFactory(self._logger).create(
                 self._config.block_merging, storage
             )
        # MD5 Analysis Stage
        if self._config.analysis.md5.enabled:
            stages["md5_analysis"] = MD5AnalysisStageFactory(self._logger).create(
                self._config.analysis.md5, storage
            )
        # SimHash Analysis Stage
        if self._config.analysis.simhash.enabled:
            stages["simhash_analysis"] = SimHashAnalysisStageFactory(self._logger).create(
                storage, self._config.analysis.simhash
            )
        # Semantic Analysis Stage
        if self._config.analysis.semantic.enabled:
            stages["semantic_analysis"] = SemanticAnalysisStageFactory(self._logger).create(
                storage, self._config.analysis.semantic
            )
        # Decision Stage
        if self._config.decision.enabled:
            stages["decision"] = DecisionStageFactory(self._logger).create(
                storage, self._config.decision
            )
        # Cleanup Stage
        if self._config.cleanup.enabled:
            stages["cleanup"] = CleanupStageFactory(self._logger).create(
                storage, self._config.cleanup
            )

        self._logger.info(f"成功创建 {len(stages)} 个 Stage 实例: {list(stages.keys())}。")
        return stages


    def build(self) -> Application:
        """
        **[指令]** 构建并返回完整的 `Application` 实例。
        **必须** 调用 `OrchestratorFactory` 并传递正确的参数。
        """
        self._logger.info("开始构建 Application 实例...")

        all_stages = self._create_stages()
        orchestrator_factory = OrchestratorFactory(self._logger)

        # [指令] 必须 从配置中获取 default_stage_order 并传递给工厂。
        # [指令] 必须 确保传递的 stage_modules 中的键与 default_stage_order 兼容 (Orchestrator 会校验)。
        # [指令] 必须 只传递 OrchestratorFactory.create 需要的参数。
        orchestrator = orchestrator_factory.create(
            stage_modules=all_stages,
            default_stage_order=self._config.orchestrator.default_stage_order,
            settings=self._config.orchestrator
        )
        self._logger.info("Orchestrator 实例创建成功。")

        # [指令] 必须 创建 Application 实例并注入 Orchestrator 和 Logger。
        app = Application(orchestrator, self._logger)
        self._logger.success("🏆 Application 实例构建成功！准备就绪。")

        return app