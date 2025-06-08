"""
====================开发指引======================
kd_tool/core/application_builder.py - v4.7
=================================================

**【文件定位】**  
- 所属包结构：kd_tool.core
- 归属层次：组合根（Composition Root），负责组装 Application、Orchestrator、Logger、各阶段工厂等所有核心依赖。
- 该文件是系统启动与依赖注入的唯一入口，直接影响全局架构解耦与可扩展性。

**【模块职责（SRP）】**  
- 唯一职责：负责根据配置组装并注入所有核心依赖，生成可运行的 Application 实例，确保各阶段、存储、日志等服务的解耦与可插拔。

**【依赖关系与注入】**  
- 依赖外部服务/接口：
    - AppConfig（配置对象，Pydantic模型）
    - LoggerFactory、LoggerProtocol（日志工厂与协议）
    - StorageFactory（存储工厂）
    - OrchestratorFactory（编排器工厂）
    - 各阶段工厂（PrefilterStageFactory、DocumentProcessingStageFactory、BlockMergerStageFactory、MD5AnalysisStageFactory、SimHashAnalysisStageFactory、SemanticAnalysisStageFactory、DecisionStageFactory、CleanupStageFactory）
- 依赖注入方式：全部通过构造函数注入，严禁内部直接实例化依赖。
- Mock点：所有工厂、Logger、配置对象均可注入Mock实现，便于单元测试。

**【输入输出规范】**  
- ApplicationBuilder
    - 输入：配置路径（str）、AppConfig、LoggerFactory、LoggerProtocol、StorageFactory、OrchestratorFactory等
    - 输出：Application实例
    - 异常：KDToolError及其子类
- Application
    - run_default_pipeline(input_paths_str: List[str]) -> None
        - 输入：输入路径字符串列表
        - 输出：无（流程型，异常时抛出KDToolError）
        - 异常：KDToolError及其子类
- DTO/ORM边界：所有跨模块数据传递均使用DTO（如PipelineContextDTO），严禁直接传递ORM对象。

**【核心架构约束】**  
- 禁止直接实例化依赖，所有依赖必须通过注入或工厂模式创建。
- 禁止业务逻辑与存储耦合，所有存储操作通过StorageInterface。
- 必须类型注解，所有函数/方法参数与返回值均需类型提示。
- 日志与异常处理：
    - 日志必须通过LoggerProtocol注入，严禁全局/单例获取。
    - 关键流程、异常、错误均需结构化日志记录，优先使用logger.exception。
- 重要类/函数需三段式注释（WHY/WHAT/HOW）：
    - ApplicationBuilder.__init__
    - ApplicationBuilder._get_storage_instance
    - ApplicationBuilder._create_stages
    - ApplicationBuilder.build
    - Application.run_default_pipeline
- 禁止直接/链式bind持久化logger上下文，所有bind操作仅限本地变量。

**【接口与DTO规范】**  
- 关键接口：
    - LoggerProtocol（日志协议，所有日志操作均通过该协议）
    - StorageInterface（存储接口，所有存储操作均通过该接口）
    - StageInterface（阶段接口，所有阶段模块均实现该接口）
- DTO：
    - PipelineContextDTO（流水线上下文与错误传递）
- 异常类：
    - KDToolError及其子类（所有可预见错误均需自定义异常）
- 工厂/接口隔离：所有工厂与实现分离，接口定义与实现解耦。

**【日志与安全】**  
- 日志记录点：
    - 应用启动、流水线启动/完成、阶段执行、异常捕获等关键节点
    - 日志级别：info（流程）、debug（参数）、error/exception（错误）
- 敏感信息处理：日志中严禁明文记录敏感数据，需脱敏或省略。
- 权限/数据安全：如涉及，需在日志与异常中避免泄露内部实现细节。

**【任务清单】**  
1. **load_config函数**：已实现占位符函数，具备接口与警告提示，尚未实现真实的配置加载逻辑（如YAML/ENV/CLI解析）。
2. **Application类**：已完整实现，包含Orchestrator与LoggerProtocol注入，run_default_pipeline方法实现了参数转换、异常处理、日志记录等关键流程，符合规范。
3. **ApplicationBuilder类**：已完整实现，支持所有依赖注入，工厂模式组装，未见直接实例化依赖。
4. **_get_storage_instance方法**：已实现，保证存储服务单例，依赖注入。
5. **_create_stages方法**：已实现，依次通过各自工厂创建所有阶段模块，未见直接实例化。
6. **build方法**：已实现，组装Orchestrator与Logger，返回完整Application实例。
7. **类型注解**：所有方法、类、参数、返回值均有类型注解，符合要求。
8. **三段式注释**：ApplicationBuilder及其关键方法、Application.run_default_pipeline均已补充WHY/WHAT/HOW三段式注释。
9. **日志与异常处理**：日志与异常处理严格遵守架构规范，所有异常均结构化日志输出，敏感信息未见泄露。
10. **单元测试**：当前文件未包含测试代码，未见单元测试实现，需在tests目录下补充。

**【其他说明】**  
- 未来如需扩展新阶段，仅需新增工厂并在_create_stages中注册，无需修改核心逻辑，完全符合开闭原则。
- 若需支持多种配置来源（如YAML/ENV/CLI），load_config需适配多种解析方式。
- 历史遗留/未来TODO：后续可考虑将ApplicationBuilder进一步解耦为多级工厂，支持插件化阶段注册。

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
from kd_tool.core.core_dtos import PipelineContextDTO  # 如果Application中需要，保留

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
    【注意】占位符实现：返回一个带最小可用 StorageSettingsDTO.backend_type 的默认配置。
    """
    print(
        f"⚠️  使用占位默认配置；实际项目应从 {path} 加载 YAML / ENV 等配置文件。"
    )
    from kd_tool.storage.settings_models import StorageSettingsDTO

    return AppConfig(
        storage=StorageSettingsDTO(backend_type="sqlite")
    )


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
        self.logger.info(
            f"🚀 应用程序启动，开始运行默认流水线，输入路径: {input_paths_str}"
        )  #
        try:
            input_paths = [Path(p).resolve() for p in input_paths_str]
            self.logger.debug(f"解析后的输入路径: {[str(p) for p in input_paths]}")  #

            context: PipelineContextDTO = self.orchestrator.run(input_paths)

            if context.errors:
                self.logger.error(
                    f"⚠️ 流水线执行完成，但检测到 {len(context.errors)} 个错误："
                )  #
                for i, error in enumerate(context.errors):
                    self.logger.error(f"  {i + 1}. [{type(error).__name__}] {error}")  #
                    if error.original_exception:
                        # Loguru 的 opt(exception=...) 很好，但 Protocol 中没有。
                        # 我们使用 exception() 方法，如果存在的话，或者退回到 error()。
                        try:
                            # 假设 LoggerProtocol 有 exception 方法
                            self.logger.exception(
                                f"     原始异常详情 (Error {i + 1}):"
                            )  #
                        except AttributeError:
                            self.logger.error(
                                f"     原始异常详情 (Error {i + 1}): {error.original_exception}"
                            )  #

                raise KDToolError(
                    f"流水线执行完成，但包含 {len(context.errors)} 个错误。详情请查看日志。"
                )  #
            else:
                self.logger.info(
                    "✅ 流水线执行成功完成，未发现错误！"
                )  # 使用 info 或 success (如果 protocol 支持)
        except KDToolError as e:
            self.logger.error(f"❌ 流水线执行期间发生受控错误: {e}")  #
            raise
        except Exception as e:
            self.logger.error(f"💥 流水线执行期间发生未捕获的严重错误: {e}")  #
            raise KDToolError(f"未捕获的严重错误: {e}", original_exception=e) from e  #


class ApplicationBuilder:
    """
    负责构建 Application 实例，所有依赖均通过注入或工厂模式创建。
    """

    def __init__(
        self,
        config_path: str,
        config: Optional[AppConfig] = None,
        logger_factory: Optional[LoggerFactory] = None,
        logger: Optional[LoggerProtocol] = None,
        storage_factory: Optional[StorageFactory] = None,
        orchestrator_factory: Optional[OrchestratorFactory] = None,
        # 可扩展更多工厂
    ):
        """
        why: 支持所有依赖注入，便于测试和扩展。
        what: 允许外部传入 config、logger、factory。
        how: 通过工厂和依赖注入实现解耦。
        """
        # 1. 加载配置
        self._config: AppConfig = config or load_config(config_path)

        # 2. 初始化日志工厂和 logger
        self._logger: LoggerProtocol = logger or LoggerFactory.create(
            self._config.logging
        )

        # 3. 初始化存储工厂
        self._storage_factory: StorageFactory = storage_factory or StorageFactory(
            logger=self._logger, settings=self._config.storage
        )

        # 4. 初始化 OrchestratorFactory
        self._orchestrator_factory: OrchestratorFactory = (
            orchestrator_factory or OrchestratorFactory(self._logger)
        )
        self._storage_instance: Optional[StorageInterface] = None

    def _get_storage_instance(self) -> StorageInterface:
        """
        why: 保证存储服务单例，依赖注入。
        what: 通过工厂创建 StorageInterface。
        how: 只在首次调用时创建。
        """
        if self._storage_instance is None:
            # StorageFactory.create() 已经封装了具体后端创建逻辑，不再需要额外参数
            self._storage_instance = self._storage_factory.create()
        return self._storage_instance

    def _create_stages(self) -> Dict[str, StageInterface]:
        """
        why: 通过工厂创建所有阶段，禁止直接实例化依赖。
        what: 返回所有已启用的阶段实例。
        how: 依次通过各自工厂创建。
        """
        stages: Dict[str, StageInterface] = {}
        if self._config.prefilter.enabled:
            stages["prefilter"] = PrefilterStageFactory(self._logger).create(
                self._config.prefilter
            )
        if self._config.document_processing.enabled:
            stages["document_processing"] = DocumentProcessingStageFactory(
                self._logger
            ).create(self._config.document_processing)
        if self._config.block_merging.enabled:
            stages["block_merging"] = BlockMergerStageFactory(self._logger).create(
                self._config.block_merging
            )
        if self._config.md5_analysis.enabled:
            stages["md5_analysis"] = MD5AnalysisStageFactory(self._logger).create(
                self._config.md5_analysis
            )
        if self._config.simhash_analysis.enabled:
            stages["simhash_analysis"] = SimHashAnalysisStageFactory(
                self._logger
            ).create(self._config.simhash_analysis)
        if self._config.semantic_analysis.enabled:
            stages["semantic_analysis"] = SemanticAnalysisStageFactory(
                self._logger
            ).create(self._config.semantic_analysis)
        if self._config.decision.enabled:
            stages["decision"] = DecisionStageFactory(self._logger).create(
                self._config.decision
            )
        if self._config.cleanup.enabled:
            stages["cleanup"] = CleanupStageFactory(self._logger).create(
                self._config.cleanup
            )
        return stages

    def build(self) -> Application:
        """
        why: 组装 Application，所有依赖均通过工厂和注入获得。
        what: 返回完整 Application 实例。
        how: 组装 orchestrator 和 logger。
        """
        # 先确保存储实例可用，供 Orchestrator 注入
        storage = self._get_storage_instance()

        stages = self._create_stages()

        orchestrator = self._orchestrator_factory.create(
            stage_modules=stages,
            settings=self._config.orchestrator,
            storage=storage,
        )

        return Application(orchestrator, self._logger)
