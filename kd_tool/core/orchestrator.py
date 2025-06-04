"""
====================开发指引======================
kd_tool/core/orchestrator.py - v4.2
=================================================

**【文件定位】**  
- 路径：kd_tool/core/orchestrator.py
- 所属模块：核心服务层（Core Service Layer）
- 作用：全局流水线编排器，负责调度各阶段模块，管理上下文生命周期。

**【模块职责（SRP）】**  
- 唯一职责：作为 KD-Tool 的全局流程调度者，负责依次调用各阶段模块的 process 方法，管理 PipelineContextDTO 的生命周期，协调日志与错误处理。

**【依赖关系与注入】**  
- 依赖项：
    - stage_modules: Dict[str, StageInterface]（所有阶段模块实例，工厂注入）
    - default_stage_order: List[str]（默认阶段顺序，工厂注入）
    - settings: OrchestratorSettings（配置对象，工厂注入）
    - logger: LoggerProtocol（日志协议，工厂注入）
    - storage: StorageInterface（存储接口，工厂注入）
- 注入方式：全部通过构造函数注入，严禁内部实例化依赖。
- Mock点：logger、storage、stage_modules 可在单元测试中替换为 Mock 实现。

**【输入输出规范】**  
- Orchestrator.run(input_paths: List[Path]) -> PipelineContextDTO
    - 输入：input_paths（待处理文件路径列表）
    - 输出：PipelineContextDTO（包含 task_id、日志、阶段处理结果、错误等）
    - 异常：抛出 OrchestratorError 及其子类，所有异常均需结构化处理并记录日志。
- 仅通过 DTO（如 PipelineContextDTO）进行数据传递，严禁传递 ORM 实体。

**【核心架构约束】**  
- 严禁持久化状态，所有运行时上下文仅通过 context 传递。
- 严禁包含具体业务逻辑，仅负责调度与生命周期管理。
- 严禁直接依赖底层存储/模型，所有依赖均通过注入。
- 必须类型注解，所有方法参数/返回值均需类型提示。
- 日志上下文绑定仅在 run() 内部，禁止多次链式 bind。
- 重要类/方法（如 Orchestrator、run、_validate_pipeline_definition、_determine_stages_to_run）需添加三段式注释（WHY/WHAT/HOW）。
- 禁止直接实例化依赖、禁止业务逻辑与存储耦合。

**【接口与DTO规范】**  
- 关键接口：
    - Orchestrator（主类）
    - run(input_paths: List[Path]) -> PipelineContextDTO
    - configure(**kwargs) -> None（预留，当前未实现）
- DTO：
    - PipelineContextDTO（上下文传递对象，Pydantic定义）
    - OrchestratorSettings（配置对象，Pydantic定义）
- 异常类：
    - OrchestratorError 及其子类（如 StageExecutionError、InvalidPipelineDefinitionError）

**【日志与安全】**  
- 日志记录点：
    - 每次 run() 生成唯一 task_id，并通过 logger.bind 绑定上下文
    - 各阶段执行前后、异常捕获、流水线结束均需记录日志
    - 日志级别：info（流程事件）、error（已知异常）、exception（未知异常）、debug（校验/调试信息）、warning（未实现方法）
- 敏感信息处理：日志中不得输出敏感数据内容，仅记录元信息（如路径、task_id）。
- 权限/安全：Orchestrator 不涉及权限控制，安全约束由上层调用方保证。

**【任务清单】**  
1. [已完成] 依赖注入与无状态性设计
2. [已完成] 日志上下文绑定与日志协议合规
3. [已完成] DTO与ORM分离、类型注解
4. [已完成] 自定义异常体系与结构化错误处理
5. [已完成] 阶段顺序与定义校验
6. [已完成] 运行时阶段覆盖与跳过机制
7. [已完成] configure方法预留与未实现警告
8. [待完成] 单元测试用例完善（Mock依赖、异常分支、日志校验等）
9. [待完成] 工厂模式集成与依赖组装规范文档补充

**【其他说明】**  
- 未来如需扩展动态阶段插拔、流水线分支、并行执行等高级特性，需在工厂与配置层预留扩展点。
- 历史遗留：早期版本可能存在依赖实例化、日志污染等问题，已在当前版本修正。
- TODO：完善 orchestrator 相关的工厂实现与集成测试，确保全链路可插拔与可观测性。
"""

import uuid
import time
import traceback
from typing import Any, Dict, List, Optional
from pathlib import Path
from collections import Counter
from kd_tool.logging.protocols import (
    LoggerProtocol,
)  # kd_tool/logging/protocols.py 日志协议
from kd_tool.core.core_dtos import PipelineContextDTO
from kd_tool.core.core_settings_models import OrchestratorSettings
from kd_tool.core.interfaces import StageInterface
from kd_tool.core.errors import KDToolError
from kd_tool.storage.storage_interface import StorageInterface


class OrchestratorError(KDToolError):
    """Orchestrator 操作相关的基本异常。"""

    def __init__(
        self,
        message: str,
        original_exception: Optional[Exception] = None,
        **kwargs: Any,
    ):
        super().__init__(
            message,
            original_exception=original_exception,
            module="orchestrator",
            **kwargs,
        )


class OrchestratorSettingsError(OrchestratorError):
    """当提供的 OrchestratorSettings 对象本身不符合要求时抛出。"""

    pass


class InvalidPipelineDefinitionError(OrchestratorError):
    """当 stage_modules 或 default_stage_order 的结构不合法时抛出。"""

    pass


class StageExecutionError(OrchestratorError):
    """当某个阶段模块在执行 process() 方法时发生未捕获的异常时，Orchestrator 可以包装并抛出此错误。"""

    def __init__(
        self,
        stage_name: str,
        original_exception: Exception,
        details: Optional[str] = None,
        **kwargs: Any,
    ):
        message = f"阶段 '{stage_name}' 执行失败。"
        if details:
            message += f" 详情: {details}"
        super().__init__(
            message,
            original_exception=original_exception,
            stage_name=stage_name,
            details=details,
            **kwargs,
        )


class OrchestratorRuntimeError(OrchestratorError):
    """Orchestrator 内部（非阶段执行）的运行时错误。"""

    pass


class Orchestrator:
    """
    WHY: 负责全局流程调度，保持无状态。
    WHAT: 依赖注入所有阶段、配置、日志。
    HOW: 只做调度，不做业务逻辑。
    """

    def __init__(
        self,
        stage_modules: Dict[str, StageInterface],
        default_stage_order: List[str],
        settings: OrchestratorSettings,
        logger: LoggerProtocol,
        storage: StorageInterface,
    ):
        """
        why: 依赖注入所有阶段、配置、日志。
        what: 初始化Orchestrator，保存依赖。
        how: 只做依赖赋值，不做业务逻辑。
        """
        self._stage_modules = stage_modules
        self._default_stage_order = default_stage_order
        self._settings = settings
        self._logger = logger  # Orchestrator 自身的 logger
        self._storage = storage
        # PSEUDO: 初始化时可以进行流水线定义校验等
        self._validate_pipeline_definition(stage_modules, default_stage_order)

    def run(self, input_paths: List[Path]) -> PipelineContextDTO:
        """
        why: 统一入口，调度所有阶段。
        what: 依次调用各阶段的process方法，管理 PipelineContextDTO 的生命周期。
        how: 严格按照规范创建 PipelineContextDTO，并注入绑定的 logger 和 task_id。
        task id仅在此处生成，所有Stage通过context.task_id获取
        """
        # ARCHITECT_TODO: 步骤 1 - 为本次运行生成唯一的 task_id。
        #   规范: 必须是 UUID。
        current_task_id: uuid.UUID = uuid.uuid4()
        self._logger.info(
            f"Orchestrator: Starting new run with task_id: {current_task_id}"
        )  # Orchestrator自身logger记录

        # ARCHITECT_TODO: 步骤 2 - 基于 Orchestrator 的 _logger，为当前 task_id 创建一个已绑定的 run_logger。
        #   规范: 调用 self._logger.bind(task_id=str(current_task_id))。
        #   规范: 确保 task_id 转换为字符串形式进行绑定，因为日志 extra 通常是字符串键。
        run_specific_logger: LoggerProtocol = self._logger.bind(
            task_id=str(current_task_id)
        )
        run_specific_logger.info(
            "Orchestrator: Bound run_specific_logger with task_id."
        )  # 使用新绑定的logger确认

        # ARCHITECT_TODO: 步骤 3 - 创建 PipelineContextDTO 实例。
        #   规范: 必须传入步骤1生成的 current_task_id。
        #   规范: 必须传入步骤2生成的 run_specific_logger 作为 context.run_logger。
        #   规范: 必须传入 input_paths。
        #   规范: 其他字段（file_records, content_blocks 等）应使用其 default_factory 初始化为空。
        context = PipelineContextDTO(
            task_id=current_task_id,
            initial_input_paths=input_paths,
            run_logger=run_specific_logger,
            # errors, shared_data 等会自动使用 Pydantic 的 default_factory
        )
        context.run_logger.info(
            "Orchestrator: PipelineContextDTO created successfully."
        )

        # ARCHITECT_TODO: 步骤 4 - 根据配置和运行时覆盖，确定要执行的阶段列表。
        #   PSEUDO: stages_to_execute = self._determine_stages_to_run(...)
        stages_to_execute = self._default_stage_order  # 简化示例，实际应更复杂

        # ARCHITECT_TODO: 步骤 5 - 依次执行选定的阶段。
        #   规范: 每个阶段接收 context 对象，并返回处理后的 context 对象。
        #   规范: 使用 context.run_logger 进行阶段执行的日志记录。
        #   规范: 实现错误处理策略 (e.g., HALT_ON_FIRST_ERROR)。
        for stage_name in stages_to_execute:
            if stage_name in self._stage_modules:
                stage_instance = self._stage_modules[stage_name]
                context.run_logger.info(
                    f"Orchestrator: Executing stage '{stage_name}'..."
                )
                try:
                    context = stage_instance.process(context)
                    context.run_logger.info(
                        f"Orchestrator: Stage '{stage_name}' completed."
                    )
                except KDToolError as e:  # 假设 KDToolError 是项目的基础错误
                    context.run_logger.error(
                        f"Orchestrator: Error in stage '{stage_name}': {e}"
                    )
                    context.add_error(e)  # 将错误记录到 context 中
                    if self._settings.on_pipeline_error_policy == "HALT_ON_FIRST_ERROR":
                        context.run_logger.error(
                            f"Orchestrator: Halting pipeline due to error in stage '{stage_name}'."
                        )
                        break  # 停止执行后续阶段
                except Exception as e:
                    context.run_logger.exception(
                        f"Orchestrator: Unexpected critical error in stage '{stage_name}'."
                    )
                    # 包装为项目定义的错误类型
                    generic_error = StageExecutionError(
                        stage_name=stage_name,
                        original_exception=e,
                        details="Unexpected critical error during stage.process()",
                    )
                    context.add_error(generic_error)
                    if self._settings.on_pipeline_error_policy == "HALT_ON_FIRST_ERROR":
                        context.run_logger.error(
                            f"Orchestrator: Halting pipeline due to critical error in stage '{stage_name}'."
                        )
                        break
            else:
                # PSEUDO: 处理阶段未在模块中找到的情况
                missing_stage_error = InvalidPipelineDefinitionError(
                    f"Stage '{stage_name}' defined in order but not found in stage_modules."
                )
                context.run_logger.error(str(missing_stage_error))
                context.add_error(missing_stage_error)
                if self._settings.on_pipeline_error_policy == "HALT_ON_FIRST_ERROR":
                    break

        # ARCHITECT_TODO: 步骤 6 - 流水线执行完毕，返回最终的 context。
        context.run_logger.info("Orchestrator: Pipeline run finished.")
        self._storage.save_pipeline_context(context)
        return context

    def _validate_pipeline_definition(
        self, stage_modules: Dict[str, StageInterface], default_stage_order: List[str]
    ) -> None:
        """
        **[指令]** 私有辅助方法，**必须** 校验流水线定义的有效性。
        **必须** 检查：
        1. `stage_modules` 和 `default_stage_order` 的键名是否完全匹配。
        2. `default_stage_order` 中是否有重复项。
        3. `stage_modules` 中的实例是否实现了 `StageInterface`。
        """
        defined_stages_names = set(stage_modules.keys())
        ordered_stages_names = set(default_stage_order)
        if defined_stages_names != ordered_stages_names:
            missing_in_order = defined_stages_names - ordered_stages_names
            missing_in_modules = ordered_stages_names - defined_stages_names
            errors = []
            if missing_in_order:
                errors.append(
                    f"模块中已定义但在顺序中缺失: {sorted(list(missing_in_order))}"
                )
            if missing_in_modules:
                errors.append(
                    f"顺序中存在但在模块中未定义: {sorted(list(missing_in_modules))}"
                )
            raise InvalidPipelineDefinitionError("; ".join(errors))
        if len(default_stage_order) != len(ordered_stages_names):
            duplicates = [
                item
                for item, count in Counter(default_stage_order).items()
                if count > 1
            ]
            raise InvalidPipelineDefinitionError(
                f"default_stage_order 中包含重复阶段: {duplicates}"
            )
        for stage_name, module_instance in stage_modules.items():
            if not isinstance(module_instance, StageInterface):
                raise InvalidPipelineDefinitionError(
                    f"阶段模块 '{stage_name}' (类型: {type(module_instance).__name__}) 未实现 StageInterface。"
                )
        self._logger.debug("流水线定义已通过校验。")

    def _determine_stages_to_run(
        self,
        stages_to_run_override: Optional[List[str]] = None,
        stages_to_skip_override: Optional[List[str]] = None,
    ) -> List[str]:
        """
        **[指令]** 私有辅助方法，**必须** 根据默认顺序和运行时覆盖选项，确定实际要执行的阶段列表。
        **必须** 优先使用 `stages_to_run_override`。如果未提供，则从默认顺序中排除 `stages_to_skip_override`。
        **必须** 校验覆盖参数中的阶段名称是否存在。
        """
        all_defined_stages = set(self._stage_modules.keys())
        if stages_to_run_override is not None:
            if not isinstance(stages_to_run_override, list):
                raise InvalidPipelineDefinitionError(
                    "stages_to_run_override 必须是一个列表。"
                )
            unknown = [s for s in stages_to_run_override if s not in all_defined_stages]
            if unknown:
                raise InvalidPipelineDefinitionError(
                    f"stages_to_run_override 中包含未定义阶段: {unknown}"
                )
            self._logger.debug(f"使用运行时指定的阶段顺序: {stages_to_run_override}")
            return list(stages_to_run_override)
        effective_order = list(self._default_stage_order)
        if stages_to_skip_override is not None:
            if not isinstance(stages_to_skip_override, list):
                raise InvalidPipelineDefinitionError(
                    "stages_to_skip_override 必须是一个列表。"
                )
            unknown = [
                s for s in stages_to_skip_override if s not in all_defined_stages
            ]
            if unknown:
                raise InvalidPipelineDefinitionError(
                    f"stages_to_skip_override 中包含未定义阶段: {unknown}"
                )
            effective_order = [
                s for s in effective_order if s not in stages_to_skip_override
            ]
            self._logger.debug(f"跳过阶段后的执行顺序: {effective_order}")
        return effective_order

    def configure(self, **kwargs) -> None:
        """
        **[指令]** (未来实现) 运行时调整 Orchestrator 行为。
        **当前**：**必须** 保持未实现状态并抛出 `NotImplementedError`。
        """
        self._logger.warning("Orchestrator.configure() 方法当前未实现。")
        raise NotImplementedError("Orchestrator.configure() 尚未实现。")
