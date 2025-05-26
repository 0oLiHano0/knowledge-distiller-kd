# kd_tool/core/orchestrator.py
# kd_tool/core/orchestrator.py (v4.1 - PipelineContextDTO 集成版)
# -*- coding: utf-8 -*-

"""
=================================================
c03.Orchestrator.py.md - 流水线编排器核心 (v4.1)
=================================================

**模块功能**:

- **核心职责**: 作为流水线调度引擎，根据配置 (`OrchestratorSettings`) 和输入，
              按顺序 (`default_stage_order` 或覆盖) 执行各个处理阶段 (`StageInterface`)。
- **技术实现**: 管理流水线生命周期，创建并传递 `PipelineContextDTO`，处理阶段错误，记录关键信息。
- **v4.1 核心变更**:
    - **[指令] 必须** 使用 `PipelineContextDTO` 作为流水线数据和状态的 **唯一** 载体。
    - **[指令] 必须** 调整 `run` 方法以创建和管理 `PipelineContextDTO`。
    - **[指令] 必须** 确保 `StageInterface.process` 的调用签名与接口定义一致。
    - **[指令] 必须** 废弃旧的 `OutputResult` 和基于 `Dict[str, Any]` 的上下文。

**架构约束**:
- **[规范] 必须** 遵循依赖注入原则，其依赖 (`Logger`, `Settings`, `Stages`) **必须** 由工厂注入。
- **[规范] 必须** 保持无状态。每次 `run` 调用都是独立的。
- **[规范] 严禁** 包含任何具体阶段的业务逻辑。
- **[规范] 必须** 负责宏观调度和错误处理策略的实施。

---
"""

# --- Python 标准库及第三方库导入 ---
import uuid
import time
import traceback
from typing import Any, Dict, List, Optional
from pathlib import Path  # <-- [指令] 必须 导入 Path
from collections import Counter

# --- Loguru 日志库导入 ---
from loguru import Logger

# --- 项目内部模块导入 ---
# [指令] 必须从同级目录下的 core_settings_models.py 导入 OrchestratorSettings
from .core_settings_models import OrchestratorSettings
# [指令] 必须从同级目录下的 core_dtos.py 导入 PipelineContextDTO
from .core_dtos import PipelineContextDTO  # <-- [指令] 必须 使用 PipelineContextDTO


# [指令] 必须 从 core 导入核心接口和错误基类
from kd_tool.core.interfaces import StageInterface
from kd_tool.core.errors import KDToolError

# ==============================
# == Orchestrator 自身错误定义 ==
# ==============================
# [规范] 这些错误定义保持不变，但确保继承自 KDToolError。

class OrchestratorError(KDToolError):
    """Orchestrator 操作相关的基本异常。"""
    def __init__(self, message: str, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, original_exception=original_exception, module="orchestrator", **kwargs)

class OrchestratorSettingsError(OrchestratorError):
    """当提供的 OrchestratorSettings 对象本身不符合要求时抛出。"""
    pass # 保持简洁

class InvalidPipelineDefinitionError(OrchestratorError):
    """当 stage_modules 或 default_stage_order 的结构不合法时抛出。"""
    pass # 保持简洁

class StageExecutionError(OrchestratorError):
    """当某个阶段模块在执行 process() 方法时发生未捕获的异常时，Orchestrator 可以包装并抛出此错误。"""
    def __init__(self, stage_name: str, original_exception: Exception, details: Optional[str] = None, **kwargs: Any):
        message = f"阶段 '{stage_name}' 执行失败。"
        if details: message += f" 详情: {details}"
        super().__init__(message, original_exception=original_exception, stage_name=stage_name, details=details, **kwargs)

class OrchestratorRuntimeError(OrchestratorError):
    """Orchestrator 内部（非阶段执行）的运行时错误。"""
    pass # 保持简洁

# ==============================
# == Orchestrator 类定义 ==
# ==============================

class Orchestrator:
    """
    流水线编排器。
    负责管理和执行一系列定义好的处理阶段（Stage）。
    """

    def __init__(self, *,
                 stage_modules: Dict[str, StageInterface],
                 default_stage_order: List[str],
                 settings: OrchestratorSettings,
                 logger: Logger):
        """
        **[指令]** 构造 Orchestrator 实例。**必须** 通过依赖注入接收所有参数。

        **参数**:
            stage_modules (Dict[str, StageInterface]): **[必须]** 阶段名称到 `StageInterface` 实例的字典。
            default_stage_order (List[str]): **[必须]** 默认阶段执行顺序列表。列表中的名称 **必须** 存在于 `stage_modules` 中。
            settings (OrchestratorSettings): **[必须]** `Orchestrator` 的配置设置实例。
            logger (Logger): **[必须]** 用于日志记录的 `Logger` 实例。

        **可能抛出的异常**:
            TypeError: 如果参数类型不正确。
            InvalidPipelineDefinitionError: 如果 `stage_modules` 或 `default_stage_order` 定义不合法。
        """
        # --- [指令] 执行严格的参数类型和结构校验 ---
        if not isinstance(settings, OrchestratorSettings):
            raise TypeError("参数 'settings' 必须是 OrchestratorSettings 的实例。")
        if not hasattr(logger, 'bind'): # 简化检查，实际应更严格
            raise TypeError("参数 'logger' 必须是一个兼容 Loguru 的 Logger 实例。")
        if not isinstance(stage_modules, dict) or not stage_modules:
            raise InvalidPipelineDefinitionError("stage_modules 必须是一个非空的字典。")
        if not isinstance(default_stage_order, list) or not default_stage_order:
            raise InvalidPipelineDefinitionError("default_stage_order 必须是一个非空的列表。")

        # --- [指令] 执行流水线定义的逻辑一致性校验 ---
        self._validate_pipeline_definition(stage_modules, default_stage_order)

        # --- [指令] 存储注入的依赖 ---
        self._stage_modules: Dict[str, StageInterface] = stage_modules
        self._default_stage_order: List[str] = default_stage_order
        self._settings: OrchestratorSettings = settings
        # [指令] logger 必须绑定 'Orchestrator' 组件上下文
        self._logger: Logger = logger.bind(component="Orchestrator")

        self._logger.info(
            f"Orchestrator 已初始化。默认阶段顺序: {self._default_stage_order}。 "
            f"错误处理策略: {self._settings.on_pipeline_error_policy}。"
        )

    def _validate_pipeline_definition(self,
                                      stage_modules: Dict[str, StageInterface],
                                      default_stage_order: List[str]) -> None:
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
            if missing_in_order: errors.append(f"模块中已定义但在顺序中缺失: {sorted(list(missing_in_order))}")
            if missing_in_modules: errors.append(f"顺序中存在但在模块中未定义: {sorted(list(missing_in_modules))}")
            raise InvalidPipelineDefinitionError("; ".join(errors))

        if len(default_stage_order) != len(ordered_stages_names):
            duplicates = [item for item, count in Counter(default_stage_order).items() if count > 1]
            raise InvalidPipelineDefinitionError(f"default_stage_order 中包含重复阶段: {duplicates}")

        for stage_name, module_instance in stage_modules.items():
            if not isinstance(module_instance, StageInterface):
                raise InvalidPipelineDefinitionError(
                    f"阶段模块 '{stage_name}' (类型: {type(module_instance).__name__}) 未实现 StageInterface。"
                )
        self._logger.debug("流水线定义已通过校验。")


    def _determine_stages_to_run(
        self,
        stages_to_run_override: Optional[List[str]] = None,
        stages_to_skip_override: Optional[List[str]] = None
    ) -> List[str]:
        """
        **[指令]** 私有辅助方法，**必须** 根据默认顺序和运行时覆盖选项，确定实际要执行的阶段列表。
        **必须** 优先使用 `stages_to_run_override`。如果未提供，则从默认顺序中排除 `stages_to_skip_override`。
        **必须** 校验覆盖参数中的阶段名称是否存在。
        """
        all_defined_stages = set(self._stage_modules.keys())

        if stages_to_run_override is not None:
            if not isinstance(stages_to_run_override, list):
                raise InvalidPipelineDefinitionError("stages_to_run_override 必须是一个列表。")
            unknown = [s for s in stages_to_run_override if s not in all_defined_stages]
            if unknown: raise InvalidPipelineDefinitionError(f"stages_to_run_override 中包含未定义阶段: {unknown}")
            self._logger.debug(f"使用运行时指定的阶段顺序: {stages_to_run_override}")
            return list(stages_to_run_override)

        effective_order = list(self._default_stage_order)
        if stages_to_skip_override is not None:
            if not isinstance(stages_to_skip_override, list):
                raise InvalidPipelineDefinitionError("stages_to_skip_override 必须是一个列表。")
            unknown = [s for s in stages_to_skip_override if s not in all_defined_stages]
            if unknown: raise InvalidPipelineDefinitionError(f"stages_to_skip_override 中包含未定义阶段: {unknown}")
            effective_order = [s for s in effective_order if s not in stages_to_skip_override]
            self._logger.debug(f"跳过阶段后的执行顺序: {effective_order}")

        return effective_order

    def run(self,
            input_paths: List[Path],  # <-- [指令] 输入参数必须是 List[Path]
            stages_to_run: Optional[List[str]] = None,
            stages_to_skip: Optional[List[str]] = None,
            initial_shared_data: Optional[Dict[str, Any]] = None
            ) -> PipelineContextDTO:  # <-- [指令] 返回类型必须是 PipelineContextDTO
        """
        **[指令]** 执行完整处理流水线。这是 `Orchestrator` 的核心方法。

        **参数**:
            input_paths (List[Path]): **[必须]** 流水线的初始输入文件或目录路径列表。
            stages_to_run (Optional[List[str]]): **[可选]** 覆盖默认顺序，只运行指定阶段。
            stages_to_skip (Optional[List[str]]): **[可选]** 从默认顺序中跳过指定阶段 (若提供 `stages_to_run` 则忽略)。
            initial_shared_data (Optional[Dict[str, Any]]): **[可选]** 注入到 `PipelineContextDTO.shared_data` 的初始数据。

        **返回**:
            PipelineContextDTO: **[必须]** 包含流水线执行结果（数据和错误）的最终上下文对象。
        """
        # --- [指令] 1. 初始化运行环境 ---
        task_id_uuid = uuid.uuid4()
        # [指令] task_id 字符串格式必须使用配置的前缀和 UUID 的 hex 表示
        task_id_str = f"{self._settings.default_task_id_prefix}{task_id_uuid.hex}"
        # [指令] 必须为本次运行创建绑定了 task_id 的 Logger
        run_logger = self._logger.bind(task_id=task_id_str)

        run_logger.info(f"流水线运行启动 (ID: {task_id_str})。输入路径: {[str(p) for p in input_paths]}。")
        total_start_time = time.monotonic()
        stage_durations: Dict[str, float] = {}

        # --- [指令] 2. 创建 PipelineContextDTO ---
        # [指令] 必须创建 PipelineContextDTO 实例，并注入 task_id, run_logger, initial_input_paths。
        context = PipelineContextDTO(
            task_id=task_id_uuid,
            initial_input_paths=input_paths,
            run_logger=run_logger,
            shared_data=(initial_shared_data or {})
        )

        try:
            # --- [指令] 3. 确定执行阶段 ---
            active_stages = self._determine_stages_to_run(stages_to_run, stages_to_skip)
            run_logger.info(f"计划执行的阶段顺序: {active_stages}")

            if not active_stages:
                run_logger.warning("没有阶段需要执行。流水线提前结束。")
                return context # [指令] 如果无阶段执行，必须返回当前 context

            # --- [指令] 4. 按顺序执行阶段 ---
            for stage_name in active_stages:
                stage_instance = self._stage_modules[stage_name]
                run_logger.info(f"▶️ 阶段 '{stage_name}' 开始执行。")
                stage_start_time = time.monotonic()
                stage_failed = False

                try:
                    # [指令] 必须调用 stage_instance.process(context)，并用其返回值更新 context。
                    context = stage_instance.process(context=context)
                    # [指令] 必须检查返回的 context 是否为 PipelineContextDTO 类型。
                    if not isinstance(context, PipelineContextDTO):
                         raise OrchestratorRuntimeError(
                             f"阶段 '{stage_name}' 未返回 PipelineContextDTO 实例，而是返回了 {type(context).__name__}。"
                         )

                except KDToolError as stage_kd_exc:
                    # [指令] 必须捕获 KDToolError，记录错误，并将错误添加到 context.errors。
                    run_logger.error(f"❌ 阶段 '{stage_name}' 发生受控错误: {stage_kd_exc}", exc_info=True)
                    context.add_error(stage_kd_exc)
                    stage_failed = True
                except Exception as stage_exc:
                    # [指令] 必须捕获所有其他异常，记录错误，包装为 StageExecutionError，并添加到 context.errors。
                    run_logger.exception(f"💥 阶段 '{stage_name}' 发生未预料的错误: {stage_exc}")
                    wrapped_error = StageExecutionError(stage_name=stage_name, original_exception=stage_exc)
                    context.add_error(wrapped_error)
                    stage_failed = True

                finally:
                    # [指令] 必须记录每个阶段的耗时。
                    stage_duration = time.monotonic() - stage_start_time
                    stage_durations[stage_name] = round(stage_duration, 4)
                    if stage_failed:
                         run_logger.error(f"⏹️ 阶段 '{stage_name}' 失败。耗时: {stage_duration:.4f} 秒。")
                    else:
                         run_logger.success(f"✅ 阶段 '{stage_name}' 成功完成。耗时: {stage_duration:.4f} 秒。")

                # --- [指令] 5. 执行错误处理策略 ---
                if stage_failed and self._settings.on_pipeline_error_policy == 'HALT_ON_FIRST_ERROR':
                    run_logger.critical(f"🛑 错误策略为 HALT_ON_FIRST_ERROR，流水线在阶段 '{stage_name}' 中止。")
                    break # [指令] 必须中断循环

            # --- [指令] 6. 最终处理与返回 ---
            context.shared_data["stage_durations"] = stage_durations # 将耗时记录到 context
            return context # [指令] 必须返回最终的 PipelineContextDTO

        except InvalidPipelineDefinitionError as e:
            run_logger.critical(f"🚨 流水线定义无效，无法执行: {e}", exc_info=False)
            context.add_error(e)
            return context # [指令] 必须返回包含错误的 context
        except Exception as e:
            run_logger.critical(f"🚨 Orchestrator 发生意外顶层错误: {e}", exc_info=True)
            context.add_error(OrchestratorRuntimeError(f"Orchestrator 顶层错误: {e}", original_exception=e))
            return context # [指令] 必须返回包含错误的 context
        finally:
            total_duration_seconds = time.monotonic() - total_start_time
            run_logger.info(f"🏁 流水线运行结束 (ID: {task_id_str})。总耗时: {total_duration_seconds:.4f} 秒。发现 {len(context.errors)} 个错误。")

    def configure(self, **kwargs) -> None:
        """
        **[指令]** (未来实现) 运行时调整 Orchestrator 行为。
        **当前**：**必须** 保持未实现状态并抛出 `NotImplementedError`。
        """
        self._logger.warning("Orchestrator.configure() 方法当前未实现。")
        raise NotImplementedError("Orchestrator.configure() 尚未实现。")

# == 5. 测试要求 ==
# - 单元测试：覆盖 __init__, run 调度逻辑、异常场景等
# - 集成测试：使用真实模块进行端到端测试

# == 7. 代码质量与文档 ==
# - 文档字符串覆盖所有公共接口和模型定义
# - 异常情况抛出相应 OrchestratorError 子类
# - context 顶层键及类型文档说明

# == 8. 日志管理 ==
# - 错误日志 (ERROR) 包含阶段名称、异常、堆栈信息
# - INFO 级别记录总流程和各阶段开始/结束及耗时
# - 使用绑定 task_id 的 Logger 确保可追踪性