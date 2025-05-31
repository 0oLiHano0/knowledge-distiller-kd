"""
=================================================
orchestrator.py - 流水线编排器核心 (v4.1)
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
import uuid
import time
import traceback
from typing import Any, Dict, List, Optional
from pathlib import Path
from collections import Counter
from kd_tool.logging.protocols import LoggerProtocol # kd_tool/logging/protocols.py 日志协议
from kd_tool.core.core_dtos import PipelineContextDTO
from kd_tool.core.core_settings_models import OrchestratorSettings
from kd_tool.core.interfaces import StageInterface
from kd_tool.core.errors import KDToolError


class OrchestratorError(KDToolError):
    """Orchestrator 操作相关的基本异常。"""

    def __init__(self, message: str, original_exception: Optional[Exception
        ]=None, **kwargs: Any):
        super().__init__(message, original_exception=original_exception,
            module='orchestrator', **kwargs)


class OrchestratorSettingsError(OrchestratorError):
    """当提供的 OrchestratorSettings 对象本身不符合要求时抛出。"""
    pass


class InvalidPipelineDefinitionError(OrchestratorError):
    """当 stage_modules 或 default_stage_order 的结构不合法时抛出。"""
    pass


class StageExecutionError(OrchestratorError):
    """当某个阶段模块在执行 process() 方法时发生未捕获的异常时，Orchestrator 可以包装并抛出此错误。"""

    def __init__(self, stage_name: str, original_exception: Exception,
        details: Optional[str]=None, **kwargs: Any):
        message = f"阶段 '{stage_name}' 执行失败。"
        if details:
            message += f' 详情: {details}'
        super().__init__(message, original_exception=original_exception,
            stage_name=stage_name, details=details, **kwargs)


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
        logger: LoggerProtocol
    ):
        """
        why: 依赖注入所有阶段、配置、日志。
        what: 初始化Orchestrator，保存依赖。
        how: 只做依赖赋值，不做业务逻辑。
        """
        self._stage_modules = stage_modules
        self._default_stage_order = default_stage_order
        self._settings = settings
        self._logger = logger

    def run(self, input_paths: List[Path]) -> PipelineContextDTO:
        """
        why: 统一入口，调度所有阶段。
        what: 依次调用各阶段的process方法。
        how: 只做调度，不做业务逻辑。
        """
        ...

    def _validate_pipeline_definition(self, stage_modules: Dict[str,
        StageInterface], default_stage_order: List[str]) ->None:
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
                    f'模块中已定义但在顺序中缺失: {sorted(list(missing_in_order))}')
            if missing_in_modules:
                errors.append(
                    f'顺序中存在但在模块中未定义: {sorted(list(missing_in_modules))}')
            raise InvalidPipelineDefinitionError('; '.join(errors))
        if len(default_stage_order) != len(ordered_stages_names):
            duplicates = [item for item, count in Counter(
                default_stage_order).items() if count > 1]
            raise InvalidPipelineDefinitionError(
                f'default_stage_order 中包含重复阶段: {duplicates}')
        for stage_name, module_instance in stage_modules.items():
            if not isinstance(module_instance, StageInterface):
                raise InvalidPipelineDefinitionError(
                    f"阶段模块 '{stage_name}' (类型: {type(module_instance).__name__}) 未实现 StageInterface。"
                    )
        self._logger.debug('流水线定义已通过校验。')

    def _determine_stages_to_run(self, stages_to_run_override: Optional[
        List[str]]=None, stages_to_skip_override: Optional[List[str]]=None
        ) ->List[str]:
        """
        **[指令]** 私有辅助方法，**必须** 根据默认顺序和运行时覆盖选项，确定实际要执行的阶段列表。
        **必须** 优先使用 `stages_to_run_override`。如果未提供，则从默认顺序中排除 `stages_to_skip_override`。
        **必须** 校验覆盖参数中的阶段名称是否存在。
        """
        all_defined_stages = set(self._stage_modules.keys())
        if stages_to_run_override is not None:
            if not isinstance(stages_to_run_override, list):
                raise InvalidPipelineDefinitionError(
                    'stages_to_run_override 必须是一个列表。')
            unknown = [s for s in stages_to_run_override if s not in
                all_defined_stages]
            if unknown:
                raise InvalidPipelineDefinitionError(
                    f'stages_to_run_override 中包含未定义阶段: {unknown}')
            self._logger.debug(f'使用运行时指定的阶段顺序: {stages_to_run_override}')
            return list(stages_to_run_override)
        effective_order = list(self._default_stage_order)
        if stages_to_skip_override is not None:
            if not isinstance(stages_to_skip_override, list):
                raise InvalidPipelineDefinitionError(
                    'stages_to_skip_override 必须是一个列表。')
            unknown = [s for s in stages_to_skip_override if s not in
                all_defined_stages]
            if unknown:
                raise InvalidPipelineDefinitionError(
                    f'stages_to_skip_override 中包含未定义阶段: {unknown}')
            effective_order = [s for s in effective_order if s not in
                stages_to_skip_override]
            self._logger.debug(f'跳过阶段后的执行顺序: {effective_order}')
        return effective_order

    def configure(self, **kwargs) ->None:
        """
        **[指令]** (未来实现) 运行时调整 Orchestrator 行为。
        **当前**：**必须** 保持未实现状态并抛出 `NotImplementedError`。
        """
        self._logger.warning('Orchestrator.configure() 方法当前未实现。')
        raise NotImplementedError('Orchestrator.configure() 尚未实现。')
