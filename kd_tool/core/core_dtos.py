"""
====================开发指引======================
kd_tool/core/core_dtos.py - v4.8
=================================================

**【文件定位】**  
- 路径：kd_tool/core/core_dtos.py
- 所属：核心服务层（core），为 Orchestrator 及各 Stage 提供全局数据上下文与关键 DTO。
- 依赖：kd_tool.logging.protocols、kd_tool.schemas.dtos、kd_tool.schemas.enums、kd_tool.core.errors、Pydantic、Pathlib、UUID。

**【模块职责（SRP）】**  
- 唯一职责：定义并管理 Orchestrator 流水线全局上下文的数据传输对象（PipelineContextDTO），作为数据与状态的唯一载体，贯穿各 Stage。

**【依赖关系与注入】**  
- 依赖外部日志协议 LoggerProtocol（run_logger 字段），通过构造器注入，严禁内部实例化。
- 依赖 DTO（FileRecordDTO、ContentBlockDTO、AnalysisResultDTO、UserDecisionDTO）、枚举（AnalysisType）、自定义异常（KDToolError），均为绝对导入。
- 不依赖底层存储或业务服务，符合解耦要求。
- 如需 Mock，LoggerProtocol 可被替换为 MockLogger。

**【输入输出规范】**  
- PipelineContextDTO 字段及方法参数均有类型注解，详见下表：

| 字段/方法名 | 类型 | 说明 |
|-------------|------|------|
| task_id | UUID | 本次流水线执行唯一标识 |
| initial_input_paths | List[Path] | 初始输入路径列表 |
| run_logger | LoggerProtocol | 绑定 task_id 的日志记录器 |
| file_records | Dict[str, FileRecordDTO] | 文件记录，file_id 为键 |
| content_blocks | Dict[str, ContentBlockDTO] | 内容块，block_id 为键 |
| analysis_results | Dict[str, Dict[AnalysisType, List[AnalysisResultDTO]]] | 分析结果 |
| user_decisions | Dict[str, UserDecisionDTO] | 用户决策，pair_analysis_id 为键 |
| errors | List[KDToolError] | 错误收集 |
| shared_data | Dict[str, Any] | 阶段间临时共享数据 |

- 主要方法：
    - get_task_id_str(self) -> str
    - add_error(self, error: KDToolError) -> None
    - add_file_record(self, record: FileRecordDTO) -> None
    - add_content_block(self, block: ContentBlockDTO) -> None
    - add_analysis_result(self, result: AnalysisResultDTO) -> None
    - add_user_decision(self, decision: UserDecisionDTO) -> None
    - get_content_blocks_for_analysis(self, analysis_type: AnalysisType, force_reprocess: bool = False) -> List[ContentBlockDTO]

- 异常类型：仅允许 KDToolError 及其子类，禁止使用 Exception 或 None 传递错误。

**【核心架构约束】**  
- 禁止直接实例化依赖，所有依赖通过注入。
- 禁止业务逻辑与存储耦合，仅为数据容器。
- 所有字段、方法参数与返回值必须类型注解。
- 重要方法（如 add_error、add_analysis_result、get_content_blocks_for_analysis）需补充三段式注释（WHY/WHAT/HOW）。
- 仅允许通过 DTO 传递数据，禁止 ORM 直传。
- 绝对导入，禁止相对导入。

**【接口与DTO规范】**  
- 仅暴露 PipelineContextDTO 及其方法。
- DTO、枚举、协议等均通过绝对路径导入，接口与实现分离。
- 不直接暴露底层实现细节。

**【日志与安全】**  
- 日志记录点：所有 add 方法、异常捕获、分析流程关键节点。
- 日志级别：trace/debug/info/error，异常必须用 error 级别。
- 敏感信息不得写入日志。
- 日志上下文绑定：run_logger 必须已绑定 task_id。

**【任务清单】**  
1. 检查并补充 PipelineContextDTO 字段的类型注解与描述，确保类型安全与文档完整。
2. 检查 run_logger 注入方式，确保无内部实例化，符合依赖注入规范。
3. 补充/完善 add_error、add_analysis_result、get_content_blocks_for_analysis 等关键方法的三段式注释（WHY/WHAT/HOW）。
4. 检查所有 add/get 方法的边界与异常处理，确保错误收集与日志记录完整。
5. 编写/完善单元测试，覆盖所有方法的正常与异常路径，重点 Mock LoggerProtocol 验证日志行为。
6. 检查所有导入语句，确保为绝对导入，符合架构规范。
7. 审查日志输出，确保无敏感信息泄露，run_logger 已绑定 task_id。
8. 评估 shared_data 字段的使用范围，补充警告与使用建议，防止滥用。
9. 未来如需扩展字段或功能，需通过工厂/接口实现，禁止直接修改核心逻辑。

**【其他说明】**  
- 该 DTO 仅为数据载体，严禁添加任何业务逻辑。
- 未来如需扩展字段，需评估对下游兼容性影响。
- 需定期与架构规范同步，防止技术债累积。
"""

import uuid
from uuid import UUID, uuid4
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from kd_tool.logging.protocols import (
    LoggerProtocol,
)  # kd_tool/logging/protocols.py 日志协议
from kd_tool.schemas.dtos import (
    FileRecordDTO,
    ContentBlockDTO,
    AnalysisResultDTO,
    UserDecisionDTO,
)
from kd_tool.schemas.enums import AnalysisType
from kd_tool.core.errors import KDToolError


class PipelineContextDTO(BaseModel):
    """
    管道上下文数据传输对象 (Pipeline Context DTO)。
    **核心职责**:
    - 在 Orchestrator 控制的流水线中，作为数据和状态的载体，从一个 Stage 传递到下一个 Stage。
    - 全局唯一上下文，唯一持有task_id，作为流水线执行的唯一标识符。task id 为uuid，仅在此维护。
    **规范**:
    - 它的生命周期与一次 `Orchestrator.run` 调用绑定。
    - **[架构指令] 严禁**将其设计为有状态对象；它只是数据的容器。
    - Stage **必须**通过更新此对象来传递结果。
    - **[架构指令 v4.7] 必须** 包含 `task_id` 和 `run_logger`，作为任务的唯一标识和日志记录器。
    """

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, arbitrary_types_allowed=True
    )
    task_id: UUID = Field(
        default_factory=uuid4, description="本次流水线执行的唯一标识符 (UUID)。"
    )
    initial_input_paths: List[Path] = Field(
        default_factory=list, description="本次任务初始请求处理的输入路径列表。"
    )
    run_logger: LoggerProtocol = Field(
        description="与当前任务绑定的日志记录器。 **规范**: 此 logger **必须**已绑定 `task_id`。"
    )
    file_records: Dict[str, FileRecordDTO] = Field(
        default_factory=dict,
        description="处理过程中涉及的文件记录 (FileRecordDTO)，以 file_id 为键。",
    )
    content_blocks: Dict[str, ContentBlockDTO] = Field(
        default_factory=dict,
        description="从文件中提取的内容块 (ContentBlockDTO)，以 block_id 为键。",
    )
    analysis_results: Dict[str, Dict[AnalysisType, List[AnalysisResultDTO]]] = Field(
        default_factory=dict,
        description="分析结果 (AnalysisResultDTO)。结构: {block_id: {AnalysisType: [AnalysisResultDTO]}}",
    )
    user_decisions: Dict[str, UserDecisionDTO] = Field(
        default_factory=dict,
        description="用户决策 (UserDecisionDTO)，以 pair_analysis_id 为键。",
    )
    errors: List[KDToolError] = Field(
        default_factory=list, description="在流水线处理过程中收集到的所有错误。"
    )
    shared_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="用于阶段间共享临时或非结构化数据的区域。**警告**: 应谨慎使用！",
    )

    def get_task_id_str(self) -> str:
        """获取字符串格式的 task_id，常用于日志或文件名。"""
        return str(self.task_id)

    def add_error(self, error: KDToolError) -> None:
        """向上下文中添加一个错误记录。"""
        self.run_logger.error(f"捕获到错误: {error}")
        self.errors.append(error)

    def add_file_record(self, record: FileRecordDTO) -> None:
        """添加文件记录。"""
        self.file_records[record.file_id] = record

    def add_content_block(self, block: ContentBlockDTO) -> None:
        """添加内容块，并初始化其分析结果槽位。"""
        block_id = block.block_id
        self.content_blocks[block_id] = block
        if block_id not in self.analysis_results:
            self.analysis_results[block_id] = {at: [] for at in AnalysisType}

    def add_analysis_result(self, result: AnalysisResultDTO) -> None:
        """添加分析结果，并将其添加到两个相关块的记录中。"""
        b1, b2, atype = (result.block_id_1, result.block_id_2, result.analysis_type)
        if b1 not in self.analysis_results:
            self.analysis_results[b1] = {at_enum: [] for at_enum in AnalysisType}
        if result not in self.analysis_results[b1][atype]:
            self.analysis_results[b1][atype].append(result)
        if b2 not in self.analysis_results:
            self.analysis_results[b2] = {at_enum: [] for at_enum in AnalysisType}
        if result not in self.analysis_results[b2][atype]:
            self.analysis_results[b2][atype].append(result)

    def add_user_decision(self, decision: UserDecisionDTO) -> None:
        """添加用户决策。"""
        self.user_decisions[decision.pair_analysis_id] = decision

    def get_content_blocks_for_analysis(
        self, analysis_type: AnalysisType, force_reprocess: bool = False
    ) -> List[ContentBlockDTO]:
        """获取需要进行指定类型分析的内容块列表 (简化逻辑)。"""
        blocks_to_process = []
        for block_id, block in self.content_blocks.items():
            if not block.analysis_text:
                self.run_logger.debug(
                    f"Block {block.block_id}缺少analysis_text, 跳过SimHash分析判断。"
                )
                if analysis_type != AnalysisType.SIMHASH:
                    blocks_to_process.append(block)
                continue
            if analysis_type == AnalysisType.SIMHASH:
                if force_reprocess or block.simhash_value is None:
                    blocks_to_process.append(block)
            else:
                blocks_to_process.append(block)
        return list({b.block_id: b for b in blocks_to_process}.values())
