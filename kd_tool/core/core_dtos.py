"""
=================================================
core_dtos.py - KD_Tool 核心数据传输对象 (v4.6)
=================================================

**模块功能**:

- **核心职责**: 定义 `core` 层特有的、关键的数据传输对象。
- **v4.6 核心变更**:
    - **[架构指令]** `PipelineContextDTO` 从原 `schemas` 目录迁移至此。
    - **[架构指令]** `PipelineContextDTO` 是本模块当前唯一的 DTO。

---
"""
import uuid
from uuid import UUID, uuid4
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from kd_tool.logging.protocols import LoggerProtocol # kd_tool/logging/protocols.py 日志协议
from kd_tool.schemas.dtos import FileRecordDTO, ContentBlockDTO, AnalysisResultDTO, UserDecisionDTO
from kd_tool.schemas.enums import AnalysisType
from kd_tool.core.errors import KDToolError


class PipelineContextDTO(BaseModel):
    """
    管道上下文数据传输对象 (Pipeline Context DTO)。
    **核心职责**: 在 Orchestrator 控制的流水线中，作为数据和状态的载体，从一个 Stage 传递到下一个 Stage。
    **规范**:
    - 它的生命周期与一次 `Orchestrator.run` 调用绑定。
    - **[架构指令] 严禁**将其设计为有状态对象；它只是数据的容器。
    - Stage **必须**通过修改此对象来传递结果。
    - **[架构指令 v4.6] 必须** 包含 `task_id` 和 `run_logger`，作为任务的唯一标识和日志记录器。
    """    
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True,
        validate_assignment=True)
    task_id: UUID = Field(default_factory=uuid4, description=
        '本次流水线执行的唯一标识符 (UUID)。')
    initial_input_paths: List[Path] = Field(default_factory=list,
        description='本次任务初始请求处理的输入路径列表。')
    run_logger: LoggerProtocol = Field(description=
        '与当前任务绑定的日志记录器。 **规范**: 此 logger **必须**已绑定 `task_id`。')
    file_records: Dict[str, FileRecordDTO] = Field(default_factory=dict,
        description='处理过程中涉及的文件记录 (FileRecordDTO)，以 file_id 为键。')
    content_blocks: Dict[str, ContentBlockDTO] = Field(default_factory=dict,
        description='从文件中提取的内容块 (ContentBlockDTO)，以 block_id 为键。')
    analysis_results: Dict[str, Dict[AnalysisType, List[AnalysisResultDTO]]
        ] = Field(default_factory=dict, description=
        '分析结果 (AnalysisResultDTO)。结构: {block_id: {AnalysisType: [AnalysisResultDTO]}}'
        )
    user_decisions: Dict[str, UserDecisionDTO] = Field(default_factory=dict,
        description='用户决策 (UserDecisionDTO)，以 pair_analysis_id 为键。')
    errors: List[KDToolError] = Field(default_factory=list, description=
        '在流水线处理过程中收集到的所有错误。')
    shared_data: Dict[str, Any] = Field(default_factory=dict, description=
        '用于阶段间共享临时或非结构化数据的区域。**警告**: 应谨慎使用！')

    def get_task_id_str(self) ->str:
        """获取字符串格式的 task_id，常用于日志或文件名。"""
        return str(self.task_id)

    def add_error(self, error: KDToolError) ->None:
        """向上下文中添加一个错误记录。"""
        self.run_logger.error(f'捕获到错误: {error}')
        self.errors.append(error)

    def add_file_record(self, record: FileRecordDTO) ->None:
        """添加文件记录。"""
        self.file_records[record.file_id] = record

    def add_content_block(self, block: ContentBlockDTO) ->None:
        """添加内容块，并初始化其分析结果槽位。"""
        block_id = block.block_id
        self.content_blocks[block_id] = block
        if block_id not in self.analysis_results:
            self.analysis_results[block_id] = {at: [] for at in AnalysisType}

    def add_analysis_result(self, result: AnalysisResultDTO) ->None:
        """添加分析结果，并将其添加到两个相关块的记录中。"""
        b1, b2, atype = (result.block_id_1, result.block_id_2, result.
            analysis_type)
        if b1 not in self.analysis_results:
            self.analysis_results[b1] = {at_enum: [] for at_enum in
                AnalysisType}
        if result not in self.analysis_results[b1][atype]:
            self.analysis_results[b1][atype].append(result)
        if b2 not in self.analysis_results:
            self.analysis_results[b2] = {at_enum: [] for at_enum in
                AnalysisType}
        if result not in self.analysis_results[b2][atype]:
            self.analysis_results[b2][atype].append(result)

    def add_user_decision(self, decision: UserDecisionDTO) ->None:
        """添加用户决策。"""
        self.user_decisions[decision.pair_analysis_id] = decision

    def get_content_blocks_for_analysis(self, analysis_type: AnalysisType,
        force_reprocess: bool=False) ->List[ContentBlockDTO]:
        """获取需要进行指定类型分析的内容块列表 (简化逻辑)。"""
        blocks_to_process = []
        for block_id, block in self.content_blocks.items():
            if not block.analysis_text:
                self.run_logger.trace(
                    f'Block {block.block_id}缺少analysis_text, 跳过SimHash分析判断。')
                if analysis_type != AnalysisType.SIMHASH:
                    blocks_to_process.append(block)
                continue
            if analysis_type == AnalysisType.SIMHASH:
                if force_reprocess or block.simhash_value is None:
                    blocks_to_process.append(block)
            else:
                blocks_to_process.append(block)
        return list({b.block_id: b for b in blocks_to_process}.values())