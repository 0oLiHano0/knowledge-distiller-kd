# kd_tool/core/core_dtos.py (v4.6 - PipelineContextDTO 迁移版)
# -*- coding: utf-8 -*-

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

# --- Python 标准库及第三方库导入 ---
import uuid
from uuid import UUID, uuid4 # [指令] 必须导入 UUID 和 uuid4
from pathlib import Path
from typing import Optional, Dict, Any, List # [指令] 必须导入所需类型

# --- Pydantic 和 Loguru 导入 ---
from pydantic import BaseModel, Field # [指令] 必须导入 Pydantic 核心
from loguru import Logger # [指令] 必须导入 Logger

# --- 项目内部模块导入 ---
# [指令] PipelineContextDTO 依赖的核心业务 DTOs 和 Enums 仍从调整后的中央 schemas 目录导入。
# [指令] 路径 "../schemas/" 是相对于当前 "kd_tool/core/" 目录而言。
from kd_tool.schemas.dtos import (
    FileRecordDTO,
    ContentBlockDTO,
    AnalysisResultDTO,
    UserDecisionDTO
)
from kd_tool.schemas.enums import AnalysisType
from kd_tool.core.errors import KDToolError


# ==============================================================================
# 管道上下文 DTO (Pipeline Context DTO)
# ==============================================================================
# [架构师说明]: PipelineContextDTO 是流水线数据和状态的核心载体。
#               其辅助方法 add_... 已更新，移除了对 DTO 内部 task_id 的校验。

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
    # --- 任务标识与输入 ---
    task_id: UUID = Field(
        default_factory=uuid4,
        description="本次流水线执行的唯一标识符 (UUID)。"
    )
    initial_input_paths: List[Path] = Field(
        default_factory=list,
        description="本次任务初始请求处理的输入路径列表。"
    )
    # --- 上下文日志 ---
    run_logger: Logger = Field(
        description="与当前任务绑定的日志记录器。 **规范**: 此 logger **必须**已绑定 `task_id`。"
    )

    # --- 数据存储 (内存中的工作集) ---
    file_records: Dict[str, FileRecordDTO] = Field(
        default_factory=dict,
        description="处理过程中涉及的文件记录 (FileRecordDTO)，以 file_id 为键。"
    )
    content_blocks: Dict[str, ContentBlockDTO] = Field(
        default_factory=dict,
        description="从文件中提取的内容块 (ContentBlockDTO)，以 block_id 为键。"
    )
    analysis_results: Dict[str, Dict[AnalysisType, List[AnalysisResultDTO]]] = Field(
        default_factory=dict,
        description="分析结果 (AnalysisResultDTO)。结构: {block_id: {AnalysisType: [AnalysisResultDTO]}}"
    )
    user_decisions: Dict[str, UserDecisionDTO] = Field(
        default_factory=dict,
        description="用户决策 (UserDecisionDTO)，以 pair_analysis_id 为键。"
    )

    # --- 状态与错误 ---
    errors: List[KDToolError] = Field(
        default_factory=list,
        description="在流水线处理过程中收集到的所有错误。"
    )

    shared_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="用于阶段间共享临时或非结构化数据的区域。**警告**: 应谨慎使用！"
    )

    # --- 辅助方法 ---
    # [架构师说明]: 提供标准方法来操作 Context，确保数据一致性。
    # [架构指令 v4.6]: add_... 方法不再校验 DTO 中的 task_id。

    def get_task_id_str(self) -> str:
        """获取字符串格式的 task_id，常用于日志或文件名。"""
        return str(self.task_id)

    def add_error(self, error: KDToolError) -> None:
        """向上下文中添加一个错误记录。"""
        self.run_logger.error(f"捕获到错误: {error}") # 使用 run_logger 记录
        self.errors.append(error)

    def add_file_record(self, record: FileRecordDTO) -> None:
        """添加文件记录。"""
        # [架构指令 v4.6] 已移除 record.task_id 校验
        self.file_records[record.file_id] = record

    def add_content_block(self, block: ContentBlockDTO) -> None:
        """添加内容块，并初始化其分析结果槽位。"""
        # [架构指令 v4.6] 已移除 block.task_id 校验
        block_id = block.block_id
        self.content_blocks[block_id] = block
        if block_id not in self.analysis_results:
             self.analysis_results[block_id] = {at: [] for at in AnalysisType}

    def add_analysis_result(self, result: AnalysisResultDTO) -> None:
        """添加分析结果，并将其添加到两个相关块的记录中。"""
        # [架构指令 v4.6] 已移除 result.task_id 校验
        b1, b2, atype = result.block_id_1, result.block_id_2, result.analysis_type
        # [指令] 确保在访问前 block_id 存在于 analysis_results 中
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
        # [架构指令 v4.6] 已移除 decision.task_id 校验
        self.user_decisions[decision.pair_analysis_id] = decision

    def get_content_blocks_for_analysis(
        self,
        analysis_type: AnalysisType,
        force_reprocess: bool = False
    ) -> List[ContentBlockDTO]:
        """获取需要进行指定类型分析的内容块列表 (简化逻辑)。"""
        blocks_to_process = []
        for block_id, block in self.content_blocks.items():
            # [指令] 确保 analysis_text 存在才进行后续判断
            if not block.analysis_text: # 或者根据需要判断 block.text_content
                self.run_logger.trace(f"Block {block.block_id}缺少analysis_text, 跳过SimHash分析判断。")
                if analysis_type != AnalysisType.SIMHASH: # 非SimHash分析则加入处理
                     blocks_to_process.append(block)
                continue

            if analysis_type == AnalysisType.SIMHASH:
                if force_reprocess or block.simhash_value is None:
                    blocks_to_process.append(block)
            else: # 对于其他分析类型，默认都需要处理 (Stage内部可进一步过滤)
                blocks_to_process.append(block)
        # [指令] 使用字典去重，确保返回列表中的 ContentBlockDTO 对象唯一
        return list({b.block_id: b for b in blocks_to_process}.values())


    class Config:
        arbitrary_types_allowed = True # [指令] 必须允许 Logger 等非 Pydantic 类型
        validate_assignment = True # [指令] 建议对字段赋值时进行验证
        json_encoders = {
            UUID: str # [指令] 必须保留，因为 task_id 是 UUID
        }