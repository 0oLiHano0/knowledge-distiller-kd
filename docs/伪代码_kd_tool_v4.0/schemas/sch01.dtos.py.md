# 0523
```python
# kd_tool/schemas/dtos.py (v4.5.1 - 增强注释版)
# -*- coding: utf-8 -*-

"""
=================================================
sch01.dtos.py.md - KD_Tool 数据传输对象 (DTOs)
=================================================

**模块功能**:

- **核心职责**: 定义项目中用于在不同层 (Orchestrator, Stages, Storage) 之间传递数据的核心数据结构。
- **技术选型**: 使用 Pydantic 定义，以实现数据的结构化、类型安全和自动验证。
- **设计原则**: 
    - **严禁**将 DTOs 与 ORM 模型混用。DTOs 是接口契约，ORM 是持久化细节。
    - DTOs 应尽量保持简单，只包含数据和必要的验证/辅助方法。
    - **必须**使用类型提示。
- **目标**: 为 coding 阶段提供清晰、稳定的数据结构契约。

**版本历史**:
- v4.4: 统一 task_id 为 UUID，添加 run_logger 到 Context。
- v4.5: 明确 AnalysisResultDTO.score，规范 SimHash 相关字段。
- v4.5.1: 【架构师决策】恢复并增强注释，明确各 DTO 的规范和设计意图。

---
"""

# 导入 Python 核心库
import uuid
from uuid import UUID, uuid4
import hashlib
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum 

# 导入 Pydantic
from pydantic import BaseModel, Field, validator, model_validator

# 导入 Loguru (用于 PipelineContextDTO)
from loguru import Logger 

# 导入项目内部依赖
from .enums import BlockType, AnalysisType, DecisionType, ProcessingStatus
from ..core.errors import KDToolError

# ==============================================================================
# 核心业务实体 DTOs
# ==============================================================================
# [架构师说明]: 这些 DTOs 代表了 KD_Tool 处理的核心对象：文件、内容块、分析结果和决策。

class FileRecordDTO(BaseModel):
    """
    代表存储系统中已注册文件的记录的 DTO。
    **规范**: 这是文件在系统中的唯一表示，贯穿整个处理流程。
    """
    file_id: str = Field(
        default_factory=lambda: f"file_{uuid.uuid4().hex}", 
        description="文件的唯一标识符。**规范**: 使用 UUID 生成，确保唯一性，保持 str 类型。"
    )
    original_path: Path = Field(
        description="文件在文件系统中的原始绝对路径。**规范**: 必须是绝对路径。"
    )
    file_hash_md5: Optional[str] = Field(
        default=None, 
        max_length=32, 
        description="文件的完整内容 MD5 哈希值。**规范**: 用于快速精确匹配，可在 Prefilter 或后续阶段计算。"
    )
    size_bytes: Optional[int] = Field(
        default=None, 
        ge=0, 
        description="文件大小（字节）。"
    )
    last_modified_at: Optional[datetime] = Field(
        default=None, 
        description="文件最后修改时间戳。**规范**: 必须是 UTC 时间。"
    )
    registered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), 
        description="文件注册时间戳。**规范**: 必须是 UTC 时间。"
    )
    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING, 
        description="文件的处理状态。**规范**: 使用 `ProcessingStatus` 枚举，由各 Stage 更新。"
    )
    processing_history: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="文件处理状态变更的历史记录。**规范**: 用于追踪和调试。"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="与文件相关的其他元数据。**规范**: 用于存储扩展信息，如 Czkawka 结果等。"
    )
    task_id: UUID = Field(
        description="此文件记录所属的处理任务的唯一标识符。**规范**: v4.4 引入，用于端到端追踪。"
    ) 

    @validator('last_modified_at', 'registered_at', pre=True, always=True)
    @classmethod
    def ensure_datetime_is_utc(cls, v: Any) -> Optional[datetime]:
        """**验证器**: 确保所有关键日期时间字段都是 UTC 时区。"""
        if v is None: return None
        if isinstance(v, str):
            try:
                v_dt = datetime.fromisoformat(v.replace('Z', '+00:00')) if v.endswith('Z') else datetime.fromisoformat(v)
            except ValueError: raise ValueError(f"无效的日期时间字符串格式: '{v}'")
        elif isinstance(v, datetime): v_dt = v
        else: raise TypeError(f"期望字符串或日期时间对象，但得到 {type(v)}")
        if v_dt.tzinfo is None: return v_dt.replace(tzinfo=timezone.utc)
        if v_dt.tzinfo != timezone.utc: return v_dt.astimezone(timezone.utc)
        return v_dt

    class Config:
        extra = 'forbid' # **规范**: 禁止未知字段。
        arbitrary_types_allowed = True # 允许 Path, UUID 等。
        json_encoders = { # **规范**: 定义序列化行为，确保一致性。
            datetime: lambda v: v.isoformat().replace('+00:00', 'Z') if v and v.tzinfo == timezone.utc else (v.isoformat() if v else None),
            Path: str,
            UUID: str
        }


class ContentBlockDTO(BaseModel):
    """
    代表从文档中解析出的一个内容块的 DTO。
    **规范**: 这是进行内容分析（MD5, SimHash, Semantic）的基本单元。
    """
    block_id: str = Field(
        default_factory=lambda: f"block_{uuid.uuid4().hex}", 
        description="内容块的唯一标识符。**规范**: 使用 UUID 生成。"
    )
    file_id: str = Field(
        description="此内容块所属的源文件的 file_id。**规范**: 关联到 `FileRecordDTO`。"
    )
    task_id: UUID = Field(
        description="此内容块创建时所属的处理任务的唯一标识符。**规范**: 用于追踪。"
    )
    text_content: str = Field(
        description="内容块的原始文本内容。"
    )
    analysis_text: Optional[str] = Field(
        default=None, 
        description="""
        用于分析的标准化文本内容。
        **规范**: 可能经过清洗（如去 HTML 标签、去停用词等）。如果为 None，默认使用 `text_content`。
        **编码要求**: 所有分析阶段应优先使用此字段。
        """
    )
    block_type: BlockType = Field(
        description="内容块的类型。**规范**: 使用 `BlockType` 枚举。"
    )
    order_in_document: Optional[int] = Field(
        default=None, ge=0, 
        description="内容块在文档中的顺序索引。"
    )
    page_number: Optional[int] = Field(
        default=None, ge=1, 
        description="内容块所在的页码 (如果适用)。"
    )
    text_hash_md5: Optional[str] = Field(
        default=None, max_length=32, 
        description="`analysis_text` 的 MD5 哈希值。**规范**: 由 MD5AnalysisStage 计算。"
    )
    simhash_value: Optional[str] = Field(
        default=None, 
        pattern=r"^[0-9a-fA-F]{16}$|^[0-9a-fA-F]{32}$", 
        description="""
        `analysis_text` 的 SimHash 指纹。
        **规范**: 64位 (16 hex chars) 或 128位 (32 hex chars)。由 SimHashAnalysisStage 计算。
        """
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="与内容块相关的其他元数据 (如来自 `unstructured` 的原始元素信息)。"
    )

    @model_validator(mode='after')
    @classmethod
    def set_analysis_text_if_none(cls, data: Any) -> Any:
        """**验证器**: 确保 `analysis_text` 如果未提供，则使用 `text_content`。"""
        if isinstance(data, cls):
            if data.analysis_text is None and data.text_content is not None:
                data.analysis_text = data.text_content
        return data

    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True
        json_encoders = { UUID: str }


class AnalysisResultDTO(BaseModel):
    """
    代表对**两个**内容块进行**一种特定分析**后得到的结果的 DTO。
    **规范**: 这是所有分析阶段 (MD5, SimHash, Semantic) 的标准输出格式。
    """
    pair_analysis_id: str = Field(
        description="""
        基于块对和分析类型的确定性唯一ID。
        **规范**: 使用 `_calculate_pair_analysis_id` 生成，确保幂等性。
        """
    )
    block_id_1: str = Field(description="第一个内容块的 block_id。")
    block_id_2: str = Field(description="第二个内容块的 block_id。")
    analysis_type: AnalysisType = Field(
        description="执行的分析类型。**规范**: 使用 `AnalysisType` 枚举。"
    )
    
    score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="""
        标准化相似度分数 (0.0 到 1.0 之间)。
        **规范**: 
        - 对于 MD5: 匹配为 1.0，不匹配为 0.0。
        - 对于 SimHash: 值为 1.0 - (汉明距离 / 哈希位数)。
        - 对于 Semantic: 直接为语义相似度分数。
        **编码要求**: 必须在此范围内，或为 None。
        """
    )
    
    details: Dict[str, Any] = Field(
        default_factory=dict, 
        description="""
        与分析结果相关的其他详细信息。
        **规范**: 
        - 对于 SimHash，**必须**包含 'hamming_distance' (int) 和 'hash_bits' (int)。
        - 对于其他分析，可用于存储额外信息（如模型名称等）。
        """
    )
    task_id: UUID = Field(
        description="此分析结果产生时所属的处理任务的唯一标识符。"
    )

    @staticmethod
    def _calculate_pair_analysis_id(block_id_1: str, block_id_2: str, analysis_type: Union[AnalysisType, str]) -> str:
        """**辅助方法**: 计算确定性的 ID。"""
        sorted_block_ids = sorted([block_id_1, block_id_2])
        analysis_type_value = analysis_type.value if isinstance(analysis_type, Enum) else str(analysis_type)
        key_string = f"pair_{sorted_block_ids[0]}__{sorted_block_ids[1]}_type_{analysis_type_value}"
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()

    @model_validator(mode='before') 
    @classmethod
    def set_pair_analysis_id_on_validation(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """**验证器**: 在验证前自动计算 `pair_analysis_id`。"""
        block_id_1 = values.get('block_id_1')
        block_id_2 = values.get('block_id_2')
        analysis_type = values.get('analysis_type')

        if block_id_1 and block_id_2 and analysis_type:
            values['pair_analysis_id'] = cls._calculate_pair_analysis_id(
                block_id_1, block_id_2, analysis_type
            )
        elif 'pair_analysis_id' not in values:
             raise ValueError("无法计算 pair_analysis_id，且未提供 block_id_1, block_id_2, 或 analysis_type。")
        return values

    @model_validator(mode='after')
    @classmethod
    def check_simhash_details(cls, data: Any) -> Any:
        """**验证器**: 如果分析类型是 SimHash，验证 details 字段是否符合规范。"""
        if isinstance(data, cls):
            if data.analysis_type == AnalysisType.SIMHASH:
                if 'hamming_distance' not in data.details or not isinstance(data.details['hamming_distance'], int):
                    raise ValueError("SimHash 分析结果必须在 details 中包含整数 'hamming_distance'。")
                if 'hash_bits' not in data.details or not isinstance(data.details['hash_bits'], int):
                     raise ValueError("SimHash 分析结果必须在 details 中包含整数 'hash_bits'。")
        return data

    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True
        json_encoders = { UUID: str }


class UserDecisionDTO(BaseModel):
    """
    代表用户或系统针对一个分析结果所做的决策的 DTO。
    **规范**: 这是 `DecisionStage` 的核心输出，也是 `CleanupStage` 的输入。
    """
    pair_analysis_id: str = Field(
        description="此决策针对的 AnalysisResultDTO 的 ID。**规范**: 关联到 `AnalysisResultDTO`。"
    )
    decision: DecisionType = Field(
        default=DecisionType.UNDECIDED, 
        description="做出的具体决策。**规范**: 使用 `DecisionType` 枚举。"
    )
    decided_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), 
        description="决策时间戳 (UTC)。**规范**: 必须是 UTC。"
    )
    decided_by: Optional[str] = Field(
        default=None, 
        description="决策者标识符 (e.g., 'system_rule_v1', 'user_admin')。"
    )
    notes: Optional[str] = Field(
        default=None, max_length=1024, 
        description="决策备注。"
    )
    task_id: UUID = Field(
        description="此决策产生时所属的处理任务的唯一标识符。"
    )

    @validator('decided_at', pre=True, always=True)
    @classmethod
    def ensure_datetime_is_utc(cls, v: Any) -> Optional[datetime]:
        """**验证器**: 确保决策时间是 UTC。"""
        if v is None: return None
        if isinstance(v, str):
            try:
                v_dt = datetime.fromisoformat(v.replace('Z', '+00:00')) if v.endswith('Z') else datetime.fromisoformat(v)
            except ValueError: raise ValueError(f"无效的日期时间字符串格式: '{v}'")
        elif isinstance(v, datetime): v_dt = v
        else: raise TypeError(f"期望字符串或日期时间对象，但得到 {type(v)}")
        if v_dt.tzinfo is None: return v_dt.replace(tzinfo=timezone.utc)
        if v_dt.tzinfo != timezone.utc: return v_dt.astimezone(timezone.utc)
        return v_dt

    class Config:
        extra = 'forbid'
        json_encoders = {
            datetime: lambda v: v.isoformat().replace('+00:00', 'Z') if v and v.tzinfo == timezone.utc else (v.isoformat() if v else None),
            UUID: str
        }

# ==============================================================================
# 管道上下文 DTO (Pipeline Context DTO)
# ==============================================================================
# [架构师说明]: 这是整个流水线的“血液”，在各个 Stage 之间传递状态和数据。
#               它的设计至关重要，既要包含足够的信息，又要避免过于臃肿。

class PipelineContextDTO(BaseModel):
    """
    管道上下文数据传输对象 (Pipeline Context DTO)。
    **核心职责**: 在 Orchestrator 控制的流水线中，作为数据和状态的载体，从一个 Stage 传递到下一个 Stage。
    **规范**: 
    - 它的生命周期与一次 `Orchestrator.run` 调用绑定。
    - **严禁**将其设计为有状态对象；它只是数据的容器。
    - Stage **必须**通过修改此对象来传递结果。
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
        description="""
        与当前任务绑定的日志记录器。
        **规范**: 此 logger **必须**已经绑定了 `task_id` 上下文。
        **编码要求**: 各 Stage **必须**使用此 logger 进行日志记录，**严禁**自行导入 `loguru.logger`。
        **注意**: Logger 对象不可序列化，若需持久化或分发 Context，需特殊处理。
        """
    )

    # --- 数据存储 (内存中的工作集) ---
    # [架构师说明]: 这些字段持有当前任务处理过程中的 *内存* 数据视图。
    #               Stage 可以从 Storage 加载数据到这里，或将这里的数据保存到 Storage。
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
        description="""
        分析结果 (AnalysisResultDTO)。
        **结构**: 第一层键是 `block_id`，第二层键是 `AnalysisType`。
        **规范**: `add_analysis_result` 方法负责维护此结构。
        """
    )
    user_decisions: Dict[str, UserDecisionDTO] = Field(
        default_factory=dict,
        description="用户决策 (UserDecisionDTO)，以 pair_analysis_id 为键。"
    )

    # --- 状态与错误 ---
    errors: List[KDToolError] = Field(
        default_factory=list,
        description="""
        在流水线处理过程中收集到的所有错误。
        **规范**: Stage **必须**将捕获到的 `KDToolError` 添加到此列表。
        """
    )
    
    shared_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="""
        用于阶段间共享临时或非结构化数据的区域。
        **警告**: 应谨慎使用此字段！过度使用会破坏架构的清晰性。
                   仅在确实无法通过 DTOs 传递数据时才考虑使用。
        """
    )

    # --- 辅助方法 ---
    # [架构师说明]: 提供标准方法来操作 Context，确保数据一致性。

    def get_task_id_str(self) -> str:
        """获取字符串格式的 task_id，常用于日志或文件名。"""
        return str(self.task_id)

    def add_error(self, error: KDToolError) -> None:
        """
        向上下文中添加一个错误记录。
        **规范**: 传入的 **必须** 是 `KDToolError` 的子类。
        """
        self.run_logger.error(f"捕获到错误: {error}") # 自动记录错误
        self.errors.append(error)

    def add_file_record(self, record: FileRecordDTO) -> None:
        """
        添加文件记录，并校验 task_id。
        **规范**: 确保添加到 Context 的数据都属于当前任务。
        """
        if record.task_id != self.task_id:
            raise ValueError(f"尝试添加的 FileRecordDTO ({record.file_id}) 的 task_id "
                             f"({record.task_id}) 与 context 的 task_id ({self.task_id}) 不符。")
        self.file_records[record.file_id] = record

    def add_content_block(self, block: ContentBlockDTO) -> None:
        """
        添加内容块，校验 task_id，并初始化其分析结果槽位。
        """
        if block.task_id != self.task_id:
            raise ValueError(f"尝试添加的 ContentBlockDTO ({block.block_id}) 的 task_id "
                             f"({block.task_id}) 与 context 的 task_id ({self.task_id}) 不符。")
            
        block_id = block.block_id
        self.content_blocks[block_id] = block
        # **重要**: 为新块预先创建分析结果的字典结构，避免后续检查。
        if block_id not in self.analysis_results:
             self.analysis_results[block_id] = {at: [] for at in AnalysisType}

    def add_analysis_result(self, result: AnalysisResultDTO) -> None:
        """
        添加分析结果，校验 task_id，并将其添加到两个相关块的记录中。
        **规范**: 确保结果被正确索引，便于 `DecisionStage` 查询。
        """
        if result.task_id != self.task_id:
             raise ValueError(f"尝试添加的 AnalysisResultDTO ({result.pair_analysis_id}) 的 task_id "
                             f"({result.task_id}) 与 context 的 task_id ({self.task_id}) 不符。")

        b1, b2, atype = result.block_id_1, result.block_id_2, result.analysis_type
        
        # 确保 block_id 存在 (通常 add_content_block 会保证)
        if b1 not in self.analysis_results: self.analysis_results[b1] = {at: [] for at in AnalysisType}
        # 避免重复添加
        if result not in self.analysis_results[b1][atype]: self.analysis_results[b1][atype].append(result)

        if b2 not in self.analysis_results: self.analysis_results[b2] = {at: [] for at in AnalysisType}
        if result not in self.analysis_results[b2][atype]: self.analysis_results[b2][atype].append(result)

    def add_user_decision(self, decision: UserDecisionDTO) -> None:
        """添加用户决策，并校验 task_id。"""
        if decision.task_id != self.task_id:
            raise ValueError(f"尝试添加的 UserDecisionDTO ({decision.pair_analysis_id}) 的 task_id "
                             f"({decision.task_id}) 与 context 的 task_id ({self.task_id}) 不符。")
        self.user_decisions[decision.pair_analysis_id] = decision

    def get_content_blocks_for_analysis(
        self,
        analysis_type: AnalysisType,
        force_reprocess: bool = False
    ) -> List[ContentBlockDTO]:
        """
        获取需要进行指定类型分析的内容块列表。
        **逻辑**: 检查 `analysis_results` 来确定哪些块尚未进行指定类型的分析。
                  **注意**: 这个逻辑比较粗略，它只检查某个块是否 *参与过* 某种分析，
                          并不保证它与 *所有其他块* 都比较过。
                          对于 SimHash 和 Semantic，可能需要更精细的逻辑或依赖
                          `find_similar_pairs` 这样的高效查找。
                          目前我们假设 SimHashStage 会自行处理比较逻辑。
        """
        blocks_to_process = []
        for block_id, block in self.content_blocks.items():
            # 简化逻辑: SimHash/Semantic 阶段会自行判断哪些需要计算/比较。
            # 此处可以返回所有块，或者根据字段 (如 simhash_value is None) 过滤。
            # 为了给 Stage 最大灵活性，可以先返回大部分，让 Stage 内部决定。
            # 或者，此方法可以被废弃，由 Stage 自行从 context.content_blocks 获取。
            # 【v4.5.1 决策】暂时保留此方法，但 Stage 实现可以不完全依赖它。
            # 让我们基于字段过滤，返回需要计算哈希的块。
            if analysis_type == AnalysisType.SIMHASH:
                if force_reprocess or block.simhash_value is None:
                    blocks_to_process.append(block)
            # TODO: 添加其他分析类型的逻辑
            else: # 默认返回所有块，让 Stage 自己判断
                blocks_to_process.append(block)
                
        # 移除重复项 (虽然理论上不应该有)
        # return list(set(blocks_to_process)) # Pydantic 对象不可哈希，不能用 set
        
        # 使用字典去重
        return list({b.block_id: b for b in blocks_to_process}.values())


    class Config:
        arbitrary_types_allowed = True # 允许 Logger 等非 Pydantic 类型
        validate_assignment = True # 对字段赋值时进行验证
        json_encoders = { # 定义序列化行为
            UUID: str
        }
```