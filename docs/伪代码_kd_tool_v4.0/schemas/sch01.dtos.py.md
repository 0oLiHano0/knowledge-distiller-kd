# 0523
```python
# knowledge_distiller_kd/schemas/dtos.py

"""
该模块定义了项目中核心的数据传输对象 (DTOs)。
这些 DTOs 主要用于模块间的数据传递、接口的输入输出以及与存储层的交互。
"""
import uuid
import hashlib
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum 

from pydantic import BaseModel, Field, validator, model_validator

# 从同级 enums.py 模块导入枚举类型
# 架构说明: enums.py 定义了 BlockType, AnalysisType, DecisionType, ProcessingStatus
from .enums import BlockType, AnalysisType, DecisionType, ProcessingStatus
# 导入核心错误类型
from ..core.errors import KDToolError


class FileRecordDTO(BaseModel):
    """代表存储系统中已注册文件的记录的 DTO。"""
    file_id: str = Field(default_factory=lambda: f"file_{uuid.uuid4()}", description="文件的唯一标识符。")
    original_path: Path = Field(description="文件在文件系统中的原始绝对路径。")
    file_hash_md5: Optional[str] = Field(default=None, max_length=32, description="文件的完整内容 MD5 哈希值。")
    size_bytes: Optional[int] = Field(default=None, ge=0, description="文件大小（字节）。")
    last_modified_at: Optional[datetime] = Field(default=None, description="文件最后修改时间戳 (UTC)。")
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="文件注册时间戳 (UTC)。")
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="文件的处理状态。")
    processing_history: List[Dict[str, Any]] = Field(default_factory=list, description="文件处理状态变更的历史。")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="与文件相关的其他元数据。")

    @validator('last_modified_at', 'registered_at', pre=True, always=True)
    @classmethod
    def ensure_datetime_is_utc(cls, v: Any) -> Optional[datetime]:
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
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat().replace('+00:00', 'Z') if v and v.tzinfo == timezone.utc else (v.isoformat() if v else None),
            Path: str
        }


class ContentBlockDTO(BaseModel):
    """代表从文档中解析出的一个内容块的 DTO。"""
    block_id: str = Field(default_factory=lambda: f"block_{uuid.uuid4()}", description="内容块的唯一标识符。")
    file_id: str = Field(description="此内容块所属的源文件的 file_id。")
    text_content: str = Field(description="内容块的原始文本内容。")
    analysis_text: Optional[str] = Field(default=None, description="用于分析的标准化文本内容。")
    block_type: BlockType = Field(description="内容块的类型。")
    order_in_document: Optional[int] = Field(default=None, ge=0, description="内容块在文档中的顺序索引。")
    page_number: Optional[int] = Field(default=None, ge=1, description="内容块所在的页码。")
    text_hash_md5: Optional[str] = Field(default=None, max_length=32, description="`analysis_text` 的 MD5 哈希值。")
    simhash_value: Optional[str] = Field(
        default=None, min_length=16, max_length=16, pattern=r"^[0-9a-fA-F]{16}$",
        description="`analysis_text` 的 SimHash 指纹（16位十六进制）。"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="与内容块相关的其他元数据。")

    @model_validator(mode='after')
    @classmethod
    def set_analysis_text_if_none(cls, data: Any) -> Any:
        if isinstance(data, cls):
            if data.analysis_text is None and data.text_content is not None:
                data.analysis_text = data.text_content
        return data

    class Config:
        extra = 'forbid'


class AnalysisResultDTO(BaseModel):
    """代表对两个内容块进行分析后得到的结果的 DTO。"""
    pair_analysis_id: str = Field(description="基于块对和分析类型的确定性唯一ID。")
    block_id_1: str = Field(description="第一个内容块的 block_id。")
    block_id_2: str = Field(description="第二个内容块的 block_id。")
    analysis_type: AnalysisType = Field(description="执行的分析类型。")
    score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="相似度分数。MD5匹配为1.0，不匹配为0.0。"
    )
    details: Dict[str, Any] = Field(default_factory=dict, description="与分析结果相关的其他详细信息。")

    @staticmethod
    def _calculate_pair_analysis_id(block_id_1: str, block_id_2: str, analysis_type: Union[AnalysisType, str]) -> str:
        sorted_block_ids = sorted([block_id_1, block_id_2])
        analysis_type_value = analysis_type.value if isinstance(analysis_type, Enum) else str(analysis_type)
        key_string = f"pair_{sorted_block_ids[0]}__{sorted_block_ids[1]}_type_{analysis_type_value}"
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()

    @model_validator(mode='before') # <-- 使用 before 模式，在创建实例前计算ID
    @classmethod
    def set_pair_analysis_id_on_validation(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        block_id_1 = values.get('block_id_1')
        block_id_2 = values.get('block_id_2')
        analysis_type = values.get('analysis_type')

        if block_id_1 and block_id_2 and analysis_type:
            values['pair_analysis_id'] = cls._calculate_pair_analysis_id(
                block_id_1, block_id_2, analysis_type
            )
        elif 'pair_analysis_id' not in values:
             # 如果无法计算且未提供，则引发错误或设置默认值（这里选择引发错误）
             raise ValueError("无法计算 pair_analysis_id，且未提供 block_id_1, block_id_2, 或 analysis_type。")
        return values

    class Config:
        extra = 'forbid'


class UserDecisionDTO(BaseModel):
    """代表用户或系统针对一个分析结果所做的决策的 DTO。"""
    pair_analysis_id: str = Field(description="此决策针对的 AnalysisResultDTO 的 ID。")
    decision: DecisionType = Field(default=DecisionType.UNDECIDED, description="做出的具体决策。")
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="决策时间戳 (UTC)。")
    decided_by: Optional[str] = Field(default=None, description="决策者标识符。")
    notes: Optional[str] = Field(default=None, max_length=1024, description="决策备注。")

    @validator('decided_at', pre=True, always=True)
    @classmethod
    def ensure_datetime_is_utc(cls, v: Any) -> Optional[datetime]: # 与FileRecordDTO中的相同
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
        }

# ==============================================================================
# 管道上下文 DTO (Pipeline Context DTO)
# ==============================================================================

class PipelineContextDTO(BaseModel):
    """
    管道上下文数据传输对象 (Pipeline Context DTO)。

    这个对象在 Orchestrator 创建的流水线中，从一个 Stage 传递到下一个 Stage。
    它携带了当前处理任务的所有相关状态、数据和错误信息。
    """
    task_id: UUID = Field(
        default_factory=uuid4,
        description="本次流水线执行的唯一标识符。"
    )
    initial_input_paths: List[Path] = Field(
        default_factory=list,
        description="本次任务初始请求处理的输入路径列表。"
    )
# ==============================================================================

    # --- 数据存储 ---
    file_records: Dict[str, FileRecordDTO] = Field(
        default_factory=dict,
        description="处理过程中涉及的文件记录 (FileRecordDTO)，以 file_id 为键。"
    )
    content_blocks: Dict[str, ContentBlockDTO] = Field(
        default_factory=dict,
        description="从文件中提取的内容块 (ContentBlockDTO)，以 block_id 为键。"
    )
    analysis_results: Dict[str, Dict[AnalysisType, List[AnalysisResultDTO]]] = Field(
        default_factory=lambda: {at: [] for at in AnalysisType},
        description="分析结果 (AnalysisResultDTO)。第一层键是 block_id，第二层键是 AnalysisType。"
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
        description="用于阶段间共享临时或非结构化数据的区域。应谨慎使用。"
    )

    # --- 辅助方法 ---

    def get_task_id(self) -> str:
        return str(self.task_id)

    def add_error(self, error: KDToolError) -> None:
        self.errors.append(error)

    def add_file_record(self, record: FileRecordDTO) -> None:
        self.file_records[record.file_id] = record

    def add_content_block(self, block: ContentBlockDTO) -> None:
        block_id = block.block_id
        self.content_blocks[block_id] = block
        if block_id not in self.analysis_results:
             self.analysis_results[block_id] = {at: [] for at in AnalysisType}

    def add_analysis_result(self, result: AnalysisResultDTO) -> None:
        """
        添加一个分析结果。
        由于结果是成对的，我们会将其添加到两个相关块的记录中。
        """
        b1, b2, atype = result.block_id_1, result.block_id_2, result.analysis_type
        
        # 确保 block_id_1 的条目存在并添加结果
        if b1 not in self.analysis_results:
            self.analysis_results[b1] = {at: [] for at in AnalysisType}
        if result not in self.analysis_results[b1][atype]: # 避免重复添加
            self.analysis_results[b1][atype].append(result)

        # 确保 block_id_2 的条目存在并添加结果
        if b2 not in self.analysis_results:
            self.analysis_results[b2] = {at: [] for at in AnalysisType}
        if result not in self.analysis_results[b2][atype]: # 避免重复添加
            self.analysis_results[b2][atype].append(result)

    def add_user_decision(self, decision: UserDecisionDTO) -> None:
        """添加一个用户决策。"""
        self.user_decisions[decision.pair_analysis_id] = decision

    def get_content_blocks_for_analysis(
        self,
        analysis_type: AnalysisType,
        force_reprocess: bool = False
    ) -> List[ContentBlockDTO]:
        """
        获取需要进行指定类型分析的内容块列表。
        它会检查 `analysis_results` 来确定哪些块尚未进行指定类型的分析。
        """
        blocks_to_process = []
        for block_id, block in self.content_blocks.items():
            # 检查这个 block_id 是否已经作为 block_id_1 或 block_id_2 出现在
            # 任何一个指定类型的 AnalysisResultDTO 中。
            has_result = False
            if block_id in self.analysis_results:
                if self.analysis_results[block_id].get(analysis_type):
                    has_result = True

            if force_reprocess or not has_result:
                blocks_to_process.append(block)

        return blocks_to_process

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
```