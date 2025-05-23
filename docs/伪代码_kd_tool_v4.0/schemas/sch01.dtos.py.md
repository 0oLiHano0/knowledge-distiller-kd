0522
```python

# 存放 `FileRecordDTO`, `ContentBlockDTO`, `AnalysisResultDTO`, `UserDecisionDTO` 等

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

from pydantic import BaseModel, Field, validator, model_validator

# 从同级 enums.py 模块导入枚举类型
from .enums import BlockType, AnalysisType, DecisionType, ProcessingStatus

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

    @model_validator(mode='after')
    @classmethod
    def set_pair_analysis_id_on_validation(cls, model_instance: Any) -> Any:
        if isinstance(model_instance, cls):
            try:
                model_instance.pair_analysis_id = cls._calculate_pair_analysis_id(
                    model_instance.block_id_1,
                    model_instance.block_id_2,
                    model_instance.analysis_type
                )
            except (ValueError, AttributeError, TypeError) as e:
                raise ValueError(f"无法计算 pair_analysis_id: {e}") from e
        return model_instance

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

```