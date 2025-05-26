"""
=================================================
sch01.dtos.py.md - KD_Tool 数据传输对象 (DTOs) (v4.6)
=================================================

**模块功能**:

- **核心职责**: 定义项目中用于在不同层 (Orchestrator, Stages, Storage) 之间传递数据的核心数据结构。
- **技术选型**: 使用 Pydantic 定义，以实现数据的结构化、类型安全和自动验证。
- **设计原则**:
    - **[架构指令] 严禁**将 DTOs 与 ORM 模型混用。DTOs 是接口契约，ORM 是持久化细节。
    - DTOs 应尽量保持简单，只包含数据和必要的验证/辅助方法。
    - **[架构指令] 必须**使用类型提示。
- **v4.6 核心变更**:
    - **[架构指令] 必须** 从 `FileRecordDTO`, `ContentBlockDTO`, `AnalysisResultDTO`, `UserDecisionDTO`
      中移除 `task_id` 字段。
    - **[架构原因]** `task_id` 现在由 `PipelineContextDTO` 全权管理。
      核心 DTOs 应专注于业务实体本身。如需追踪，应通过日志或元数据实现。
    - **[架构指令] 必须** 更新 `PipelineContextDTO` 的 `add_...` 辅助方法，移除其中对 DTO `task_id` 的校验。

---
"""
import uuid
from uuid import UUID, uuid4
import hashlib
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
from loguru import Logger
from kd_tool.schemas.enums import BlockType, AnalysisType, DecisionType, ProcessingStatus
from kd_tool.core.errors import KDToolError


class FileRecordDTO(BaseModel):
    """
    代表存储系统中已注册文件的记录的 DTO。
    **规范**: 这是文件在系统中的唯一表示，贯穿整个处理流程。
    """
    file_id: str = Field(default_factory=lambda :
        f'file_{uuid.uuid4().hex}', description=
        '文件的唯一标识符。**规范**: 使用 UUID 生成，确保唯一性，保持 str 类型。')
    original_path: Path = Field(description='文件在文件系统中的原始绝对路径。**规范**: 必须是绝对路径。')
    file_hash_md5: Optional[str] = Field(default=None, max_length=32,
        description='文件的完整内容 MD5 哈希值。**规范**: 用于快速精确匹配。')
    size_bytes: Optional[int] = Field(default=None, ge=0, description=
        '文件大小（字节）。')
    last_modified_at: Optional[datetime] = Field(default=None, description=
        '文件最后修改时间戳。**规范**: 必须是 UTC 时间。')
    registered_at: datetime = Field(default_factory=lambda : datetime.now(
        timezone.utc), description='文件注册时间戳。**规范**: 必须是 UTC 时间。')
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.
        PENDING, description='文件的处理状态。**规范**: 使用 `ProcessingStatus` 枚举。')
    processing_history: List[Dict[str, Any]] = Field(default_factory=list,
        description='文件处理状态变更的历史记录。**规范**: 用于追踪和调试。')
    metadata: Dict[str, Any] = Field(default_factory=dict, description=
        '与文件相关的其他元数据。**规范**: 用于存储扩展信息。')

    @field_validator('last_modified_at', 'registered_at', mode='before',
        always=True)
    def ensure_datetime_is_utc(cls, v: Any) ->Optional[datetime]:
        """**验证器**: 确保所有关键日期时间字段都是 UTC 时区。"""
        if v is None:
            return None
        if isinstance(v, str):
            try:
                v_dt = datetime.fromisoformat(v.replace('Z', '+00:00')
                    ) if v.endswith('Z') else datetime.fromisoformat(v)
            except ValueError:
                raise ValueError(f"无效的日期时间字符串格式: '{v}'")
        elif isinstance(v, datetime):
            v_dt = v
        else:
            raise TypeError(f'期望字符串或日期时间对象，但得到 {type(v)}')
        if v_dt.tzinfo is None:
            return v_dt.replace(tzinfo=timezone.utc)
        if v_dt.tzinfo != timezone.utc:
            return v_dt.astimezone(timezone.utc)
        return v_dt


    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True
        json_encoders = {datetime: lambda v: v.isoformat().replace('+00:00',
            'Z') if v and v.tzinfo == timezone.utc else v.isoformat() if v else
            None, Path: str}


class ContentBlockDTO(BaseModel):
    """
    代表从文档中解析出的一个内容块的 DTO。
    **规范**: 这是进行内容分析（MD5, SimHash, Semantic）的基本单元。
    """
    block_id: str = Field(default_factory=lambda :
        f'block_{uuid.uuid4().hex}', description=
        '内容块的唯一标识符。**规范**: 使用 UUID 生成。')
    file_id: str = Field(description=
        '此内容块所属的源文件的 file_id。**规范**: 关联到 `FileRecordDTO`。')
    text_content: str = Field(description='内容块的原始文本内容。')
    analysis_text: Optional[str] = Field(default=None, description=
        '用于分析的标准化文本内容。**规范**: 如为 None，分析时默认使用 `text_content`。')
    block_type: BlockType = Field(description=
        '内容块的类型。**规范**: 使用 `BlockType` 枚举。')
    order_in_document: Optional[int] = Field(default=None, ge=0,
        description='内容块在文档中的顺序索引。')
    page_number: Optional[int] = Field(default=None, ge=1, description=
        '内容块所在的页码 (如果适用)。')
    text_hash_md5: Optional[str] = Field(default=None, max_length=32,
        description='`analysis_text` 的 MD5 哈希值。')
    simhash_value: Optional[str] = Field(default=None, pattern=
        '^[0-9a-fA-F]{16}$|^[0-9a-fA-F]{32}$', description=
        '`analysis_text` 的 SimHash 指纹 (64位或128位)。')
    metadata: Dict[str, Any] = Field(default_factory=dict, description=
        '与内容块相关的其他元数据。')

    @model_validator(mode='after')
    @classmethod
    def set_analysis_text_if_none(cls, data: Any) ->Any:
        """**验证器**: 确保 `analysis_text` 如果未提供，则使用 `text_content`。"""
        if isinstance(data, cls):
            if data.analysis_text is None and data.text_content is not None:
                data.analysis_text = data.text_content
        return data


    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True


class AnalysisResultDTO(BaseModel):
    """
    代表对**两个**内容块进行**一种特定分析**后得到的结果的 DTO。
    **规范**: 这是所有分析阶段 (MD5, SimHash, Semantic) 的标准输出格式。
    """
    pair_analysis_id: str = Field(description='基于块对和分析类型的确定性唯一ID。')
    block_id_1: str = Field(description='第一个内容块的 block_id。')
    block_id_2: str = Field(description='第二个内容块的 block_id。')
    analysis_type: AnalysisType = Field(description=
        '执行的分析类型。**规范**: 使用 `AnalysisType` 枚举。')
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0,
        description='标准化相似度分数 (0.0 到 1.0 之间)。')
    details: Dict[str, Any] = Field(default_factory=dict, description=
        '与分析结果相关的其他详细信息。')

    @staticmethod
    def _calculate_pair_analysis_id(block_id_1: str, block_id_2: str,
        analysis_type: Union[AnalysisType, str]) ->str:
        """**辅助方法**: 计算确定性的 ID。"""
        sorted_block_ids = sorted([block_id_1, block_id_2])
        analysis_type_value = analysis_type.value if isinstance(analysis_type,
            Enum) else str(analysis_type)
        key_string = (
            f'pair_{sorted_block_ids[0]}__{sorted_block_ids[1]}_type_{analysis_type_value}'
            )
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()

    @field_validator(mode='before')
    @classmethod
    def set_pair_analysis_id_on_validation(cls, values: Any) ->Any:
        """**验证器**: 在验证前自动计算 `pair_analysis_id`。"""
        if isinstance(values, dict):
            block_id_1 = values.get('block_id_1')
            block_id_2 = values.get('block_id_2')
            analysis_type = values.get('analysis_type')
            if block_id_1 and block_id_2 and analysis_type:
                values['pair_analysis_id'] = cls._calculate_pair_analysis_id(
                    block_id_1, block_id_2, analysis_type)
            elif 'pair_analysis_id' not in values:
                raise ValueError(
                    '无法计算 pair_analysis_id，且未提供 block_id_1, block_id_2, 或 analysis_type。'
                    )
        return values

    @field_validator(mode='after')
    @classmethod
    def check_simhash_details(cls, data: Any) ->Any:
        """**验证器**: 如果分析类型是 SimHash，验证 details 字段是否符合规范。"""
        if isinstance(data, cls):
            if data.analysis_type == AnalysisType.SIMHASH:
                if 'hamming_distance' not in data.details or not isinstance(
                    data.details['hamming_distance'], int):
                    raise ValueError(
                        "SimHash 分析结果必须在 details 中包含整数 'hamming_distance'。")
                if 'hash_bits' not in data.details or not isinstance(data.
                    details['hash_bits'], int):
                    raise ValueError(
                        "SimHash 分析结果必须在 details 中包含整数 'hash_bits'。")
        return data


    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True


class UserDecisionDTO(BaseModel):
    """
    代表用户或系统针对一个分析结果所做的决策的 DTO。
    **规范**: 这是 `DecisionStage` 的核心输出，也是 `CleanupStage` 的输入。
    """
    pair_analysis_id: str = Field(description=
        '此决策针对的 AnalysisResultDTO 的 ID。**规范**: 关联到 `AnalysisResultDTO`。')
    decision: DecisionType = Field(default=DecisionType.UNDECIDED,
        description='做出的具体决策。**规范**: 使用 `DecisionType` 枚举。')
    decided_at: datetime = Field(default_factory=lambda : datetime.now(
        timezone.utc), description='决策时间戳 (UTC)。**规范**: 必须是 UTC。')
    decided_by: Optional[str] = Field(default=None, description=
        "决策者标识符 (e.g., 'system_rule_v1', 'user_admin')。")
    notes: Optional[str] = Field(default=None, max_length=1024, description
        ='决策备注。')

    @field_validator('decided_at', mode='before')
    def ensure_datetime_is_utc(cls, v: Any) ->Optional[datetime]:
        """**验证器**: 确保决策时间是 UTC。"""
        if v is None:
            return None
        if isinstance(v, str):
            try:
                v_dt = datetime.fromisoformat(v.replace('Z', '+00:00')
                    ) if v.endswith('Z') else datetime.fromisoformat(v)
            except ValueError:
                raise ValueError(f"无效的日期时间字符串格式: '{v}'")
        elif isinstance(v, datetime):
            v_dt = v
        else:
            raise TypeError(f'期望字符串或日期时间对象，但得到 {type(v)}')
        if v_dt.tzinfo is None:
            return v_dt.replace(tzinfo=timezone.utc)
        if v_dt.tzinfo != timezone.utc:
            return v_dt.astimezone(timezone.utc)
        return v_dt


    class Config:
        extra = 'forbid'
        json_encoders = {datetime: lambda v: v.isoformat().replace('+00:00',
            'Z') if v and v.tzinfo == timezone.utc else v.isoformat() if v else
            None}
