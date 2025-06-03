"""
kd_tool/schemas/dtos.py  ‑  KD_Tool 数据传输对象 (DTOs)  v4.6
================================================================
WHY  : 各层之间的数据契约，保证结构化与类型安全。
WHAT : 定义 FileRecordDTO / ContentBlockDTO / AnalysisResultDTO /
       UserDecisionDTO 四个核心 DTO。
HOW  : 依 Pydantic v2 构建，所有字段明确类型与验证，
       严格禁止与 ORM 混用，遵守架构设计总则 v4.0‑v4.6。
- "PipelineContextDTO 仅在 core/core_dtos.py 定义，禁止在此重复定义。"
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------- 业务枚举 & 错误 ----------
from kd_tool.schemas.enums import (
    AnalysisType,
    BlockType,
    DecisionType,
    ProcessingStatus,
)
from kd_tool.core.errors import KDToolError

__all__ = [
    "FileRecordDTO",
    "ContentBlockDTO",
    "AnalysisResultDTO",
    "UserDecisionDTO",
]


# =============================================================================
# FileRecordDTO
# =============================================================================
class FileRecordDTO(BaseModel):
    """WHY  文件唯一标识；WHAT  贯穿全流程；HOW  Pydantic v2 DTO。"""

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, arbitrary_types_allowed=True
    )

    file_id: str = Field(  # noqa: WPS110 (保持业务词)
        default_factory=lambda: f"file_{uuid.uuid4().hex}",
        description="文件唯一标识，UUID‑hex",
    )
    original_path: Path = Field(description="文件系统绝对路径")
    file_hash_md5: Optional[str] = Field(
        default=None,
        max_length=32,
        description="文件内容 MD5，用于精确匹配",
    )
    size_bytes: Optional[int] = Field(default=None, ge=0, description="文件大小，字节")
    last_modified_at: Optional[datetime] = Field(
        default=None,
        description="文件最后修改 UTC 时间",
    )
    registered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="注册时间 UTC",
    )
    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="处理状态枚举",
    )
    processing_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="状态变更历史",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="扩展元数据")

    # ---- 验证 ---------------------------------------------------------------
    @field_validator(
        "last_modified_at", "registered_at", mode="before", check_fields=False
    )
    @classmethod
    def _ensure_utc(cls, v: Any) -> Any:  # noqa: N805 (Pydantic 规范)
        """保证日期字段为 UTC。"""
        if v is None:
            return None
        if isinstance(v, str):
            v_dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        elif isinstance(v, datetime):
            v_dt = v
        else:
            raise TypeError(f"期望 str 或 datetime, 得到 {type(v)}")
        return v_dt.astimezone(timezone.utc)


# =============================================================================
# ContentBlockDTO
# =============================================================================
class ContentBlockDTO(BaseModel):
    """WHY  文档基本分析单元；WHAT  存储文本块与哈希；HOW  DTO。"""

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, arbitrary_types_allowed=True
    )

    block_id: str = Field(
        default_factory=lambda: f"block_{uuid.uuid4().hex}",
        description="内容块唯一标识",
    )
    file_id: str = Field(description="所属 FileRecordDTO.file_id")
    text_content: str = Field(description="原始文本内容")
    analysis_text: Optional[str] = Field(
        default=None,
        description="标准化文本，默认 text_content",
    )
    block_type: BlockType = Field(description="内容块类型枚举")
    order_in_document: Optional[int] = Field(
        default=None, ge=0, description="文档内顺序"
    )
    page_number: Optional[int] = Field(default=None, ge=1, description="页码")
    text_hash_md5: Optional[str] = Field(
        default=None, max_length=32, description="标准化文本 MD5"
    )
    simhash_value: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{16}$|^[0-9a-fA-F]{32}$",
        description="SimHash 指纹 64/128 bit",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="扩展元数据")

    # ---- 自动回填 -----------------------------------------------------------
    @model_validator(mode="after")
    def _fill_analysis_text(
        self,
    ) -> "ContentBlockDTO":  # noqa: D401 (Pydantic 自身返回类型)
        """若 analysis_text 为空，则使用 text_content。"""
        if self.analysis_text is None:
            object.__setattr__(self, "analysis_text", self.text_content)
        return self


# =============================================================================
# AnalysisResultDTO
# =============================================================================
class AnalysisResultDTO(BaseModel):
    """WHY  块对分析结果；WHAT  提供统一输出；HOW  DTO。"""

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, arbitrary_types_allowed=True
    )

    pair_analysis_id: str = Field(description="块对+类型 唯一哈希")
    block_id_1: str = Field(description="内容块一 ID")
    block_id_2: str = Field(description="内容块二 ID")
    analysis_type: AnalysisType = Field(description="分析类型枚举")
    score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="相似度分数"
    )
    details: Dict[str, Any] = Field(default_factory=dict, description="分析附加详情")

    # ---- 内部工具 -----------------------------------------------------------
    @staticmethod
    def _make_id(b1: str, b2: str, a_type: Union[AnalysisType, str]) -> str:
        ids = sorted([b1, b2])
        at_val = a_type.value if isinstance(a_type, Enum) else str(a_type)
        raw = f"pair_{ids[0]}__{ids[1]}_type_{at_val}"
        return hashlib.md5(raw.encode()).hexdigest()

    # ---- 验证 ---------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def _populate_id(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if "pair_analysis_id" not in data:
            cls_fields = (
                data.get("block_id_1"),
                data.get("block_id_2"),
                data.get("analysis_type"),
            )
            if all(cls_fields):
                data["pair_analysis_id"] = cls._make_id(*cls_fields)
            else:
                raise ValueError("缺少生成 pair_analysis_id 所需字段")
        return data

    @model_validator(mode="after")
    def _check_simhash(self) -> "AnalysisResultDTO":
        if self.analysis_type is AnalysisType.SIMHASH:
            if not isinstance(self.details.get("hamming_distance"), int):
                raise ValueError("SimHash 结果 details 必含整数 hamming_distance")
            if not isinstance(self.details.get("hash_bits"), int):
                raise ValueError("SimHash 结果 details 必含整数 hash_bits")
        return self


# =============================================================================
# UserDecisionDTO
# =============================================================================
class UserDecisionDTO(BaseModel):
    """WHY  人机决策记录；WHAT  输入给 CleanupStage；HOW  DTO。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    pair_analysis_id: str = Field(description="对应 AnalysisResultDTO ID")
    decision: DecisionType = Field(
        default=DecisionType.UNDECIDED, description="决策枚举"
    )
    decided_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="决策时间 UTC",
    )
    decided_by: Optional[str] = Field(default=None, description="决策者标识符")
    notes: Optional[str] = Field(default=None, max_length=1024, description="备注")

    # ---- 验证 ---------------------------------------------------------------
    @field_validator("decided_at", mode="before", check_fields=False)
    @classmethod
    def _ensure_utc(cls, v: Any) -> Any:  # noqa: N805
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime):
            return v.astimezone(timezone.utc)
        raise TypeError("decided_at 必须为 datetime 或 ISO str")
