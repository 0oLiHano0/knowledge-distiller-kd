"""
=================================================
dtos.py - Prefilter Stage 数据传输对象 (v4.6)
=================================================

**模块功能**:

- **核心职责**: 定义 Prefilter Stage (`kd_tool.stages.prefilter`) 及其适配器
              所需的特定数据传输对象。
- **v4.6 核心变更**:
    - **[架构指令]** `CzkawkaDuplicateResultDTO` 和 `CzkawkaScanOutputDTO`
      从原 `adapter_interface.py` 迁移至此。

---
"""

from pathlib import Path
from typing import List
from pydantic import BaseModel, Field, ConfigDict


class CzkawkaDuplicateResultDTO(BaseModel):
    """
    表示 Czkawka 找到的一组重复文件。
    """

    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, validate_assignment=True
    )
    original_file: Path = Field(..., description="作为基准的原始文件路径")
    duplicates: List[Path] = Field(..., description="与原始文件重复的文件路径列表")
    size_bytes: int = Field(..., description="文件大小（字节）", ge=0)


class CzkawkaScanOutputDTO(BaseModel):
    """
    表示 Czkawka 扫描操作的完整输出。
    """

    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, validate_assignment=True
    )
    all_scanned_files: List[Path] = Field(
        ..., description="所有被 Czkawka 扫描到的文件列表"
    )
    duplicate_groups: List[CzkawkaDuplicateResultDTO] = Field(
        ..., description="找到的所有重复文件组列表"
    )
