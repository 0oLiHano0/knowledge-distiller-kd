"""
=================================================
adapter_interface.py.md - Prefilter 阶段接口与 DTO 定义
=================================================

**模块功能**:

- 定义 Prefilter 阶段所需的抽象接口和数据传输对象 (DTOs)。
- 特别是定义 Czkawka 适配器相关的接口和数据结构。

---
"""

import abc
from pathlib import Path
from typing import List
from kd_tool.stages.prefilter.dtos import CzkawkaScanOutputDTO
from pydantic import BaseModel, Field, ConfigDict


class CzkawkaDuplicateResultDTO(BaseModel):
    """
    表示 Czkawka 找到的一组重复文件。
    """

    original_file: Path = Field(..., description="作为基准的原始文件路径")
    duplicates: List[Path] = Field(..., description="与原始文件重复的文件路径列表")
    size_bytes: int = Field(..., description="文件大小（字节）")
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CzkawkaScanOutputDTO(BaseModel):
    """
    表示 Czkawka 扫描操作的完整输出。
    """

    all_scanned_files: List[Path] = Field(
        ..., description="所有被 Czkawka 扫描到的文件列表"
    )
    duplicate_groups: List[CzkawkaDuplicateResultDTO] = Field(
        ..., description="找到的所有重复文件组列表"
    )
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CzkawkaAdapterInterface(abc.ABC):
    """
    WHY: 适配Czkawka CLI工具。
    WHAT: 定义scan_and_find_duplicates方法。
    HOW: 只定义接口，无实现。
    """

    @abc.abstractmethod
    def scan_and_find_duplicates(self) -> CzkawkaScanOutputDTO:
        """
        运行 Czkawka 扫描并查找重复项。

        :return: 包含扫描结果的 CzkawkaScanOutputDTO 对象。
        :raises CzkawkaExecutionError: 如果 Czkawka 执行失败。
        :raises CzkawkaParseError: 如果解析 Czkawka 输出失败。
        """
        raise NotImplementedError
