# kd_tool/storage/settings_models.py
"""
WHY  : 集中定义存储配置。
WHAT : 使用 Pydantic 模型保证类型安全。
HOW  : 通过 DI 把最小配置子集注入存储实现。
"""
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


class StorageBackend(str, Enum):
    SQLITE = "sqlite"  # 未来可扩展 "postgres", "duckdb" 等


class StorageSettingsDTO(BaseModel):
    """WHY 提供配置；WHAT backend、url 等；HOW 由外部注入。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    backend: StorageBackend = Field(default=StorageBackend.SQLITE)
    db_path: Path = Field(default=Path("./kd_tool.db"))
    echo_sql: bool = False
    extras: Optional[dict[str, Any]] = None
    backend_type: str = Field(..., description="存储后端类型")

    # ---------- 验证 ----------
    @field_validator("db_path")
    @classmethod
    def _ensure_parent_exists(cls, v: Path) -> Path:
        if not v.parent.exists():
            raise ValueError("db_path 父目录不存在")
        return v
