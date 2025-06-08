# kd_tool/storage/models_sqlalchemy.py
"""
WHY  : 存储层内部 ORM 定义，严禁泄漏到外部。
WHAT : 映射 ContentBlock 实体，用于 SQLiteStorage。
HOW  : Declarative Base + 类型提示，便于迁移。
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import Column, DateTime, Integer, LargeBinary, String
from sqlalchemy.orm import declarative_base, Mapped, mapped_column

Base = declarative_base()


class ContentBlockORM(Base):
    """WHY 映射表；WHAT 存储文本块+哈希；HOW 供 SQLiteStorage 使用。"""

    __tablename__ = "content_blocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    md5: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    file_id: Mapped[str] = mapped_column(String(36), nullable=False)  # UUID 长度
    block_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # BlockType 枚举值
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow
    )
