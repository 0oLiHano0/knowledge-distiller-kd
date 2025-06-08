# kd_tool/storage/sqlite_storage.py
"""
WHY  : SQLite 实现 StorageInterface，满足本地部署。
WHAT : 提供事务、CRUD，实现 ORM ↔ DTO 转换。
HOW  : 依赖注入 settings/logger，使用 SQLAlchemy + Session。
"""
from __future__ import annotations

from typing import List, Optional

from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from kd_tool.core.core_dtos import ContentBlockDTO
from kd_tool.storage.errors import (
    StorageInitializationError,
    StorageError,
    DuplicateContentError,
    RecordNotFoundError,
)
from kd_tool.storage.models_sqlalchemy import Base, ContentBlockORM
from kd_tool.storage.settings_models import StorageSettingsDTO
from kd_tool.storage.storage_interface import StorageInterface
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.schemas.enums import BlockType


class SQLiteStorage(StorageInterface):
    """WHY 本地轻量数据库；WHAT 具体实现；HOW 依赖注入实现 SRP。"""

    def __init__(self, settings: StorageSettingsDTO, logger: LoggerProtocol) -> None:
        self._settings = settings
        self._logger = logger
        self._engine = create_engine(
            f"sqlite:///{settings.db_path}", echo=settings.echo_sql, future=True
        )
        self._Session: sessionmaker[Session] = sessionmaker(
            bind=self._engine, autoflush=False, expire_on_commit=False, future=True
        )

    # ---------- LifeCycle ----------
    def initialize(self) -> None:
        """创建表结构。"""
        try:
            self._logger.debug("SQLite 初始化: 创建表...")
            Base.metadata.create_all(self._engine)
        except SQLAlchemyError as e:
            self._logger.exception("SQLite 初始化失败")
            raise StorageInitializationError("sqlite", e) from e

    # ---------- 事务 ----------
    def begin_transaction(self) -> None:
        self._session = self._Session()
        self._session.begin()

    def commit_transaction(self) -> None:
        try:
            self._session.commit()
        except SQLAlchemyError as e:
            self._logger.exception("事务提交失败")
            self._session.rollback()
            raise StorageError("事务提交失败") from e
        finally:
            self._session.close()

    def rollback_transaction(self) -> None:
        self._session.rollback()
        self._session.close()

    # ---------- CRUD ----------
    def save_content_blocks(self, blocks: List[ContentBlockDTO]) -> None:
        """
        批量写入。
        强制要求：如检测到外部事务（self._session存在），所有写操作不得自动commit，必须由外部统一提交。
        否则（无事务）自动管理session和commit。
        """
        if hasattr(self, "_session") and self._session is not None:
            session = self._session
            for dto in blocks:
                obj = ContentBlockORM(
                    md5=dto.text_hash_md5,
                    content=dto.text_content.encode("utf-8"),
                    file_id=dto.file_id,
                    block_type=dto.block_type.value
                )
                session.add(obj)
        else:
            with self._Session() as session:
                for dto in blocks:
                    obj = ContentBlockORM(
                        md5=dto.text_hash_md5,
                        content=dto.text_content.encode("utf-8"),
                        file_id=dto.file_id,
                        block_type=dto.block_type.value
                    )
                    session.add(obj)
                session.commit()

    def get_content_block(self, md5: str) -> Optional[ContentBlockDTO]:
        with self._Session() as session:
            stmt = select(ContentBlockORM).where(ContentBlockORM.md5 == md5)
            obj = session.scalar(stmt)
            if not obj:
                return None
            return ContentBlockDTO(
                text_hash_md5=obj.md5,
                text_content=obj.content.decode("utf-8"),
                file_id=obj.file_id,
                block_type=BlockType(obj.block_type)
            )

    # ---------- Close ----------
    def close(self) -> None:
        self._engine.dispose()
