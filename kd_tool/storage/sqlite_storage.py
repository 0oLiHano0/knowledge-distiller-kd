"""
=================================================
sqlite_storage.py - SQLite 存储实现 (v4.6)
=================================================

**模块功能**:

- `StorageInterface` 的 SQLite 具体实现。
- 使用 SQLAlchemy 操作 SQLite 数据库。

**架构决策与约束**:
- ... (原有约束保持) ...
- **v4.6 核心变更**:
    - **[架构指令]** 所有方法实现 **必须** 与更新后的 `StorageInterface` (v4.6) 签名保持一致。
    - **[架构指令]** DTOs (`FileRecordDTO`, `ContentBlockDTO`, `AnalysisResultDTO`, `UserDecisionDTO`)
                   的转换和使用 **必须** 反映它们已不再包含 `task_id` 字段。
    - **[架构指令]** `save_content_blocks` **必须** 移除 `file_id` 参数，依赖 DTO 内部信息。
    - **[架构指令]** `save_user_decisions` **必须** 实现为批量操作。
    - **[架构指令]** 新增 `delete_file_records` 方法的实现。
    - **[架构指令]** DTO 与 ORM 转换方法需同步更新。
    - **[架构指令]** 返回持久化后的 DTO 列表（如果接口定义如此）。
**主要变更点解释：**

1. **DTOs 的 `task_id`**：所有 DTO<->ORM 转换方法 (`_convert_...`) 保持不变，因为 ORM 模型本身就没有 `task_id`，而 DTOs 现在也已经移除了 `task_id`。
2. **`register_file`**：
    - 签名变为 `register_file(self, file_dto: FileRecordDTO) -> FileRecordDTO`。
    - 实现逻辑调整为直接使用 `file_dto` 的属性，不再有单独的 `file_path` 和 `metadata` 参数。
3. **`register_files`**：
    - 签名保持 `register_files(self, files_data: List[FileRecordDTO]) -> List[FileRecordDTO]`。
    - 其内部逻辑已能处理不含 `task_id` 的 `FileRecordDTO`。关键在于它会尝试根据 `file_id` 或 `original_path` 更新现有记录，或创建新记录。返回值也更新为 `List[FileRecordDTO]`。
    - **重要实现细节**：为了正确返回持久化后的DTOs，在批量操作中，`session.flush()`之后，需要一种方式确保获取到的是数据库中最新状态的ORM实例，然后再转换为DTO。我调整了实现，使其在 `flush` 后重新查询这些记录。
4. **`save_content_blocks`**:
    - 签名变为 `save_content_blocks(self, blocks_data: List[ContentBlockDTO]) -> List[ContentBlockDTO]`。
    - 移除了 `file_id` 参数。实现时，会遍历 `blocks_data`，并从每个 `ContentBlockDTO` 中获取其 `file_id`来检查关联的 `FileOrmModel` 是否存在。
    - 返回值变为 `List[ContentBlockDTO]`。
5. **`save_analysis_results`**:
    - 签名保持 `save_analysis_results(self, results_data: List[AnalysisResultDTO])`。
    - 返回值变为 `List[AnalysisResultDTO]`。
6. **`save_user_decisions`** (原 `save_user_decision`):
    - 方法重命名为 `save_user_decisions`。
    - 签名变为 `save_user_decisions(self, decisions_data: List[UserDecisionDTO]) -> List[UserDecisionDTO]`。
    - 实现已改为迭代处理 `decisions_data` 列表中的每个 `UserDecisionDTO`，进行保存或更新。
    - 返回值变为 `List[UserDecisionDTO]`。
7. **`delete_file_records`**:
    - 新增此方法的实现，使用 `sqlalchemy.delete()` 进行批量删除。
8. **返回持久化后的 DTOs**: 所有 `save_...` 和 `register_...` 方法现在都会在 `session.flush()` 后，将持久化的 ORM 对象转换回 DTO 并返回。这确保了调用者能获得包含任何数据库生成/更新的值（如主键、默认时间戳）的最新 DTO。
---
"""
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
from loguru import Logger
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, NoResultFound
from sqlalchemy.orm.attributes import flag_modified
from kd_tool.storage.settings_models import StorageSettingsDTO
from kd_tool.storage.storage_interface import StorageInterface
from kd_tool.storage.models_sqlalchemy import Base, FileOrmModel, ContentBlockOrmModel, AnalysisResultOrmModel, UserDecisionOrmModel
from kd_tool.schemas.dtos import FileRecordDTO, ContentBlockDTO, AnalysisResultDTO, UserDecisionDTO
from kd_tool.schemas.enums import ProcessingStatus
from kd_tool.storage.errors import StorageError, StorageConfigurationError, StorageConnectionError, StorageOperationError, RecordNotFoundError, DuplicateRecordError, TransactionError


class SQLiteStorage(StorageInterface):
    """
    使用 SQLite 和 SQLAlchemy 实现 StorageInterface。
    """

    def __init__(self, settings: StorageSettingsDTO, logger: Logger):
        super().__init__(settings, logger)
        self.logger.info(f'SQLiteStorage: 正在配置实例 (父类初始化已完成)...')
        if not isinstance(self.settings, StorageSettingsDTO):
            msg = (
                'SQLiteStorage: settings 参数必须是 StorageSettingsDTO 的实例 (由父类初始化后检查)。'
                )
            self.logger.error(msg)
            raise TypeError(msg)
        if not isinstance(self.logger, Logger):
            msg = 'SQLiteStorage: logger 参数必须是兼容的 Logger 实例 (由父类初始化后检查)。'
            self.logger.error(msg)
            raise TypeError(msg)
        if self.settings.backend_type != 'sqlite':
            msg = (
                f"SQLiteStorage: 配置的后端类型为 '{self.settings.backend_type}'，但此类仅支持 'sqlite'。"
                )
            self.logger.error(msg)
            raise StorageConfigurationError(msg)
        if not self.settings.connection_string:
            msg = "SQLiteStorage: 初始化失败，配置中必须提供 'connection_string'。"
            self.logger.error(msg)
            raise StorageConfigurationError(msg)
        self.engine: Optional[create_engine] = None
        self.SessionLocal: Optional[sessionmaker[Session]] = None
        self._current_session: Optional[Session] = None
        self.logger.debug(
            f"SQLiteStorage: 实例已配置，连接字符串: '{self.settings.connection_string}'。数据库连接和会话工厂将在 initialize() 中建立。"
            )

    def initialize(self) ->None:
        if self.engine is not None and self.SessionLocal is not None:
            self.logger.info('SQLiteStorage: 已初始化，跳过重复初始化。')
            return
        self.logger.info(
            f'SQLiteStorage: 正在执行数据库引擎和会话工厂初始化，连接目标: {self.settings.connection_string}'
            )
        try:
            self.engine = create_engine(self.settings.connection_string,
                connect_args={'check_same_thread': False})
            self.logger.debug('SQLiteStorage: SQLAlchemy 引擎已创建。')

            @event.listens_for(self.engine, 'connect')
            def _enable_sqlite_foreign_keys(dbapi_connection, connection_record
                ):
                print(
                    f'SQLiteStorage: [EVENT] Setting PRAGMA foreign_keys=ON for new connection.'
                    )
                cursor = dbapi_connection.cursor()
                try:
                    cursor.execute('PRAGMA foreign_keys=ON')
                finally:
                    cursor.close()
            self.logger.debug('SQLiteStorage: PRAGMA foreign_keys=ON 事件监听器已设置。'
                )
            self.logger.info(
                'SQLiteStorage: 表结构创建和迁移由 Alembic 在应用层面负责，initialize() 跳过此步骤。')
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=
                False, bind=self.engine)
            self.logger.debug(
                'SQLiteStorage: SQLAlchemy 会话工厂 (SessionLocal) 已创建。')
            self.logger.info('SQLiteStorage: 数据库引擎和会话工厂初始化成功。')
        except SQLAlchemyError as e:
            self.logger.exception(
                f"SQLiteStorage: 初始化过程中发生 SQLAlchemy 错误。连接字符串: '{self.settings.connection_string}'. 错误: {e}"
                )
            raise StorageConnectionError(
                f'连接到 {self.settings.connection_string} 失败',
                original_exception=e) from e
        except Exception as e:
            self.logger.exception(f'SQLiteStorage: 初始化过程中发生未知错误: {e}')
            raise StorageOperationError(operation=
                'initialize_storage_engine_unknown', original_exception=e
                ) from e

    def finalize(self) ->None:
        self.logger.info('SQLiteStorage: 正在执行清理 (finalize)...')
        if self._current_session:
            if self._current_session.is_active:
                self.logger.warning('SQLiteStorage: 清理时发现活动的显式事务，将执行回滚。')
                try:
                    self._current_session.rollback()
                    self.logger.debug('SQLiteStorage: 活动的显式事务已回滚。')
                except SQLAlchemyError as e_rollback:
                    self.logger.exception(
                        f'SQLiteStorage: 在 finalize 期间回滚活动事务失败: {e_rollback}')
            try:
                self._current_session.close()
                self.logger.debug('SQLiteStorage: _current_session 已关闭。')
            except SQLAlchemyError as e_close:
                self.logger.exception(
                    f'SQLiteStorage: 在 finalize 期间关闭 _current_session 失败: {e_close}'
                    )
            finally:
                self._current_session = None
        if self.engine:
            self.logger.debug(f'SQLiteStorage: 正在释放 SQLAlchemy 引擎 (连接池)...')
            try:
                self.engine.dispose()
                self.logger.info('SQLiteStorage: SQLAlchemy 引擎已成功释放。')
            except SQLAlchemyError as e_dispose:
                self.logger.exception(
                    f'SQLiteStorage: 释放 SQLAlchemy 引擎时发生错误: {e_dispose}')
            finally:
                self.engine = None
        else:
            self.logger.debug('SQLiteStorage: SQLAlchemy 引擎未初始化或已释放，无需操作。')
        if self.SessionLocal:
            self.SessionLocal = None
            self.logger.debug('SQLiteStorage: 会话工厂 (SessionLocal) 已清理。')
        self.logger.info('SQLiteStorage: 清理操作完成。')

    @contextmanager
    def _session_scope(self) ->Session:
        """
        提供一个 SQLAlchemy 会话的上下文管理器。
        (详细架构约束见 v4 版本)
        """
        if self._current_session and self._current_session.is_active:
            self.logger.trace('SQLiteStorage._session_scope: 复用活动的显式事务会话。')
            yield self._current_session
            return
        if not self.SessionLocal:
            msg = (
                'SQLiteStorage._session_scope: SessionLocal 未初始化。请先调用 initialize()。'
                )
            self.logger.error(msg)
            raise StorageConnectionError(msg)
        session = self.SessionLocal()
        self.logger.trace('SQLiteStorage._session_scope: 新会话已创建。')
        try:
            yield session
            session.commit()
            self.logger.trace('SQLiteStorage._session_scope: 会话已提交。')
        except SQLAlchemyError as e:
            self.logger.exception(
                f'SQLiteStorage._session_scope: 会话中发生 SQLAlchemyError，将回滚。错误: {e}'
                )
            session.rollback()
            if isinstance(e, IntegrityError):
                raise DuplicateRecordError(record_type=
                    'UnknownInSessionScope', record_identifier='Unknown',
                    details=str(e)) from e
            raise StorageOperationError(operation=
                '_session_scope_operation', original_exception=e) from e
        except Exception as e:
            self.logger.exception(
                f'SQLiteStorage._session_scope: 会话中发生错误，将回滚。错误: {type(e).__name__}: {e}'
                )
            session.rollback()
            if isinstance(e, StorageError):
                raise
            raise StorageOperationError(operation=
                '_session_scope_unknown_error', original_exception=e) from e
        finally:
            self.logger.trace('SQLiteStorage._session_scope: 会话正在关闭。')
            session.close()

    def _convert_file_record_orm_to_dto(self, orm_instance: FileOrmModel
        ) ->FileRecordDTO:
        """将 FileOrmModel 转换为 FileRecordDTO。"""
        self.logger.trace(
            f'SQLiteStorage: 转换 FileOrmModel (ID: {orm_instance.file_id}) 到 DTO。'
            )
        return FileRecordDTO(file_id=orm_instance.file_id, original_path=
            Path(orm_instance.original_path), file_hash_md5=orm_instance.
            file_hash_md5, size_bytes=orm_instance.size_bytes,
            last_modified_at=orm_instance.last_modified_at, registered_at=
            orm_instance.registered_at, processing_status=ProcessingStatus(
            orm_instance.processing_status), processing_history=list(
            orm_instance.processing_history or []), metadata=dict(
            orm_instance.metadata_ or {}))

    def _convert_file_record_dto_to_orm(self, dto: FileRecordDTO,
        existing_orm: Optional[FileOrmModel]=None) ->FileOrmModel:
        """将 FileRecordDTO 转换为 FileOrmModel (可用于新建或更新)。"""
        self.logger.trace(
            f'SQLiteStorage: 转换 FileRecordDTO (ID: {dto.file_id}) 到 ORM。')
        orm = existing_orm or FileOrmModel()
        if not existing_orm:
            orm.file_id = dto.file_id
        orm.original_path = str(dto.original_path)
        orm.file_hash_md5 = dto.file_hash_md5
        orm.size_bytes = dto.size_bytes
        orm.last_modified_at = dto.last_modified_at
        orm.registered_at = dto.registered_at
        orm.processing_status = dto.processing_status
        orm.processing_history = dto.processing_history
        orm.metadata_ = dto.metadata
        return orm

    def _convert_content_block_orm_to_dto(self, orm_instance:
        ContentBlockOrmModel) ->ContentBlockDTO:
        self.logger.trace(
            f'SQLiteStorage: 转换 ContentBlockOrmModel (ID: {orm_instance.block_id}) 到 DTO。'
            )
        return ContentBlockDTO(block_id=orm_instance.block_id, file_id=
            orm_instance.file_id, text_content=orm_instance.text_content,
            analysis_text=orm_instance.analysis_text, block_type=
            orm_instance.block_type, order_in_document=orm_instance.
            order_in_document, page_number=orm_instance.page_number,
            text_hash_md5=orm_instance.text_hash_md5, simhash_value=
            orm_instance.simhash_value, metadata=dict(orm_instance.
            metadata_ or {}))

    def _convert_content_block_dto_to_orm(self, dto: ContentBlockDTO,
        existing_orm: Optional[ContentBlockOrmModel]=None
        ) ->ContentBlockOrmModel:
        self.logger.trace(
            f'SQLiteStorage: 转换 ContentBlockDTO (ID: {dto.block_id}) 到 ORM。')
        orm = existing_orm or ContentBlockOrmModel()
        if not existing_orm:
            orm.block_id = dto.block_id
        orm.file_id = dto.file_id
        orm.text_content = dto.text_content
        orm.analysis_text = dto.analysis_text
        orm.block_type = dto.block_type
        orm.order_in_document = dto.order_in_document
        orm.page_number = dto.page_number
        orm.text_hash_md5 = dto.text_hash_md5
        orm.simhash_value = dto.simhash_value
        orm.metadata_ = dto.metadata
        return orm

    def _convert_analysis_result_orm_to_dto(self, orm_instance:
        AnalysisResultOrmModel) ->AnalysisResultDTO:
        self.logger.trace(
            f'SQLiteStorage: 转换 AnalysisResultOrmModel (ID: {orm_instance.pair_analysis_id}) 到 DTO。'
            )
        return AnalysisResultDTO(block_id_1=orm_instance.block_id_1,
            block_id_2=orm_instance.block_id_2, analysis_type=orm_instance.
            analysis_type, score=orm_instance.score, details=dict(
            orm_instance.details or {}), pair_analysis_id=orm_instance.
            pair_analysis_id)

    def _convert_analysis_result_dto_to_orm(self, dto: AnalysisResultDTO,
        existing_orm: Optional[AnalysisResultOrmModel]=None
        ) ->AnalysisResultOrmModel:
        self.logger.trace(
            f'SQLiteStorage: 转换 AnalysisResultDTO (ID: {dto.pair_analysis_id}) 到 ORM。'
            )
        orm = existing_orm or AnalysisResultOrmModel(pair_analysis_id=dto.
            pair_analysis_id)
        if not existing_orm:
            orm.pair_analysis_id = dto.pair_analysis_id
        orm.block_id_1 = dto.block_id_1
        orm.block_id_2 = dto.block_id_2
        orm.analysis_type = dto.analysis_type
        orm.score = dto.score
        orm.details = dto.details
        return orm

    def _convert_user_decision_orm_to_dto(self, orm_instance:
        UserDecisionOrmModel) ->UserDecisionDTO:
        self.logger.trace(
            f'SQLiteStorage: 转换 UserDecisionOrmModel (Pair ID: {orm_instance.pair_analysis_id}) 到 DTO。'
            )
        return UserDecisionDTO(pair_analysis_id=orm_instance.
            pair_analysis_id, decision=orm_instance.decision, decided_at=
            orm_instance.decided_at, decided_by=orm_instance.decided_by,
            notes=orm_instance.notes)

    def _convert_user_decision_dto_to_orm(self, dto: UserDecisionDTO,
        existing_orm: Optional[UserDecisionOrmModel]=None
        ) ->UserDecisionOrmModel:
        self.logger.trace(
            f'SQLiteStorage: 转换 UserDecisionDTO (Pair ID: {dto.pair_analysis_id}) 到 ORM。'
            )
        orm = existing_orm or UserDecisionOrmModel(pair_analysis_id=dto.
            pair_analysis_id)
        if not existing_orm:
            orm.pair_analysis_id = dto.pair_analysis_id
        orm.decision = dto.decision
        orm.decided_at = dto.decided_at
        orm.decided_by = dto.decided_by
        orm.notes = dto.notes
        return orm

    def register_file(self, file_dto: FileRecordDTO) ->FileRecordDTO:
        self.logger.info(f'SQLiteStorage: 尝试注册文件: {file_dto.original_path}')
        operation = 'register_file'
        try:
            with self._session_scope() as session:
                self.logger.debug(
                    f"{operation}: 检查路径 '{file_dto.original_path}' 是否已存在。")
                existing_orm = session.query(FileOrmModel).filter(
                    FileOrmModel.original_path == str(file_dto.original_path)
                    ).one_or_none()
                if existing_orm:
                    self.logger.info(
                        f"{operation}: 文件 '{file_dto.original_path}' 已注册 (ID: {existing_orm.file_id})。"
                        )
                    updated_orm = self._convert_file_record_dto_to_orm(file_dto
                        , existing_orm=existing_orm)
                    session.add(updated_orm)
                    session.flush()
                    return self._convert_file_record_orm_to_dto(updated_orm)
                self.logger.debug(
                    f"{operation}: 文件 '{file_dto.original_path}' 未注册，创建新记录。")
                new_orm = self._convert_file_record_dto_to_orm(file_dto)
                session.add(new_orm)
                session.flush()
                self.logger.info(
                    f"{operation}: 文件 '{file_dto.original_path}' 已新注册为 ID '{new_orm.file_id}'。"
                    )
                return self._convert_file_record_orm_to_dto(new_orm)
        except IntegrityError as e:
            self.logger.error(
                f"{operation}: 注册文件 '{file_dto.original_path}' 时发生完整性错误: {e}")
            raise DuplicateRecordError(record_type='FileRecord',
                record_identifier=file_dto.original_path, details=str(e)
                ) from e
        except SQLAlchemyError as e:
            self.logger.exception(
                f"{operation}: 注册文件 '{file_dto.original_path}' 时发生数据库错误。")
            raise StorageOperationError(operation=operation,
                original_exception=e) from e
        except Exception as e:
            self.logger.exception(
                f"{operation}: 注册文件 '{file_dto.original_path}' 时发生未知错误。")
            if isinstance(e, StorageError):
                raise
            raise StorageOperationError(operation=f'{operation}_unknown',
                original_exception=e) from e

    def register_files(self, files_data: List[FileRecordDTO]) ->List[
        FileRecordDTO]:
        self.logger.info(f'SQLiteStorage: 批量注册/更新 {len(files_data)} 个文件记录。')
        operation = 'register_files'
        try:
            with self._session_scope() as session:
                result_dtos = []
                for file_dto in files_data:
                    existing_orm = None
                    if file_dto.file_id:
                        existing_orm = session.query(FileOrmModel).filter(
                            FileOrmModel.file_id == file_dto.file_id
                            ).one_or_none()
                    if not existing_orm:
                        existing_orm = session.query(FileOrmModel).filter(
                            FileOrmModel.original_path == str(file_dto.
                            original_path)).one_or_none()
                    if existing_orm:
                        updated_orm = self._convert_file_record_dto_to_orm(
                            file_dto, existing_orm=existing_orm)
                        session.add(updated_orm)
                    else:
                        new_orm = self._convert_file_record_dto_to_orm(file_dto
                            )
                        session.add(new_orm)
                session.flush()
                file_ids = [f.file_id for f in files_data]
                orms = session.query(FileOrmModel).filter(FileOrmModel.
                    file_id.in_(file_ids)).all()
                result_dtos = [self._convert_file_record_orm_to_dto(orm) for
                    orm in orms]
                return result_dtos
        except IntegrityError as e:
            self.logger.error(f'{operation}: 批量注册文件时发生完整性错误: {e}')
            raise DuplicateRecordError(record_type='FileRecord',
                record_identifier='batch', details=str(e)) from e
        except SQLAlchemyError as e:
            self.logger.exception(f'{operation}: 批量注册文件时发生数据库错误。')
            raise StorageOperationError(operation=operation,
                original_exception=e) from e
        except Exception as e:
            self.logger.exception(f'{operation}: 批量注册文件时发生未知错误。')
            if isinstance(e, StorageError):
                raise
            raise StorageOperationError(operation=f'{operation}_unknown',
                original_exception=e) from e

    def delete_file_records(self, file_ids: List[str]) ->int:
        self.logger.info(f'SQLiteStorage: 批量删除文件记录: {file_ids}')
        operation = 'delete_file_records'
        try:
            with self._session_scope() as session:
                result = session.query(FileOrmModel).filter(FileOrmModel.
                    file_id.in_(file_ids)).delete(synchronize_session=False)
                self.logger.info(f'{operation}: 成功删除 {result} 条文件记录。')
                return result
        except SQLAlchemyError as e:
            self.logger.exception(f'{operation}: 批量删除文件记录时发生数据库错误。')
            raise StorageOperationError(operation=operation,
                original_exception=e) from e
        except Exception as e:
            self.logger.exception(f'{operation}: 批量删除文件记录时发生未知错误。')
            if isinstance(e, StorageError):
                raise
            raise StorageOperationError(operation=f'{operation}_unknown',
                original_exception=e) from e

    def save_content_blocks(self, blocks_data: List[ContentBlockDTO]) ->List[
        ContentBlockDTO]:
        self.logger.info(f'SQLiteStorage: 尝试保存 {len(blocks_data)} 个内容块。')
        operation = 'save_content_blocks'
        try:
            with self._session_scope() as session:
                result_dtos = []
                for block_dto in blocks_data:
                    file_exists = session.query(FileOrmModel.file_id).filter(
                        FileOrmModel.file_id == block_dto.file_id).scalar(
                        ) is not None
                    if not file_exists:
                        self.logger.error(
                            f'{operation}: 关联的文件记录 (ID: {block_dto.file_id}) 未找到。'
                            )
                        raise RecordNotFoundError(record_type='FileRecord',
                            record_id=block_dto.file_id)
                    existing_orm_block = session.query(ContentBlockOrmModel
                        ).filter(ContentBlockOrmModel.block_id == block_dto
                        .block_id).one_or_none()
                    orm_block = self._convert_content_block_dto_to_orm(
                        block_dto, existing_orm=existing_orm_block)
                    session.add(orm_block)
                session.flush()
                block_ids = [b.block_id for b in blocks_data]
                orms = session.query(ContentBlockOrmModel).filter(
                    ContentBlockOrmModel.block_id.in_(block_ids)).all()
                result_dtos = [self._convert_content_block_orm_to_dto(orm) for
                    orm in orms]
                return result_dtos
        except RecordNotFoundError as e:
            self.logger.error(f'{operation}: 保存内容块失败，因为关联文件未找到: {e}')
            raise
        except IntegrityError as e:
            self.logger.error(f'{operation}: 保存内容块时发生完整性错误: {e}')
            raise DuplicateRecordError(record_type='ContentBlock',
                record_identifier='batch', details=str(e)) from e
        except SQLAlchemyError as e:
            self.logger.exception(f'{operation}: 保存内容块时发生数据库错误。')
            raise StorageOperationError(operation=operation,
                original_exception=e) from e
        except Exception as e:
            self.logger.exception(f'{operation}: 保存内容块时发生未知错误。')
            if isinstance(e, StorageError):
                raise
            raise StorageOperationError(operation=f'{operation}_unknown',
                original_exception=e) from e

    def get_content_block(self, block_id: str) ->Optional[ContentBlockDTO]:
        self.logger.debug(f'SQLiteStorage: 尝试获取内容块: {block_id}')
        operation = 'get_content_block'
        try:
            with self._session_scope() as session:
                orm_instance = session.query(ContentBlockOrmModel).filter(
                    ContentBlockOrmModel.block_id == block_id).one_or_none()
                if orm_instance:
                    return self._convert_content_block_orm_to_dto(orm_instance)
                return None
        except SQLAlchemyError as e:
            self.logger.exception(
                f'{operation}: 获取内容块 (ID: {block_id}) 时发生数据库错误。')
            raise StorageOperationError(operation=operation,
                original_exception=e) from e
        except Exception as e:
            self.logger.exception(
                f'{operation}: 获取内容块 (ID: {block_id}) 时发生未知错误。')
            if isinstance(e, StorageError):
                raise
            raise StorageOperationError(operation=f'{operation}_unknown',
                original_exception=e) from e

    def get_content_blocks_for_file(self, file_id: str, criteria: Optional[
        Dict[str, Any]]=None) ->List[ContentBlockDTO]:
        self.logger.debug(
            f'SQLiteStorage: 尝试获取文件 {file_id} 的内容块，条件: {criteria}')
        operation = 'get_content_blocks_for_file'
        try:
            with self._session_scope() as session:
                query = session.query(ContentBlockOrmModel).filter(
                    ContentBlockOrmModel.file_id == file_id)
                if criteria:
                    self.logger.trace(f'{operation}: 应用查询条件: {criteria}')
                    for key, value in criteria.items():
                        if hasattr(ContentBlockOrmModel, key):
                            query = query.filter(getattr(
                                ContentBlockOrmModel, key) == value)
                        else:
                            self.logger.warning(
                                f'{operation}: 未知或不支持的查询条件键: {key}')
                all_orms = query.order_by(ContentBlockOrmModel.
                    order_in_document.asc().nullslast()).all()
                return [self._convert_content_block_orm_to_dto(orm) for orm in
                    all_orms]
        except SQLAlchemyError as e:
            self.logger.exception(
                f'{operation}: 获取文件 (ID: {file_id}) 的内容块时发生数据库错误。')
            raise StorageOperationError(operation=operation,
                original_exception=e) from e
        except Exception as e:
            self.logger.exception(
                f'{operation}: 获取文件 (ID: {file_id}) 的内容块时发生未知错误。')
            if isinstance(e, StorageError):
                raise
            raise StorageOperationError(operation=f'{operation}_unknown',
                original_exception=e) from e

    def get_all_content_blocks(self, criteria: Optional[Dict[str, Any]]=None
        ) ->List[ContentBlockDTO]:
        self.logger.debug(f'SQLiteStorage: 尝试获取所有内容块，条件: {criteria}')
        self.logger.warning(
            'SQLiteStorage: get_all_content_blocks 可能返回大量数据，请谨慎使用或考虑实现分页。')
        operation = 'get_all_content_blocks'
        try:
            with self._session_scope() as session:
                query = session.query(ContentBlockOrmModel)
                if criteria:
                    self.logger.trace(f'{operation}: 应用查询条件: {criteria}')
                    for key, value in criteria.items():
                        if hasattr(ContentBlockOrmModel, key):
                            query = query.filter(getattr(
                                ContentBlockOrmModel, key) == value)
                        else:
                            self.logger.warning(
                                f'{operation}: 未知或不支持的查询条件键: {key}')
                all_orms = query.all()
                return [self._convert_content_block_orm_to_dto(orm) for orm in
                    all_orms]
        except SQLAlchemyError as e:
            self.logger.exception(f'{operation}: 获取所有内容块时发生数据库错误。')
            raise StorageOperationError(operation=operation,
                original_exception=e) from e
        except Exception as e:
            self.logger.exception(f'{operation}: 获取所有内容块时发生未知错误。')
            if isinstance(e, StorageError):
                raise
            raise StorageOperationError(operation=f'{operation}_unknown',
                original_exception=e) from e

    def save_analysis_results(self, results_data: List[AnalysisResultDTO]
        ) ->List[AnalysisResultDTO]:
        self.logger.info(f'SQLiteStorage: 尝试保存 {len(results_data)} 个分析结果。')
        operation = 'save_analysis_results'
        try:
            with self._session_scope() as session:
                for result_dto in results_data:
                    existing_orm = session.query(AnalysisResultOrmModel
                        ).filter(AnalysisResultOrmModel.pair_analysis_id ==
                        result_dto.pair_analysis_id).one_or_none()
                    orm_result = self._convert_analysis_result_dto_to_orm(
                        result_dto, existing_orm=existing_orm)
                    session.add(orm_result)
                session.flush()
                pair_ids = [r.pair_analysis_id for r in results_data]
                orms = session.query(AnalysisResultOrmModel).filter(
                    AnalysisResultOrmModel.pair_analysis_id.in_(pair_ids)).all(
                    )
                result_dtos = [self._convert_analysis_result_orm_to_dto(orm
                    ) for orm in orms]
                return result_dtos
        except IntegrityError as e:
            self.logger.error(f'{operation}: 保存分析结果时发生完整性错误: {e}')
            raise StorageOperationError(operation=operation,
                original_exception=e, details='一个或多个 block_id 可能无效') from e
        except SQLAlchemyError as e:
            self.logger.exception(f'{operation}: 保存分析结果时发生数据库错误。')
            raise StorageOperationError(operation=operation,
                original_exception=e) from e
        except Exception as e:
            self.logger.exception(f'{operation}: 保存分析结果时发生未知错误。')
            if isinstance(e, StorageError):
                raise
            raise StorageOperationError(operation=f'{operation}_unknown',
                original_exception=e) from e

    def get_analysis_results(self, criteria: Optional[Dict[str, Any]]=None
        ) ->List[AnalysisResultDTO]:
        self.logger.debug(f'SQLiteStorage: 尝试获取分析结果，条件: {criteria}')
        operation = 'get_analysis_results'
        try:
            with self._session_scope() as session:
                query = session.query(AnalysisResultOrmModel)
                if criteria:
                    self.logger.trace(f'{operation}: 应用查询条件: {criteria}')
                    for key, value in criteria.items():
                        if hasattr(AnalysisResultOrmModel, key):
                            query = query.filter(getattr(
                                AnalysisResultOrmModel, key) == value)
                        else:
                            self.logger.warning(
                                f'{operation}: 未知或不支持的查询条件键: {key}')
                all_orms = query.all()
                return [self._convert_analysis_result_orm_to_dto(orm) for
                    orm in all_orms]
        except SQLAlchemyError as e:
            self.logger.exception(f'{operation}: 获取分析结果时发生数据库错误。')
            raise StorageOperationError(operation=operation,
                original_exception=e) from e
        except Exception as e:
            self.logger.exception(f'{operation}: 获取分析结果时发生未知错误。')
            if isinstance(e, StorageError):
                raise
            raise StorageOperationError(operation=f'{operation}_unknown',
                original_exception=e) from e

    def save_user_decisions(self, decisions_data: List[UserDecisionDTO]
        ) ->List[UserDecisionDTO]:
        self.logger.info(f'SQLiteStorage: 批量保存 {len(decisions_data)} 个用户决策。')
        operation = 'save_user_decisions'
        try:
            with self._session_scope() as session:
                for decision_dto in decisions_data:
                    existing_orm = session.query(UserDecisionOrmModel).filter(
                        UserDecisionOrmModel.pair_analysis_id ==
                        decision_dto.pair_analysis_id).one_or_none()
                    if not existing_orm:
                        analysis_result_exists = session.query(
                            AnalysisResultOrmModel.pair_analysis_id).filter(
                            AnalysisResultOrmModel.pair_analysis_id ==
                            decision_dto.pair_analysis_id).scalar() is not None
                        if not analysis_result_exists:
                            self.logger.error(
                                f'{operation}: 关联的分析结果 (PairID: {decision_dto.pair_analysis_id}) 未找到。无法保存决策。'
                                )
                            raise RecordNotFoundError(record_type=
                                'AnalysisResult', record_id=decision_dto.
                                pair_analysis_id)
                    orm_decision = self._convert_user_decision_dto_to_orm(
                        decision_dto, existing_orm=existing_orm)
                    session.add(orm_decision)
                session.flush()
                pair_ids = [d.pair_analysis_id for d in decisions_data]
                orms = session.query(UserDecisionOrmModel).filter(
                    UserDecisionOrmModel.pair_analysis_id.in_(pair_ids)).all()
                result_dtos = [self._convert_user_decision_orm_to_dto(orm) for
                    orm in orms]
                return result_dtos
        except RecordNotFoundError as e:
            self.logger.error(f'{operation}: 批量保存用户决策失败，因为关联分析结果未找到: {e}')
            raise
        except IntegrityError as e:
            self.logger.error(f'{operation}: 批量保存用户决策时发生完整性错误: {e}')
            raise StorageOperationError(operation=operation,
                original_exception=e, details='一个或多个 pair_analysis_id 可能无效'
                ) from e
        except SQLAlchemyError as e:
            self.logger.exception(f'{operation}: 批量保存用户决策时发生数据库错误。')
            raise StorageOperationError(operation=operation,
                original_exception=e) from e
        except Exception as e:
            self.logger.exception(f'{operation}: 批量保存用户决策时发生未知错误。')
            if isinstance(e, StorageError):
                raise
            raise StorageOperationError(operation=f'{operation}_unknown',
                original_exception=e) from e

    def get_user_decisions(self, criteria: Optional[Dict[str, Any]]=None
        ) ->List[UserDecisionDTO]:
        self.logger.debug(f'SQLiteStorage: 尝试获取用户决策，条件: {criteria}')
        operation = 'get_user_decisions'
        try:
            with self._session_scope() as session:
                query = session.query(UserDecisionOrmModel)
                if criteria:
                    self.logger.trace(f'{operation}: 应用查询条件: {criteria}')
                    for key, value in criteria.items():
                        if hasattr(UserDecisionOrmModel, key):
                            query = query.filter(getattr(
                                UserDecisionOrmModel, key) == value)
                        else:
                            self.logger.warning(
                                f'{operation}: 未知或不支持的查询条件键: {key}')
                all_orms = query.all()
                return [self._convert_user_decision_orm_to_dto(orm) for orm in
                    all_orms]
        except SQLAlchemyError as e:
            self.logger.exception(f'{operation}: 获取用户决策时发生数据库错误。')
            raise StorageOperationError(operation=operation,
                original_exception=e) from e
        except Exception as e:
            self.logger.exception(f'{operation}: 获取用户决策时发生未知错误。')
            if isinstance(e, StorageError):
                raise
            raise StorageOperationError(operation=f'{operation}_unknown',
                original_exception=e) from e

    def begin_transaction(self) ->None:
        self.logger.info('SQLiteStorage: 开始显式事务...')
        if not self.SessionLocal:
            msg = 'SQLiteStorage.begin_transaction: SessionLocal 未初始化。'
            self.logger.error(msg)
            raise StorageConnectionError(msg)
        if self._current_session and self._current_session.is_active:
            msg = 'SQLiteStorage.begin_transaction: 已存在活动的显式事务，不支持嵌套事务。'
            self.logger.error(msg)
            raise TransactionError(operation='begin_transaction', message=
                '嵌套事务不受支持')
        self._current_session = self.SessionLocal()
        self.logger.debug('SQLiteStorage: 显式事务的会话已创建。')

    def commit_transaction(self) ->None:
        self.logger.info('SQLiteStorage: 尝试提交显式事务...')
        if not self._current_session or not self._current_session.is_active:
            msg = 'SQLiteStorage.commit_transaction: 没有活动的显式事务可供提交。'
            self.logger.error(msg)
            raise TransactionError(operation='commit_transaction', message=
                '无活动事务')
        try:
            self._current_session.commit()
            self.logger.info('SQLiteStorage: 显式事务已成功提交。')
        except SQLAlchemyError as e:
            self.logger.exception(f'SQLiteStorage: 提交显式事务失败，将尝试回滚。错误: {e}')
            try:
                self._current_session.rollback()
                self.logger.info('SQLiteStorage: 事务因提交失败已回滚。')
            except SQLAlchemyError as rb_exc:
                self.logger.exception(
                    f'SQLiteStorage: 【严重】提交失败后，回滚事务也失败。错误: {rb_exc}')
            raise TransactionError(operation='commit_transaction',
                original_exception=e) from e
        finally:
            if self._current_session:
                self._current_session.close()
                self._current_session = None
            self.logger.debug('SQLiteStorage: 显式事务的会话已关闭。')

    def rollback_transaction(self) ->None:
        self.logger.info('SQLiteStorage: 尝试回滚显式事务...')
        if not self._current_session:
            self.logger.warning(
                'SQLiteStorage.rollback_transaction: 没有显式事务会话 (_current_session is None)。'
                )
            return
        if not self._current_session.is_active:
            self.logger.warning(
                'SQLiteStorage.rollback_transaction: 显式事务会话已非活动状态 (可能已被提交或回滚)。'
                )
            try:
                self._current_session.close()
            except SQLAlchemyError as e_close:
                self.logger.exception(f'SQLiteStorage: 关闭非活动事务会话时出错: {e_close}'
                    )
            finally:
                self._current_session = None
                return
        try:
            self._current_session.rollback()
            self.logger.info('SQLiteStorage: 显式事务已成功回滚。')
        except SQLAlchemyError as e:
            self.logger.exception(f'SQLiteStorage: 回滚显式事务失败。错误: {e}')
            raise TransactionError(operation='rollback_transaction',
                original_exception=e) from e
        finally:
            if self._current_session:
                self._current_session.close()
                self._current_session = None
            self.logger.debug('SQLiteStorage: 显式事务的会话已关闭（在回滚后）。')
