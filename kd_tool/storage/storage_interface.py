# kd_tool/storage/storage_interface.py
"""
WHY  : 定义存储抽象接口，隔离持久化细节。
WHAT : 规定所有存储实现必须遵守的最小行为契约。
HOW  : 使用 abc.ABC + 类型提示，方法无状态、纯签名。
"""
from __future__ import annotations

import abc
from typing import List, Optional, Protocol

from kd_tool.core.core_dtos import ContentBlockDTO, PipelineContextDTO  # 仅导入 DTO，无 ORM 依赖
from kd_tool.core.errors import KDToolError


class TransactionError(KDToolError):
    """事务生命周期错误。"""
    pass


class StorageInterface(abc.ABC):
    """WHY: 统一访问层；WHAT: 定义核心方法；HOW: 依赖注入后使用。"""

    @abc.abstractmethod
    def initialize(self) -> None:
        """WHY 初始化；WHAT 建立连接、建表等；HOW 调用后可用。"""
        ...

    # ---------- 事务 ----------
    @abc.abstractmethod
    def begin_transaction(self) -> None: ...
    @abc.abstractmethod
    def commit_transaction(self) -> None: ...
    @abc.abstractmethod
    def rollback_transaction(self) -> None: ...

    # ---------- CRUD ----------
    @abc.abstractmethod
    def save_content_blocks(self, blocks: List[ContentBlockDTO]) -> None: ...
    @abc.abstractmethod
    def get_content_block(self, md5: str) -> Optional[ContentBlockDTO]: ...
    @abc.abstractmethod
    def close(self) -> None: ...

    def save_pipeline_context(self, context: PipelineContextDTO) -> None:
        """批量持久化整个流水线上下文，内部自动处理事务。"""

