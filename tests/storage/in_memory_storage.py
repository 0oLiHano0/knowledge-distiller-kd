# tests/logging/in_memory_storage.py
"""
WHY  : 提供 StorageInterface 的内存实现，用于测试。
WHAT : 模拟数据库行为，但不实际写入磁盘。
HOW  : 实现 StorageInterface 的所有方法，数据存储在内存列表中。
"""
from typing import List, Optional

# 注意: 这里我们从正确的路径导入 StorageInterface
from kd_tool.storage.storage_interface import StorageInterface
from kd_tool.core.core_dtos import ContentBlockDTO
from kd_tool.logging.protocols import LoggerProtocol # 导入 LoggerProtocol 以便类型提示
from kd_tool.storage.errors import RecordNotFoundError # 可能需要导入错误类型

class InMemoryStorage(StorageInterface):
    """
    StorageInterface 的内存测试实现。
    """

    def __init__(self, logger: Optional[LoggerProtocol] = None):
        """
        初始化内存存储。
        
        参数:
            logger (Optional[LoggerProtocol]): 可选的日志记录器。
        """
        self._logger = logger
        self._blocks: List[ContentBlockDTO] = []
        self._initialized = False
        self._in_transaction = False
        if self._logger:
            self._logger.info("InMemoryStorage: Instance created.")

    def initialize(self) -> None:
        """
        模拟存储初始化。
        WHY : 满足接口要求；WHAT 清空列表，标记已初始化；HOW 简单赋值。
        """
        if self._logger:
            self._logger.debug("InMemoryStorage: Initializing...")
        self._blocks = []
        self._initialized = True
        if self._logger:
            self._logger.info("InMemoryStorage: Initialized successfully.")

    def begin_transaction(self) -> None:
        """
        模拟开始事务。
        WHY : 满足接口要求；WHAT 标记进入事务状态；HOW 简单赋值。
        """
        if self._logger:
            self._logger.debug("InMemoryStorage: Beginning transaction...")
        self._in_transaction = True

    def commit_transaction(self) -> None:
        """
        模拟提交事务。
        WHY : 满足接口要求；WHAT 标记离开事务状态；HOW 简单赋值。
        """
        if self._logger:
            self._logger.debug("InMemoryStorage: Committing transaction...")
        if not self._in_transaction:
            # 在真实的模拟中可能需要更复杂的处理或抛出特定异常
            if self._logger:
                self._logger.warning("InMemoryStorage: Commit called but not in transaction.")
        self._in_transaction = False

    def rollback_transaction(self) -> None:
        """
        模拟回滚事务。
        WHY : 满足接口要求；WHAT 标记离开事务状态；HOW 简单赋值 (可扩展)。
        """
        if self._logger:
            self._logger.debug("InMemoryStorage: Rolling back transaction...")
        if not self._in_transaction:
             if self._logger:
                self._logger.warning("InMemoryStorage: Rollback called but not in transaction.")
        # 在内存模拟中，回滚可能需要清除自 begin_transaction 以来的更改
        # 这里为了简单起见，仅退出事务状态
        self._in_transaction = False

    def save_content_blocks(self, blocks: List[ContentBlockDTO]) -> None:
        """
        将内容块保存到内存列表中。
        WHY : 实现核心保存逻辑；WHAT 添加到 self._blocks；HOW 使用 extend。
        """
        if self._logger:
            self._logger.debug(f"InMemoryStorage: Saving {len(blocks)} blocks...")
        # 注意: 原始测试用例 `storage.save_content_blocks(["hello"])` 是有问题的，
        # 它应该传递 `ContentBlockDTO` 列表。
        # 此实现假设会收到正确的 DTO 列表。
        self._blocks.extend(blocks)
        if self._logger:
            self._logger.debug(f"InMemoryStorage: {len(blocks)} blocks saved.")

    def get_content_block(self, md5: str) -> Optional[ContentBlockDTO]:
        """
        模拟根据 MD5 获取内容块。
        WHY : 实现获取逻辑；WHAT 遍历查找；HOW 返回找到的或 None。
        """
        if self._logger:
            self._logger.debug(f"InMemoryStorage: Searching for block with md5: {md5}...")
        # 假设 ContentBlockDTO 有一个 'md5' 或 'text_hash_md5' 属性。
        # 根据 storage_interface.py 的签名，我们查找 'md5'。
        # 根据 dtos.py，它可能是 'text_hash_md5'。我们需要做一个假设或检查。
        # 这里假设它有一个 'md5' 属性或者 'text_hash_md5'。
        for block in self._blocks:
            # 尝试匹配 text_hash_md5，如果不存在，则匹配 block_id (作为后备，尽管不正确)
            if getattr(block, 'text_hash_md5', None) == md5:
                if self._logger:
                    self._logger.debug(f"InMemoryStorage: Found block {block.block_id}.")
                return block
        if self._logger:
            self._logger.warning(f"InMemoryStorage: Block with md5 {md5} not found.")
        # 根据接口定义，找不到应该返回 None，而不是抛出 RecordNotFoundError
        return None

    def close(self) -> None:
        """
        模拟关闭存储连接。
        WHY : 满足接口要求；WHAT 执行清理；HOW 简单 pass。
        """
        if self._logger:
            self._logger.debug("InMemoryStorage: Closing.")
        pass # 内存存储通常不需要显式关闭

    # 保留原始测试中使用的 'count' 方法，尽管它不是接口的一部分
    # 这表明测试可能需要调整，或者接口需要扩展（但不推荐）
    def _count(self) -> int:
        """
        返回当前存储的块数量 (非接口方法，仅为测试保留)。
        """
        return len(self._blocks)