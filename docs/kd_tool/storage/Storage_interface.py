"""
=================================================
s01.Storage_interface.py.md - 存储接口定义 (v4.6)
=================================================

**模块功能**:

- 定义了存储层的抽象接口（契约）。
- 所有具体的存储实现（例如 SQLiteStorage）都必须遵守此接口。
- **v4.6 核心变更**:
    - **[架构指令]** `register_file` 方法的 `task_id` 参数已移除，因为 `FileRecordDTO` 已不包含 `task_id`。
    - **[架构指令]** `save_content_blocks` 方法确认不接受 `file_id` 参数；`ContentBlockDTO` 自身包含 `file_id`。
    - **[架构指令]** `save_analysis_results` 方法的 `task_id` 参数已移除，因为 `AnalysisResultDTO` 已不包含 `task_id`。
    - **[架构指令]** `save_user_decision` 方法已重构为 `save_user_decisions`，接受 `List[UserDecisionDTO]`。
    - **[架构指令]** 在文档字符串中补充事务管理相关的说明。

---
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from loguru import Logger
from kd_tool.storage.settings_models import StorageSettingsDTO
from kd_tool.schemas.dtos import FileRecordDTO, ContentBlockDTO, AnalysisResultDTO, UserDecisionDTO


class StorageInterface(ABC):
    """
    定义了知识蒸馏工具中所有存储操作的契约的抽象基类。
    """

    @abstractmethod
    def __init__(self, settings: StorageSettingsDTO, logger: Logger):
        """构造函数，用于依赖注入。"""
        self.settings = settings
        self.logger = logger

    @abstractmethod
    def initialize(self) ->None:
        """设置并准备存储后端。"""
        raise NotImplementedError

    @abstractmethod
    def finalize(self) ->None:
        """执行清理任务，释放资源。"""
        raise NotImplementedError

    @abstractmethod
    def register_file(self, file_dto: FileRecordDTO) ->FileRecordDTO:
        """
        【v4.6 修改】注册单个文件记录，或获取/更新已注册记录。
        参数直接为 FileRecordDTO，不再需要单独的 file_path 和 metadata。
        FileRecordDTO 内部已不含 task_id。
        """
        raise NotImplementedError

    @abstractmethod
    def register_files(self, files_data: List[FileRecordDTO]) ->List[
        FileRecordDTO]:
        """
        批量注册或更新文件记录。
        FileRecordDTO 内部已不含 task_id。
        """
        raise NotImplementedError

    @abstractmethod
    def get_file_record(self, file_id: str) ->Optional[FileRecordDTO]:
        """根据文件的唯一 ID 检索文件记录。"""
        raise NotImplementedError

    @abstractmethod
    def list_file_records(self, criteria: Optional[Dict[str, Any]]=None
        ) ->List[FileRecordDTO]:
        """列出所有文件记录，可选过滤。"""
        raise NotImplementedError

    @abstractmethod
    def update_file_record(self, file_id: str, updates: Dict[str, Any]
        ) ->Optional[FileRecordDTO]:
        """
        更新指定 file_id 的文件记录。
        **注意**: `updates` 字典中的键应对应 `FileRecordDTO` 的字段名。
        """
        raise NotImplementedError

    @abstractmethod
    def delete_file_records(self, file_ids: List[str]) ->int:
        """
        【v4.6 新增】批量删除指定 file_id 的文件记录。
        返回成功删除的记录数量。
        **注意**: 此方法仅删除文件记录本身，不保证删除相关联的内容块等。
                  级联删除行为应由数据库层面（如果配置）或更上层的业务逻辑处理。
        """
        raise NotImplementedError

    @abstractmethod
    def save_content_blocks(self, blocks_data: List[ContentBlockDTO]) ->List[
        ContentBlockDTO]:
        """
        【v4.6 修改】保存或更新一批内容块。
        移除了 file_id 参数，因为 ContentBlockDTO 已包含 file_id。
        ContentBlockDTO 内部已不含 task_id。
        返回保存/更新后的 ContentBlockDTO 列表（可能包含数据库生成的ID或时间戳）。
        """
        raise NotImplementedError

    @abstractmethod
    def get_content_block(self, block_id: str) ->Optional[ContentBlockDTO]:
        """通过 ID 检索内容块。"""
        raise NotImplementedError

    @abstractmethod
    def get_content_blocks_for_file(self, file_id: str, criteria: Optional[
        Dict[str, Any]]=None) ->List[ContentBlockDTO]:
        """检索指定文件的所有内容块。"""
        raise NotImplementedError

    @abstractmethod
    def get_all_content_blocks(self, criteria: Optional[Dict[str, Any]]=None
        ) ->List[ContentBlockDTO]:
        """检索所有内容块（谨慎使用）。"""
        raise NotImplementedError

    @abstractmethod
    def save_analysis_results(self, results_data: List[AnalysisResultDTO]
        ) ->List[AnalysisResultDTO]:
        """
        【v4.6 修改】保存或更新一批分析结果。
        AnalysisResultDTO 内部已不含 task_id。
        返回保存/更新后的 AnalysisResultDTO 列表。
        """
        raise NotImplementedError

    @abstractmethod
    def get_analysis_results(self, criteria: Optional[Dict[str, Any]]=None
        ) ->List[AnalysisResultDTO]:
        """根据条件检索分析结果。"""
        raise NotImplementedError

    @abstractmethod
    def save_user_decisions(self, decisions_data: List[UserDecisionDTO]
        ) ->List[UserDecisionDTO]:
        """
        【v4.6 修改】保存或更新一批用户决策。
        原 `save_user_decision` (单数) 已修改为此批量方法。
        UserDecisionDTO 内部已不含 task_id。
        返回保存/更新后的 UserDecisionDTO 列表。
        """
        raise NotImplementedError

    @abstractmethod
    def get_user_decisions(self, criteria: Optional[Dict[str, Any]]=None
        ) ->List[UserDecisionDTO]:
        """根据条件检索用户决策。"""
        raise NotImplementedError

    @abstractmethod
    def begin_transaction(self) ->None:
        """显式开始一个新的事务。"""
        raise NotImplementedError

    @abstractmethod
    def commit_transaction(self) ->None:
        """提交当前活动的事务。"""
        raise NotImplementedError

    @abstractmethod
    def rollback_transaction(self) ->None:
        """回滚当前活动的事务。"""
        raise NotImplementedError
