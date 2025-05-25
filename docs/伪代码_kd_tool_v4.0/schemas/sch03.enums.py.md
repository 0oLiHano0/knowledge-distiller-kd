```python

# kd_tool/schemas/enums.py (v4.7.1 - Cleanup 更新 & 注释增强)
# -*- coding: utf-8 -*-

"""
=================================================
sch03.enums.py.md - KD_Tool 核心枚举类型
=================================================

**模块功能**:

- **核心职责**: 定义项目中使用的核心枚举类型 (Enums)。
- **作用**: 提高代码的可读性、可维护性，并为 Pydantic 模型提供类型安全的选项。
- **规范**: 
    - 尽可能使用 `(str, Enum)` 继承，以便与 Pydantic 和序列化良好集成。
    - 枚举值应清晰、具有描述性。

**版本历史**:
- v4.0: 初始设计。
- v4.7: 为 `ProcessingStatus` 添加 Cleanup 阶段相关状态。
- v4.7.1: 【架构师决策】增强注释，明确各枚举的用途。

---
"""

from enum import Enum

class BlockType(str, Enum):
    """
    定义内容块的类型。
    **用途**: 由 `DocumentProcessingStage` 生成，`BlockMergingStage` 可能使用。
    """
    TITLE = "Title"
    NARRATIVE_TEXT = "NarrativeText"
    LIST_ITEM = "ListItem"
    CODE_SNIPPET = "CodeSnippet"
    TABLE = "Table"
    IMAGE = "Image"
    UNCATEGORIZED = "Uncategorized"

class AnalysisType(str, Enum):
    """
    定义分析的类型。
    **用途**: 
    - 标记 `AnalysisResultDTO` 的来源。
    - 在 `PipelineContextDTO` 中组织分析结果。
    - 在 `DecisionStage` 中用于区分不同分析结果以应用规则。
    """
    MD5 = "MD5"
    SIMHASH = "SimHash"
    SEMANTIC = "Semantic"

class DecisionType(str, Enum):
    """
    定义用户或系统对重复/相似项的决策类型。
    **用途**: 
    - 作为 `DecisionStage` 的输出 (`UserDecisionDTO.decision`)。
    - 作为 `CleanupStage` 的输入，指导其执行操作。
    - 作为 `DecisionStageSettings` 中规则定义的一部分。
    """
    UNDECIDED = "undecided" # 尚未决定或规则无法确定
    KEEP = "keep"           # 明确保留 (通常是基准或选择保留的项)
    DELETE = "delete"         # 明确删除 (标记为冗余的项)
    IGNORE_PAIR = "ignore_pair" # 明确忽略这对相似项，不再处理

class ProcessingStatus(str, Enum):
    """
    定义文件记录 (`FileRecordDTO`) 的处理状态。
    **用途**: 追踪文件在整个流水线中的生命周期。
    **规范**: 各个 Stage 负责在处理开始和结束时更新此状态。
    """
    # --- 初始 & 预过滤 ---
    PENDING = "pending" # 初始状态，等待处理
    DUPLICATE = "duplicate" # 被 Prefilter 识别为（可能）的重复文件（基于 Czkawka）

    # --- 文档预处理 (可能包含提取、清洗等) ---
    PREPROCESSING_SCHEDULED = "preprocessing_scheduled"
    PREPROCESSING_RUNNING = "preprocessing_running"
    PREPROCESSING_COMPLETED = "preprocessing_completed"
    PREPROCESSING_SKIPPED = "preprocessing_skipped"
    PREPROCESSING_FAILED = "preprocessing_failed"

    # --- 块提取 ---
    BLOCK_EXTRACTION_SCHEDULED = "block_extraction_scheduled"
    BLOCK_EXTRACTION_RUNNING = "block_extraction_running"
    BLOCK_EXTRACTION_COMPLETED = "block_extraction_completed"
    BLOCK_EXTRACTION_FAILED = "block_extraction_failed"

    # --- MD5 分析 ---
    MD5_ANALYSIS_SCHEDULED = "md5_analysis_scheduled"
    MD5_ANALYSIS_RUNNING = "md5_analysis_running"
    MD5_ANALYSIS_COMPLETED = "md5_analysis_completed"
    MD5_ANALYSIS_SKIPPED = "md5_analysis_skipped"
    MD5_ANALYSIS_FAILED = "md5_analysis_failed"

    # --- SimHash 分析 ---
    SIMHASH_ANALYSIS_SCHEDULED = "simhash_analysis_scheduled"
    SIMHASH_ANALYSIS_RUNNING = "simhash_analysis_running"
    SIMHASH_ANALYSIS_COMPLETED = "simhash_analysis_completed"
    SIMHASH_ANALYSIS_SKIPPED = "simhash_analysis_skipped"
    SIMHASH_ANALYSIS_FAILED = "simhash_analysis_failed"

    # --- 语义分析 ---
    SEMANTIC_ANALYSIS_SCHEDULED = "semantic_analysis_scheduled"
    SEMANTIC_ANALYSIS_RUNNING = "semantic_analysis_running"
    SEMANTIC_ANALYSIS_COMPLETED = "semantic_analysis_completed"
    SEMANTIC_ANALYSIS_SKIPPED = "semantic_analysis_skipped"
    SEMANTIC_ANALYSIS_FAILED = "semantic_analysis_failed"

    # --- 分析汇总 ---
    ALL_ANALYSES_COMPLETED = "all_analyses_completed"
    PARTIAL_ANALYSES_COMPLETED = "partial_analyses_completed"

    # --- 决策 ---
    DECISION_MAKING_PENDING = "decision_making_pending"
    DECISION_MAKING_IN_PROGRESS = "decision_making_in_progress"
    DECISIONS_APPLIED_PARTIALLY = "decisions_applied_partially" # 如果未来支持手动决策
    DECISIONS_APPLIED_COMPLETELY = "decisions_applied_completely"

    # --- 清理 (v4.7 新增) ---
    CLEANUP_PENDING = "cleanup_pending"
    MARKED_FOR_DELETION = "marked_for_deletion" # 仅标记状态
    MOVED_TO_TRASH = "moved_to_trash"         # 已移动到回收站
    PERMANENTLY_DELETED = "permanently_deleted" # 已永久删除 (文件 + 可能的记录)
    CLEANUP_SKIPPED = "cleanup_skipped"       # 无需清理或跳过
    CLEANUP_FAILED = "cleanup_failed"         # 清理操作失败

    # --- 最终工作流状态 ---
    WORKFLOW_COMPLETED_SUCCESSFULLY = "workflow_completed_successfully" # 文件已处理且无需清理，或已成功清理
    WORKFLOW_TERMINATED_WITH_ERRORS = "workflow_terminated_with_errors"
    WORKFLOW_CANCELLED = "workflow_cancelled"


```