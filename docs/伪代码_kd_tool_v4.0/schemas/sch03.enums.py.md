# kd_tool/schemas/enums.py
"""
该模块定义了项目中使用的核心枚举类型。
这些枚举被 Pydantic 模型（DTOs 和 Settings）以及业务逻辑广泛使用。
"""
from enum import Enum

class BlockType(str, Enum):
    """定义内容块的类型。"""
    TITLE = "Title"
    NARRATIVE_TEXT = "NarrativeText"
    LIST_ITEM = "ListItem"
    CODE_SNIPPET = "CodeSnippet"
    TABLE = "Table"
    IMAGE = "Image"
    UNCATEGORIZED = "Uncategorized"

class AnalysisType(str, Enum):
    """定义分析的类型。"""
    MD5 = "MD5"
    SIMHASH = "SimHash"
    SEMANTIC = "Semantic"

class DecisionType(str, Enum):
    """定义用户或系统对重复/相似项的决策类型。"""
    UNDECIDED = "undecided"
    KEEP = "keep"
    DELETE = "delete"
    IGNORE_PAIR = "ignore_pair"

class ProcessingStatus(str, Enum):
    """定义文件记录的处理状态。"""
    PENDING = "pending"
    DUPLICATE = "duplicate"

    PREPROCESSING_SCHEDULED = "preprocessing_scheduled"
    PREPROCESSING_RUNNING = "preprocessing_running"
    PREPROCESSING_COMPLETED = "preprocessing_completed"
    PREPROCESSING_SKIPPED = "preprocessing_skipped"
    PREPROCESSING_FAILED = "preprocessing_failed"

    BLOCK_EXTRACTION_SCHEDULED = "block_extraction_scheduled"
    BLOCK_EXTRACTION_RUNNING = "block_extraction_running"
    BLOCK_EXTRACTION_COMPLETED = "block_extraction_completed"
    BLOCK_EXTRACTION_FAILED = "block_extraction_failed"

    MD5_ANALYSIS_SCHEDULED = "md5_analysis_scheduled"
    MD5_ANALYSIS_RUNNING = "md5_analysis_running"
    MD5_ANALYSIS_COMPLETED = "md5_analysis_completed"
    MD5_ANALYSIS_SKIPPED = "md5_analysis_skipped"
    MD5_ANALYSIS_FAILED = "md5_analysis_failed"

    SIMHASH_ANALYSIS_SCHEDULED = "simhash_analysis_scheduled"
    SIMHASH_ANALYSIS_RUNNING = "simhash_analysis_running"
    SIMHASH_ANALYSIS_COMPLETED = "simhash_analysis_completed"
    SIMHASH_ANALYSIS_SKIPPED = "simhash_analysis_skipped"
    SIMHASH_ANALYSIS_FAILED = "simhash_analysis_failed"

    SEMANTIC_ANALYSIS_SCHEDULED = "semantic_analysis_scheduled"
    SEMANTIC_ANALYSIS_RUNNING = "semantic_analysis_running"
    SEMANTIC_ANALYSIS_COMPLETED = "semantic_analysis_completed"
    SEMANTIC_ANALYSIS_SKIPPED = "semantic_analysis_skipped"
    SEMANTIC_ANALYSIS_FAILED = "semantic_analysis_failed"

    ALL_ANALYSES_COMPLETED = "all_analyses_completed"
    PARTIAL_ANALYSES_COMPLETED = "partial_analyses_completed"

    DECISION_MAKING_PENDING = "decision_making_pending"
    DECISION_MAKING_IN_PROGRESS = "decision_making_in_progress"
    DECISIONS_APPLIED_PARTIALLY = "decisions_applied_partially"
    DECISIONS_APPLIED_COMPLETELY = "decisions_applied_completely"

    WORKFLOW_COMPLETED_SUCCESSFULLY = "workflow_completed_successfully"
    WORKFLOW_TERMINATED_WITH_ERRORS = "workflow_terminated_with_errors"
    WORKFLOW_CANCELLED = "workflow_cancelled"


```