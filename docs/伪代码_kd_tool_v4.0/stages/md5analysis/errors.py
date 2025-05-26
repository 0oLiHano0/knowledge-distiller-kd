# ------------------------------------------------------------------------------
# 文件名: kd_tool/stages/md5analysis/errors.py
# 模块: MD5 分析阶段 - 自定义异常
# 描述:
#   定义 MD5AnalysisStage 可能抛出的特定异常。
#   所有异常必须继承自 KDToolError。
# ------------------------------------------------------------------------------

from typing import Optional, Any

# 导入核心基础异常类 (必须继承)
from kd_tool.core.errors import KDToolError

# ==============================================================================
# MD5 分析阶段基础异常 (MD5AnalysisError)
# ==============================================================================

class MD5AnalysisError(KDToolError):
    """
    MD5 分析阶段的基类错误。

    架构说明:
        - 所有本阶段的错误都应继承此类。
        - 构造时自动设置 'module' = 'MD5AnalysisStage'。
    """
    def __init__(
        self,
        message: str,
        original_exception: Optional[Exception] = None,
        **kwargs: Any
    ):
        super().__init__(
            message,
            original_exception=original_exception,
            module="MD5AnalysisStage", # <-- 强制模块上下文
            **kwargs
        )

# ==============================================================================
# MD5 分析阶段特定异常 (简化版)
# ==============================================================================

class MD5InputError(MD5AnalysisError):
    """
    当 MD5 分析阶段的输入数据 (来自 PipelineContextDTO 的 ContentBlockDTO)
    不满足计算 MD5 的要求时抛出。

    架构说明:
        - **coding 阶段要求**: 在尝试计算 MD5 之前，必须检查 `ContentBlockDTO`
          的 `analysis_text` (或 `text_content`) 字段。如果该字段为 `None` 或
          空字符串 (根据业务逻辑决定空字符串是否算错误)，则必须抛出此异常。
        - **必须包含** `block_id` 和具体 `reason` 作为上下文信息。
    """
    def __init__(
        self,
        block_id: str,
        reason: str,
        original_exception: Optional[Exception] = None,
        **kwargs: Any
    ):
        message = f"MD5 分析输入错误 (Block ID: {block_id}): {reason}"
        super().__init__(
            message,
            original_exception=original_exception,
            block_id=block_id, # <-- 必须的上下文
            reason=reason,     # <-- 必须的上下文
            **kwargs
        )

class MD5CalculationError(MD5AnalysisError):
    """
    当为 ContentBlockDTO 计算 MD5 哈希值的过程中发生技术性错误时抛出。

    架构说明:
        - **coding 阶段要求**: 在调用 `hashlib.md5(...)` 时，如果发生
          例如 `UnicodeEncodeError` (虽然使用 'utf-8' 可能性小但仍需考虑)
          或任何其他来自 `hashlib` 库的预期外错误，必须捕获它们并包装为此异常后抛出。
        - **必须包含** `block_id` 和原始异常作为上下文信息。
    """
    def __init__(
        self,
        block_id: str,
        original_exception: Exception, # <-- 计算错误通常由原始异常触发
        **kwargs: Any
    ):
        message = f"为 Block ID '{block_id}' 计算 MD5 时发生内部错误。"
        super().__init__(
            message,
            original_exception=original_exception, # <-- 必须包含原始异常
            block_id=block_id,                     # <-- 必须的上下文
            **kwargs
        )

# 架构说明:
#   我们保持错误定义的简化，仅关注 MD5 分析阶段 *最独特* 的错误源。
#   与存储交互的错误应由 Storage 层抛出 (或 Stage 捕获后重抛为通用 KDToolError)。
#   与 PipelineContextDTO 结构相关的错误应由 Orchestrator 或 Context 自身处理。