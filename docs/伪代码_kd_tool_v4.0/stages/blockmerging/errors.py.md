```python
# ------------------------------------------------------------------------------
# 文件名: kd_tool/stages/blockmerging/errors.py.md
# 模块: P04 - 块合并阶段 (BlockMergingStage) - 自定义异常
# 描述:
#   此模块定义了 P04 BlockMergingStage 及其相关操作可能抛出的特定异常。
#   所有在此定义的异常都必须继承自核心异常类 KDToolError。
# 架构约束:
#   - 所有异常必须继承自 KDToolError (或其子类)。
#   - 异常应携带足够上下文信息 (通过 **kwargs 传递给 KDToolError)。
#   - 异常命名应清晰反映错误类型。
#   - 'module' 上下文信息在此基类中固定为 'BlockMergingStageP04'。
# ------------------------------------------------------------------------------

from typing import Optional, Any, List # <-- 确保导入 List
from kd_tool.core.errors import KDToolError # <-- 从核心错误模块导入

# ==============================================================================
# P04 - 块合并阶段基础异常 (BlockMergingError)
# ==============================================================================

class BlockMergingError(KDToolError):
    """
    P04 - 块合并阶段 (BlockMergingStage) 中发生的错误的基类。

    架构说明:
        - 这是所有 P04 阶段特定错误的父类。
        - 在构造时，固定 'module' 上下文信息为 'BlockMergingStageP04'。
    """
    def __init__(
        self,
        message: str,
        original_exception: Optional[Exception] = None,
        # 架构说明: 可以考虑添加如 file_id 或正在处理的 block_ids 列表作为通用上下文
        # file_id: Optional[str] = None, 
        # processing_block_ids: Optional[List[str]] = None,
        **kwargs: Any
    ):
        # if file_id: kwargs['file_id'] = file_id
        # if processing_block_ids: kwargs['processing_block_ids'] = processing_block_ids
        
        super().__init__(
            message,
            original_exception=original_exception,
            module="BlockMergingStageP04",  # <-- 强制模块上下文
            **kwargs
        )

# ==============================================================================
# P04 - 块合并阶段特定异常
# ==============================================================================

class MergeRuleConflictError(BlockMergingError):
    """
    当块合并的规则之间存在逻辑冲突，或者某条规则无法应用于当前块序列时抛出。
    例如：两条规则对同一块给出了不同的合并指令，且无法自动解决。

    架构说明:
        - **coding 阶段要求**: 如果合并逻辑包含复杂的规则引擎或条件判断，
          当检测到无法解决的规则冲突时，应抛出此异常。
        - **必须包含** 描述冲突的 `rule_description`。
        - **可选包含** 导致冲突的 `conflicting_block_ids`。
    """
    def __init__(
        self,
        rule_description: str, # 描述冲突的规则或情况
        conflicting_block_ids: Optional[List[str]] = None,
        original_exception: Optional[Exception] = None
    ):
        message = f"块合并规则冲突或无法应用: {rule_description}"
        super().__init__(
            message,
            original_exception=original_exception,
            rule_description=rule_description,
            conflicting_block_ids=conflicting_block_ids
        )

class InvalidBlockSequenceError(BlockMergingError):
    """
    当提供给合并算法的块序列不符合预期时抛出。
    例如：期望合并的是连续的文本块，但序列中混入了非文本块，且规则未处理此情况。

    架构说明:
        - **coding 阶段要求**: 在具体的合并函数（如 `_merge_text_blocks`）开始处理前，
          如果发现输入的块列表在类型、顺序或其他结构上不满足合并前提，应抛出此异常。
        - **必须包含** 描述序列无效原因的 `message`。
        - **可选包含** 相关的 `block_ids_involved`。
    """
    def __init__(
        self,
        message: str, # 例如 "期望连续的TEXT块，但遇到了CODE块"
        block_ids_involved: Optional[List[str]] = None,
        original_exception: Optional[Exception] = None
    ):
        # message 参数已由调用者提供具体的错误描述
        super().__init__(
            message,
            original_exception=original_exception,
            block_ids_involved=block_ids_involved
        )

class MergingFailedError(BlockMergingError):
    """
    当某个具体的合并操作（例如，合并两个文本块的文本内容）由于意外原因失败时抛出。
    这通常指示合并算法内部的非预期错误。

    架构说明:
        - **coding 阶段要求**: 在执行实际的合并动作（如字符串拼接、元数据聚合）时，
          如果发生例如 `TypeError`, `ValueError` 等标准库异常，应捕获并包装为此错误。
        - **必须包含** 具体的失败 `reason` 和原始异常。
        - **可选包含** 正在处理的 `processing_block_ids`。
    """
    def __init__(
        self,
        reason: str, # 描述合并失败的具体环节或原因
        original_exception: Optional[Exception] = None,
        processing_block_ids: Optional[List[str]] = None
    ):
        message = f"块合并操作失败: {reason}"
        super().__init__(
            message,
            original_exception=original_exception,
            reason=reason,
            processing_block_ids=processing_block_ids
        )

# 架构说明:
#   - P04 块合并阶段的错误类型相对聚焦于合并逻辑本身。
#   - 如果合并过程中需要读取或写入数据到存储层（P04不直接操作存储，而是更新Context），
#     那么与存储相关的错误应该由 Storage 层抛出，或者被 Orchestrator 捕获。
#   - 输入的 ContentBlockDTO 本身的有效性问题（如缺少关键字段）理论上应由P03或其之前的阶段保证，
#     或者在P04开始时进行一次前置校验（如果校验失败，可能抛出 InvalidBlockSequenceError）。


```