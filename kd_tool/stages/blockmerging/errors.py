from typing import Optional, Any, List
from kd_tool.core.errors import KDToolError


class BlockMergingError(KDToolError):
    """
    P04 - 块合并阶段 (BlockMergingStage) 中发生的错误的基类。

    架构说明:
        - 这是所有 P04 阶段特定错误的父类。
        - 在构造时，固定 'module' 上下文信息为 'BlockMergingStageP04'。
    """

    def __init__(self, message: str, original_exception: Optional[Exception
        ]=None, **kwargs: Any):
        super().__init__(message, original_exception=original_exception,
            module='BlockMergingStageP04', **kwargs)


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

    def __init__(self, rule_description: str, conflicting_block_ids:
        Optional[List[str]]=None, original_exception: Optional[Exception]=None
        ):
        message = f'块合并规则冲突或无法应用: {rule_description}'
        super().__init__(message, original_exception=original_exception,
            rule_description=rule_description, conflicting_block_ids=
            conflicting_block_ids)


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

    def __init__(self, message: str, block_ids_involved: Optional[List[str]
        ]=None, original_exception: Optional[Exception]=None):
        super().__init__(message, original_exception=original_exception,
            block_ids_involved=block_ids_involved)


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

    def __init__(self, reason: str, original_exception: Optional[Exception]
        =None, processing_block_ids: Optional[List[str]]=None):
        message = f'块合并操作失败: {reason}'
        super().__init__(message, original_exception=original_exception,
            reason=reason, processing_block_ids=processing_block_ids)


class BlockMergingStageError(KDToolError):
    """WHY: 块合并阶段通用异常；WHAT: 统一捕获；HOW: 继承 KDToolError。"""
    pass
