# kd_tool/stages/blockmerging/settings_models.py (v4.6 - BlockMerging Settings 迁移版)
# -*- coding: utf-8 -*-

"""
=================================================
settings_models.py - BlockMerging Stage 配置模型 (v4.6)
=================================================

**模块功能**:

- **核心职责**: 定义 BlockMerging Stage (`kd_tool.stages.blockmerging`) 所需的配置设置模型。
- **v4.6 核心变更**:
    - **[架构指令]** `BlockMergerStageSettings` 从原 `schemas` 目录迁移至此。
    - **[架构指令]** 内部嵌套的特定合并设置 (如 `CodeBlockMergeSettings`, `TextBlockMergeSettings`)
                   也在此定义，以保证模块内聚性。

---
"""

# --- Python 标准库及第三方库导入 ---
from typing import List, Optional # [指令] 导入所需类型
from pydantic import BaseModel, Field, PositiveInt

# --- 项目内部模块导入 ---
# [指令] 从中央 schemas 目录导入共享枚举
from ....schemas.enums import BlockType #


# ==============================================================================
# 特定类型的合并设置 (示例)
# ==============================================================================
# [架构师说明]: 这些是 BlockMergerStageSettings 可能引用的具体合并规则配置。
#               它们之前并未在 schemas/settings_models.py 中明确列出，
#               但根据 block_merging_stage.py 的使用情况在此补充和迁移。

class CodeBlockMergeSettings(BaseModel):
    """代码块合并规则的具体配置。"""
    enabled: bool = Field(default=True, description="是否启用代码块合并。")
    # 示例配置项 (根据实际需求添加)
    max_lines_between_blocks_to_merge: int = Field(
        default=1,
        description="允许合并的连续代码块之间的最大空行数（或非代码元素数）。"
    )
    # [指令] coding 阶段：需要根据实际合并逻辑细化这里的配置项。

    class Config:
        extra = 'forbid'

class TextBlockMergeSettings(BaseModel):
    """文本块（Text, ListItem 等）合并规则的具体配置。"""
    enabled: bool = Field(default=True, description="是否启用文本块合并。")
    # 示例配置项 (从 block_merging_stage.py 的暗示和 settings_models.py 推断)
    short_text_char_threshold: PositiveInt = Field(
        default=50, # 假设值
        description="被视为空短文本块的字符数阈值，这类块更容易被合并。"
    )
    max_merged_text_block_length_char: PositiveInt = Field(
        default=2000, # 与 BlockMergerStageSettings 的 max_block_length_char 对应
        description="合并后文本块的最大允许字符长度。"
    )
    # [指令] coding 阶段：需要根据实际合并逻辑细化这里的配置项。

    class Config:
        extra = 'forbid'

# ==============================================================================
# 块合并阶段配置 (BlockMergerStageSettings)
# ==============================================================================
# [架构师说明]: BlockMergerStageSettings 是 BlockMerging Stage 的顶层配置。
#               它之前位于 schemas/settings_models.py。

class BlockMergerStageSettings(BaseModel):
    """P04 - 块合并阶段的配置。"""
    enabled: bool = Field(
        default=True,
        description="是否启用 P04 - 块合并阶段。"
    )
    # [指令] types_to_attempt_merge 用于全局控制哪些类型的块可以尝试合并
    types_to_attempt_merge: List[BlockType] = Field(
        default=[BlockType.CODE_SNIPPET, BlockType.NARRATIVE_TEXT, BlockType.LIST_ITEM], # 示例值
        description="一个 BlockType 列表，定义了哪些类型的块会尝试应用其特定的合并规则。"
    )
    # [指令] preserve_blocks_with_min_char_length 用于防止过长的“有意义”文本块被错误合并
    preserve_blocks_with_min_char_length: Optional[PositiveInt] = Field(
        default=300, # 假设值
        description="如果一个文本块的字符长度大于等于此值，则即使它符合其他合并条件，也倾向于保留它不被合并。设置为 None 则禁用此保留逻辑。"
    )
    # [指令] 以下的 min/max block_length_char 是合并后的启发式目标，
    #       但具体的合并逻辑在 CodeBlockMergeSettings 和 TextBlockMergeSettings 中。
    #       这些顶层参数可能用于最终校验或作为默认值传递。
    min_block_length_char: PositiveInt = Field(
        default=100,
        description="合并后块的期望最小字符长度（启发式规则）。"
    )
    max_block_length_char: PositiveInt = Field(
        default=2000,
        description="合并后块的期望最大字符长度（启发式规则）。"
    )

    # [指令] 嵌套的配置模型，用于特定类型的合并规则
    code_block_settings: CodeBlockMergeSettings = Field(default_factory=CodeBlockMergeSettings)
    text_block_settings: TextBlockMergeSettings = Field(default_factory=TextBlockMergeSettings)

    # [指令] 未来可以添加更多合并策略配置，如按标题合并、按空行合并等。
    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True # 允许枚举等类型