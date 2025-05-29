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
from typing import List, Optional
from pydantic import BaseModel, Field, PositiveInt
from kd_tool.schemas.enums import BlockType


class CodeBlockMergeSettings(BaseModel):
    """代码块合并规则的具体配置。"""
    enabled: bool = Field(default=True, description='是否启用代码块合并。')
    max_lines_between_blocks_to_merge: int = Field(default=1, description=
        '允许合并的连续代码块之间的最大空行数（或非代码元素数）。')


    class Config:
        extra = 'forbid'


class TextBlockMergeSettings(BaseModel):
    """文本块（Text, ListItem 等）合并规则的具体配置。"""
    enabled: bool = Field(default=True, description='是否启用文本块合并。')
    short_text_char_threshold: PositiveInt = Field(default=50, description=
        '被视为空短文本块的字符数阈值，这类块更容易被合并。')
    max_merged_text_block_length_char: PositiveInt = Field(default=2000,
        description='合并后文本块的最大允许字符长度。')


    class Config:
        extra = 'forbid'


class BlockMergerStageSettings(BaseModel):
    """P04 - 块合并阶段的配置。"""
    enabled: bool = Field(default=True, description='是否启用 P04 - 块合并阶段。')
    types_to_attempt_merge: List[BlockType] = Field(default=[BlockType.
        CODE_SNIPPET, BlockType.NARRATIVE_TEXT, BlockType.LIST_ITEM],
        description='一个 BlockType 列表，定义了哪些类型的块会尝试应用其特定的合并规则。')
    preserve_blocks_with_min_char_length: Optional[PositiveInt] = Field(default
        =300, description=
        '如果一个文本块的字符长度大于等于此值，则即使它符合其他合并条件，也倾向于保留它不被合并。设置为 None 则禁用此保留逻辑。')
    min_block_length_char: PositiveInt = Field(default=100, description=
        '合并后块的期望最小字符长度（启发式规则）。')
    max_block_length_char: PositiveInt = Field(default=2000, description=
        '合并后块的期望最大字符长度（启发式规则）。')
    code_block_settings: CodeBlockMergeSettings = Field(default_factory=
        CodeBlockMergeSettings)
    text_block_settings: TextBlockMergeSettings = Field(default_factory=
        TextBlockMergeSettings)


    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True
