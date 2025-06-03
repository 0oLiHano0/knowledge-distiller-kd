"""
=================================================
settings_models.py - DocumentProcessing Stage 配置模型 (v4.6)
=================================================

**模块功能**:

- **核心职责**: 定义 DocumentProcessing Stage (`kd_tool.stages.docprocessing`) 所需的配置设置模型。
- **v4.6 核心变更**:
    - **[架构指令]** `DocumentProcessingStageSettings` 从原 `schemas` 目录迁移至此。

---
"""
from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict


class DocumentProcessingStageSettings(BaseModel):
    """P03 - 文档处理阶段 (原始提取) 的配置模型。"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    enabled: bool = Field(default=True, description='是否启用 P03 - 文档处理阶段。')
    parser_type: Literal['unstructured', 'pdfplumber'] = Field(default='unstructured', description='解析器类型')
    parsing_strategy: Literal['auto', 'fast', 'hi_res'] = Field(default='auto', description='底层解析库 (如 `unstructured`) 使用的解析策略。')
    supported_extensions: List[str] = Field(default=['.md', '.txt', '.docx',
        '.pdf'], description='此阶段尝试处理的文件扩展名列表。')


