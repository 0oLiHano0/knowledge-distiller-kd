```python
# 存放 `StorageSettingsDTO`, `OrchestratorSettings`, `LoggingSettings`, 各个 StageSettings 等

# kd_tool/schemas/settings_models.py
# -*- coding: utf-8 -*-

"""
=================================================
sch02.settings_models.py.md - KD_Tool 配置设置模型 (v4.1)
=================================================

**模块功能**:

- 定义项目中核心组件和服务所需的配置设置模型 (Settings Models)。
- 使用 Pydantic 定义，用于结构化和验证应用程序配置。
- **v4.1 更新**: 引入了 PrefilterStageSettings，替换旧的 Czkawka 配置，并明确了 BlockMerger 配置。

---
"""

from pydantic import BaseModel, Field, FilePath, PositiveInt, model_validator
from typing import Optional, Literal, Any, List
from pathlib import Path

# ==============================================================================
# 基础服务配置
# ==============================================================================

class StorageSettingsDTO(BaseModel):
    """存储服务的配置设置 DTO。"""
    backend_type: str = Field(
        default="sqlite",
        description="存储后端类型。例如 'sqlite', 'memory_debug'。"
    )
    connection_string: Optional[str] = Field(
        default=None,
        description="数据库连接字符串。对于 'sqlite'，例如：'sqlite:///./data/kd_tool.db'。"
    )
    base_directory: Optional[Path] = Field(
        default=None,
        description="文件系统存储的基础目录路径 (如果后端使用文件系统)。"
    )

    @model_validator(mode='after')
    @classmethod
    def check_consistency(cls, data: Any) -> Any:
        if isinstance(data, cls):
            if data.backend_type == "sqlite" and not data.connection_string:
                raise ValueError("对于 'sqlite' 后端类型, 'connection_string' 必须提供。")
        return data

    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True


class OrchestratorSettings(BaseModel):
    """Orchestrator 模块的配置设置。"""
    on_pipeline_error_policy: Literal['HALT_ON_FIRST_ERROR', 'CONTINUE_IGNORING_ERROR'] = Field(
        default='HALT_ON_FIRST_ERROR',
        description="流水线错误处理策略。"
    )
    default_task_id_prefix: str = Field(
        default='kd_task_',
        description="为生成的 task_id 添加的可选前缀。"
    )
    # 【新增】明确定义默认流水线顺序，Orchestrator 可以使用它
    default_stage_order: List[str] = Field(
        default=[
            "prefilter", 
            "document_processing", 
            "block_merging", 
            "md5_analysis",
            "semantic_analysis", # 假设顺序
            "decision" # 假设顺序
        ],
        description="默认情况下流水线中各个阶段的执行顺序和名称。"
    )

    class Config:
        extra = 'forbid'


class LoggingSettings(BaseModel):
    """日志系统的配置。"""
    log_level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="全局日志级别。"
    )
    log_to_console: bool = Field(default=True, description="是否将日志输出到控制台。")
    log_to_file: bool = Field(default=False, description="是否将日志输出到文件。")
    log_file_path: Optional[Path] = Field(default=None, description="日志文件路径。") # <--【修改】FilePath 变为 Path，更通用
    log_rotation: Optional[str] = Field(
        default="10 MB", 
        description="日志文件轮换策略。" 
    )
    log_format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Loguru 日志格式字符串。"
    )
    enqueue: bool = Field(
        default=True, description="是否启用异步日志记录。"
    )
    serialize: bool = Field(
        default=False, description="是否将日志消息序列化为 JSON 格式。"
    )

    @model_validator(mode='after')
    @classmethod
    def check_log_file_path_if_logging_to_file(cls, data: Any) -> Any:
        if isinstance(data, cls):
            if data.log_to_file and not data.log_file_path:
                raise ValueError("如果 'log_to_file' 为 True, 'log_file_path' 必须提供。")
        return data

    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True
        validate_assignment = True

# ==============================================================================
# 阶段配置: P02 - 预过滤阶段 (Prefilter Stage) 
# ==============================================================================
class CzkawkaSettings(BaseModel):
    """Czkawka 工具相关的具体配置"""
    executable_path: Path = Field( # <--【修改】FilePath 变为 Path, 且非 Optional (如果启用)
        ..., 
        description="Czkawka CLI 工具的可执行文件路径。"
    )
    directories_to_scan: List[Path] = Field(
        ..., 
        description="需要进行扫描的根目录列表。"
    )
    scan_mode: Literal["duplicates"] = Field( # <--【修改】目前只关注 duplicates
        "duplicates", 
        description="Czkawka 的扫描模式 (当前固定为 'duplicates')。"
    )
    min_file_size: Optional[int] = Field(
        default=1024, # 设置一个默认值，例如 1KB
        description="Czkawka 扫描时要考虑的最小文件大小 (字节)。"
    )
    allowed_extensions: Optional[List[str]] = Field( # <--【确认】包含文件类型过滤
        default=None, 
        description="只扫描包含这些扩展名的文件 (如果为 None 或空，则由 Czkawka 决定或全扫描)。"
    )
    output_format: Literal["json"] = Field( # <--【修改】强制 json
        "json", 
        description="期望 Czkawka 输出的格式 (强制要求 json)。"
    )
    extra_args: List[str] = Field(
        default_factory=list, 
        description="传递给 Czkawka 的其他命令行参数。"
    )
    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True

class PrefilterStageSettings(BaseModel):
    """P02 - 预过滤阶段的配置。"""
    enabled: bool = Field(True, description="是否启用 P02 - 预过滤阶段。")
    tool: Literal["czkawka"] = Field("czkawka", description="当前阶段使用的预过滤工具。")
    czkawka: Optional[CzkawkaSettings] = Field(None, description="Czkawka 工具的具体配置。")
    register_files_in_storage: bool = Field(
        True, 
        description="是否在预过滤后将扫描到的文件信息注册到存储服务。"
    )

    @model_validator(mode='after')
    @classmethod
    def check_czkawka_if_enabled_and_tool_is_czkawka(cls, data: Any) -> Any:
        if isinstance(data, cls):
            if data.enabled and data.tool == "czkawka" and not data.czkawka:
                raise ValueError("如果预过滤阶段已启用且工具为 'czkawka', 'czkawka' 配置必须提供。")
        return data

    class Config:
        extra = 'forbid'

# ==============================================================================
# 阶段配置: P03 - 文档处理阶段 (Document Processing Stage)
# ==============================================================================
class DocumentProcessingStageSettings(BaseModel):
    """P03 - 文档处理阶段 (原始提取) 的配置模型。"""
    enabled: bool = Field(True, description="是否启用 P03 - 文档处理阶段。")
    parsing_strategy: Literal['auto', 'fast', 'hi_res'] = Field(
        default='auto', description="底层解析库使用的解析策略。"
    )
    supported_extensions: List[str] = Field(
        default=[".md", ".txt", ".docx", ".pdf"], # <--【扩展】包含更多类型
        description="此阶段尝试处理的文件扩展名列表。"
    )
    class Config: extra = 'forbid'

# ==============================================================================
# 阶段配置: P04 - 块合并阶段 (Block Merging Stage) - 【更新】
# ==============================================================================
class BlockMergerStageSettings(BaseModel):
    """P04 - 块合并阶段的配置。"""
    enabled: bool = Field(default=True, description="是否启用 P04 - 块合并阶段。")
    min_block_length_char: PositiveInt = Field(default=100, description="合并后块的最小字符长度（启发式规则）。")
    max_block_length_char: PositiveInt = Field(default=2000, description="合并后块的最大字符长度（启发式规则）。")
    # ... 其他合并规则 ...
    class Config: extra = 'forbid'

# ==============================================================================
# 阶段配置: 分析阶段 (Analysis Stages) - (保持不变，结构良好)
# ==============================================================================
class MD5AnalysisStageSettings(BaseModel): 
    """MD5 分析阶段的配置。"""
    enabled: bool = Field(default=True, description="是否启用 MD5 分析阶段。")
    class Config: extra = 'forbid'

class SimHashAnalysisStageSettings(BaseModel): 
    """SimHash 分析阶段的配置。"""
    enabled: bool = Field(default=True, description="是否启用 SimHash 分析阶段。")
    similarity_threshold_bits: PositiveInt = Field(default=3, description="SimHash 汉明距离阈值。")
    class Config: extra = 'forbid'

class SemanticAnalysisStageSettings(BaseModel): 
    """语义分析阶段的配置。"""
    enabled: bool = Field(default=True, description="是否启用语义分析阶段。")
    model_name_or_path: str = Field(default="shibing624/text2vec-base-chinese", description="语义分析模型名称或路径。")
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="语义相似度得分阈值。")
    batch_size: PositiveInt = Field(default=32, description="向量嵌入批处理大小。")
    device: Optional[str] = Field(default=None, description="运行模型设备 (e.g., 'cpu', 'cuda')。")
    class Config: extra = 'forbid'

class AnalysisSettings(BaseModel):
    """所有分析阶段参数的顶层配置集合。"""
    md5: MD5AnalysisStageSettings = Field(default_factory=MD5AnalysisStageSettings) 
    simhash: SimHashAnalysisStageSettings = Field(default_factory=SimHashAnalysisStageSettings) 
    semantic: SemanticAnalysisStageSettings = Field(default_factory=SemanticAnalysisStageSettings) 
    class Config: extra = 'forbid'

# ... (未来可以添加 P08 DecisionStageSettings, P09 CleanupStageSettings 等) ...
```