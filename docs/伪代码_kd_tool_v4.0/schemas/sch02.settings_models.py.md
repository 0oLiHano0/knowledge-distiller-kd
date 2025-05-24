```python
# 存放 `StorageSettingsDTO`, `OrchestratorSettings`, `LoggingSettings`, 各个 StageSettings 等

# knowledge_distiller_kd/schemas/settings_models.py
"""
该模块定义了项目中核心组件和服务所需的配置设置模型 (Settings Models)。
这些模型使用 Pydantic 定义，用于结构化和验证应用程序配置。
"""
from pydantic import BaseModel, Field, FilePath, PositiveInt, model_validator
from typing import Optional, Literal, Any, List # <--- 确保 List 已导入
from pathlib import Path

# 注意：这里不应该有对其他非Pydantic基础类型的运行时代码的依赖，
# 它的职责是纯粹的配置结构定义。

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

    class Config:
        extra = 'forbid'


class LoggingSettings(BaseModel):
    """日志系统的配置。"""
    log_level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="全局日志级别 (Loguru 支持 TRACE 和 SUCCESS)。"
    )
    log_to_console: bool = Field(default=True, description="是否将日志输出到控制台。")
    log_to_file: bool = Field(default=False, description="是否将日志输出到文件。")
    log_file_path: Optional[FilePath] = Field(default=None, description="日志文件路径。")
    log_rotation: Optional[str] = Field(
        default="10 MB", 
        description="日志文件轮换策略。默认: '10 MB'。Loguru支持其他值如 '1 day', '00:00' 等。" 
    )
    log_format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Loguru 日志格式字符串。"
    )
    enqueue: bool = Field(
        default=True, description="是否启用异步日志记录。推荐启用以提高性能并确保多线程/进程安全。"
    )
    serialize: bool = Field(
        default=False, description="是否将日志消息序列化为 JSON 格式，便于机器处理。"
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


class CzkawkaPrefilterSettings(BaseModel): # <--- 根据文件内容，这应该是 P01 或 P02 阶段的配置
    """Czkawka 预处理阶段的配置。""" # <--- 更新描述以匹配阶段概念
    enabled: bool = Field(default=True, description="是否启用 Czkawka 预处理阶段。")
    executable_path: Optional[FilePath] = Field(default=None, description="Czkawka CLI 可执行文件路径。")

    @model_validator(mode='after')
    @classmethod
    def check_executable_path_if_enabled(cls, data: Any) -> Any:
        if isinstance(data, cls):
            if data.enabled and not data.executable_path:
                raise ValueError("如果 Czkawka 预处理阶段已启用, 'executable_path' 必须提供。")
        return data

    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True

# ==============================================================================
# 阶段配置: P03 - 文档处理阶段 (Document Processing Stage - Raw Extraction)
# ==============================================================================
class DocumentProcessingStageSettings(BaseModel):
    """
    P03 - 文档处理阶段 (原始提取) 的配置模型。
    用于控制此阶段如何从文件中提取初步的内容元素。
    """
    enabled: bool = Field(
        default=True,
        description="是否启用 P03 - 文档处理阶段（原始提取）。"
    )
    parsing_strategy: Literal['auto', 'fast', 'hi_res'] = Field(
        default='auto',
        description="底层解析库（如 unstructured）使用的解析策略。"
    )
    supported_extensions: List[str] = Field(
        default=[".md"], # 当前仅处理markdown文件，未来计划扩展 ".txt", ".docx", ".pdf"等
        description="此阶段尝试处理的文件扩展名列表。不在此列表中的文件将被跳过。"
    )
    class Config:
        extra = 'forbid'

# ==============================================================================
# 阶段配置: P04 - 块合并阶段 (Block Merging Stage) - (预留位置)
# ==============================================================================
# class BlockMergingStageSettings(BaseModel):
#     """P04 - 块合并阶段的配置。"""
#     enabled: bool = Field(default=True, description="是否启用 P04 - 块合并阶段。")
#     min_block_length_char: int = Field(default=50, description="合并后块的最小字符长度（启发式规则）。")
#     # ... 其他合并规则 ...
#     class Config: extra = 'forbid'


# ==============================================================================
# 阶段配置: 分析阶段 (Analysis Stages)
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

class AnalysisSettings(BaseModel): # <--- 这是分析阶段配置的聚合器
    """所有分析阶段参数的顶层配置集合。"""
    md5: MD5AnalysisStageSettings = Field(default_factory=MD5AnalysisStageSettings) 
    simhash: SimHashAnalysisStageSettings = Field(default_factory=SimHashAnalysisStageSettings) 
    semantic: SemanticAnalysisStageSettings = Field(default_factory=SemanticAnalysisStageSettings) 
    class Config: extra = 'forbid'

# ... (未来可以添加其他 Stage 的 Settings DTO, 如 DecisionStageSettings 等) ...
```