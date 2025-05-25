```python
# kd_tool/schemas/settings_models.py (v4.5.1 - 增强注释版)
# -*- coding: utf-8 -*-

"""
=================================================
sch02.settings_models.py.md - KD_Tool 配置设置模型
=================================================

**模块功能**:

- **核心职责**: 定义项目中所有核心组件和服务所需的配置设置模型 (Settings Models)。
- **技术选型**: 使用 Pydantic 定义，以实现配置的结构化、类型安全和自动验证。
- **目标**: 为 coding 阶段提供清晰、无歧义的配置项定义和约束。

**版本历史**:
- v4.1: 引入 PrefilterStageSettings，明确 BlockMerger 配置。
- v4.5: 详细定义 SimHashAnalysisStageSettings，并为 Decision 和 Cleanup 阶段添加占位符。
- v4.5.1: 【架构师决策】恢复并增强注释，明确各配置项的规范和设计意图。

---
"""

# 导入 Pydantic 核心组件和 Python 类型提示
from pydantic import BaseModel, Field, PositiveInt, model_validator, NonNegativeInt, conint, confloat
from typing import Optional, Literal, Any, List, Dict
from pathlib import Path

# 从 .enums 导入 DecisionType 
from .enums import DecisionType 

# ==============================================================================
# 基础服务配置 (Base Services Configuration)
# ==============================================================================
# [架构师说明]: 这部分定义了 KD_Tool 运行所需的基础服务的配置，
#               如存储、编排器和日志记录。它们是整个应用的基础设施。

class StorageSettingsDTO(BaseModel):
    """
    存储服务的配置设置 DTO。
    **规范**: 定义所有与数据持久化相关的配置。
    """
    backend_type: str = Field(
        default="sqlite",
        description="""
        存储后端类型。
        **规范**: 目前主要支持 'sqlite'。未来可扩展至 'memory_debug' 等。
        **编码要求**: StorageFactory 将根据此类型选择具体的 StorageInterface 实现。
        """
    )
    connection_string: Optional[str] = Field(
        default=None,
        description="""
        数据库连接字符串。
        **规范**: 对于 'sqlite'，格式为 'sqlite:///path/to/your/db/file.db'。
        **编码要求**: 如果 backend_type 为 'sqlite'，此字段 **必须** 提供。
        """
    )
    base_directory: Optional[Path] = Field(
        default=None,
        description="""
        文件系统存储的基础目录路径 (如果后端使用文件系统)。
        **规范**: 用于存储可能需要持久化的非数据库文件（如原始文件备份、缓存等）。
        **编码要求**: 具体使用方式由具体的 StorageInterface 实现决定。
        """
    )

    @model_validator(mode='after')
    @classmethod
    def check_consistency(cls, data: Any) -> Any:
        """
        **验证器**: 确保 'sqlite' 后端提供了 'connection_string'。
        **规范**: 必须确保配置的内部一致性。
        """
        if isinstance(data, cls):
            if data.backend_type == "sqlite" and not data.connection_string:
                raise ValueError("对于 'sqlite' 后端类型, 'connection_string' 必须提供。")
        return data

    class Config:
        extra = 'forbid' # **规范**: 禁止 AppConfig 中出现未在此定义的额外字段。
        arbitrary_types_allowed = True # 允许 Path 等非 Pydantic 基本类型。

class OrchestratorSettings(BaseModel):
    """
    Orchestrator 模块的配置设置。
    **规范**: 定义流水线调度和执行行为的参数。
    """
    on_pipeline_error_policy: Literal['HALT_ON_FIRST_ERROR', 'CONTINUE_IGNORING_ERROR'] = Field(
        default='HALT_ON_FIRST_ERROR',
        description="""
        流水线错误处理策略。
        - HALT_ON_FIRST_ERROR: 遇到第一个 Stage 错误时，立即停止整个流水线。
        - CONTINUE_IGNORING_ERROR: 记录错误并继续执行下一个 Stage。
        **编码要求**: Orchestrator 的 `run` 方法必须根据此策略进行错误处理。
        """
    )
    default_task_id_prefix: str = Field(
        default='kd_task_',
        description="""
        为生成的 task_id 添加的可选前缀 (如果未使用 UUID 或需要附加信息)。
        **规范**: 主要用于日志追踪和调试。
        **编码要求**: 在 PipelineContextDTO 初始化时考虑使用。
        """
    )
    default_stage_order: List[str] = Field(
        default=[
            "prefilter", 
            "document_processing", 
            "block_merging", 
            "md5_analysis",
            "simhash_analysis", 
            "semantic_analysis", 
            "decision", 
            "cleanup" 
        ],
        description="""
        默认情况下流水线中各个阶段的执行顺序和名称。
        **规范**: 这里的名称 **必须** 与 `ApplicationBuilder` 中注册 Stage 时使用的键名一致。
        **编码要求**: Orchestrator 将按此列表顺序执行 Stage。
        """
    )

    class Config:
        extra = 'forbid'

class LoggingSettings(BaseModel):
    """
    日志系统的配置。
    **规范**: 定义 Loguru 日志系统的所有行为参数。
    """
    log_level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", 
        description="全局日志级别。"
    )
    log_to_console: bool = Field(default=True, description="是否将日志输出到控制台。")
    log_to_file: bool = Field(default=False, description="是否将日志输出到文件。")
    log_file_path: Optional[Path] = Field(
        default=None, 
        description="日志文件路径。**规范**: 如果 log_to_file 为 True，此项必填。"
    )
    log_rotation: Optional[str] = Field(
        default="10 MB", 
        description="日志文件轮换策略 (Loguru 格式, e.g., '10 MB', '1 week', '00:00')。" 
    )
    log_format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Loguru 日志格式字符串。"
    )
    enqueue: bool = Field(
        default=True, 
        description="是否启用异步日志记录，以提高性能。"
    )
    serialize: bool = Field(
        default=False, 
        description="是否将日志消息序列化为 JSON 格式，便于机器处理。"
    )

    @model_validator(mode='after')
    @classmethod
    def check_log_file_path_if_logging_to_file(cls, data: Any) -> Any:
        """
        **验证器**: 确保 'log_to_file' 为 True 时提供了 'log_file_path'。
        """
        if isinstance(data, cls):
            if data.log_to_file and not data.log_file_path:
                raise ValueError("如果 'log_to_file' 为 True, 'log_file_path' 必须提供。")
        return data

    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True
        validate_assignment = True # 允许在运行时修改并验证

# ==============================================================================
# 阶段配置: P02 - 预过滤阶段 (Prefilter Stage) 
# ==============================================================================
# [架构师说明]: 定义用于预处理阶段 (如使用 Czkawka 进行初步去重) 的配置。

class CzkawkaSettings(BaseModel):
    """Czkawka 工具相关的具体配置。"""
    executable_path: Path = Field( 
        ..., # '...' 表示此字段是必需的
        description="Czkawka CLI 工具的可执行文件路径。"
    )
    directories_to_scan: List[Path] = Field(
        ..., 
        description="需要进行扫描的根目录列表。"
    )
    scan_mode: Literal["duplicates"] = Field( 
        "duplicates", 
        description="Czkawka 的扫描模式。**规范**: 当前 v4.0 只关注 'duplicates'。"
    )
    min_file_size: Optional[int] = Field(
        default=1024, 
        description="Czkawka 扫描时要考虑的最小文件大小 (字节)。默认 1KB。"
    )
    allowed_extensions: Optional[List[str]] = Field( 
        default=None, 
        description="只扫描包含这些扩展名的文件 (如果为 None 或空，则由 Czkawka 决定或全扫描)。"
    )
    output_format: Literal["json"] = Field( 
        "json", 
        description="期望 Czkawka 输出的格式。**规范**: PrefilterStage **必须**处理 JSON 输出。"
    )
    extra_args: List[str] = Field(
        default_factory=list, 
        description="传递给 Czkawka 的其他命令行参数 (高级用户选项)。"
    )
    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True

class PrefilterStageSettings(BaseModel):
    """P02 - 预过滤阶段的配置。"""
    enabled: bool = Field(True, description="是否启用 P02 - 预过滤阶段。")
    tool: Literal["czkawka"] = Field("czkawka", description="当前阶段使用的预过滤工具。**规范**: 未来可扩展支持其他工具。")
    czkawka: Optional[CzkawkaSettings] = Field(None, description="Czkawka 工具的具体配置。**规范**: 如果 enabled 且 tool 为 'czkawka'，此项必填。")
    register_files_in_storage: bool = Field(
        True, 
        description="是否在预过滤后将扫描到的文件信息注册到存储服务。**规范**: 推荐 True，以便后续阶段使用。"
    )

    @model_validator(mode='after')
    @classmethod
    def check_czkawka_if_enabled_and_tool_is_czkawka(cls, data: Any) -> Any:
        """**验证器**: 确保 'czkawka' 配置在需要时提供。"""
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
        default='auto', 
        description="底层解析库 (如 `unstructured`) 使用的解析策略。"
    )
    supported_extensions: List[str] = Field(
        default=[".md", ".txt", ".docx", ".pdf"], 
        description="此阶段尝试处理的文件扩展名列表。"
    )
    class Config: extra = 'forbid'

# ==============================================================================
# 阶段配置: P04 - 块合并阶段 (Block Merging Stage)
# ==============================================================================
class BlockMergerStageSettings(BaseModel):
    """P04 - 块合并阶段的配置。"""
    enabled: bool = Field(default=True, description="是否启用 P04 - 块合并阶段。")
    min_block_length_char: PositiveInt = Field(default=100, description="合并后块的最小字符长度（启发式规则）。")
    max_block_length_char: PositiveInt = Field(default=2000, description="合并后块的最大字符长度（启发式规则）。")
    # **未来设计**: 可以添加更多合并策略配置，如按标题合并、按空行合并等。
    class Config: extra = 'forbid'

# ==============================================================================
# 阶段配置: P05 - MD5 分析阶段
# ==============================================================================
class MD5AnalysisStageSettings(BaseModel): 
    """P05 - MD5 分析阶段的配置。"""
    enabled: bool = Field(default=True, description="是否启用 P05 - MD5 分析阶段 (用于精确去重)。")
    class Config: extra = 'forbid'

# ==============================================================================
# 阶段配置: P06 - SimHash 分析阶段 - 【v4.5.1 细化注释】
# ==============================================================================
class SimHashAnalysisStageSettings(BaseModel): 
    """P06 - SimHash 分析阶段的配置。"""
    enabled: bool = Field(default=True, description="是否启用 P06 - SimHash 分析阶段 (用于近似去重)。")
    
    hash_bits: Literal[64, 128] = Field(
        default=64, 
        description="""
        SimHash 指纹的位数。
        **规范**: 必须是 64 或 128。64 位速度更快，128 位精度更高。
        **编码要求**: SimHash 适配器和 Stage 必须处理此配置。
        """
    )
    
    hamming_distance_threshold: conint(ge=0, le=128) = Field( 
        default=3, 
        description="""
        SimHash 汉明距离阈值。
        **规范**: 两个块的汉明距离 <= 此阈值时，被视为相似。
                 取值范围必须在 [0, hash_bits] 之间。
        **编码要求**: Stage 必须使用此阈值过滤比较结果。
        """
    )
    
    force_recalculate: bool = Field(
        default=False,
        description="是否强制重新计算所有块的 SimHash 值，即使它们已存在。**规范**: 用于调试或策略变更。"
    )
    
    @model_validator(mode='after')
    @classmethod
    def check_threshold_within_bits(cls, data: Any) -> Any:
        """**验证器**: 确保汉明距离阈值不超过哈希位数。"""
        if isinstance(data, cls):
            if data.hamming_distance_threshold > data.hash_bits:
                raise ValueError(
                    f"汉明距离阈值 ({data.hamming_distance_threshold}) "
                    f"不能大于哈希位数 ({data.hash_bits})。"
                )
        return data

    class Config: extra = 'forbid'

# ==============================================================================
# 阶段配置: P07 - 语义分析阶段
# ==============================================================================
class SemanticAnalysisStageSettings(BaseModel): 
    """
    P07 - 语义分析阶段的配置。
    **规范**: 定义语义分析模型、阈值和执行参数。
    """
    enabled: bool = Field(default=True, description="是否启用 P07 - 语义分析阶段 (用于语义去重)。")
    
    model_name_or_path: str = Field(
        default="shibing624/text2vec-base-chinese", 
        description="""
        语义分析模型名称 (来自 Hugging Face 等) 或本地路径。
        **规范**: 需要选择适合中文且性能可接受的模型。适配器将使用此路径加载模型。
        """
    )
    
    similarity_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, 
        description="""
        语义相似度得分阈值。
        **规范**: [0.0, 1.0] 范围。两个块的余弦相似度 >= 此阈值时，被视为相似。
        """
    )
    
    batch_size: PositiveInt = Field(
        default=32, 
        description="""
        向量嵌入批处理大小。
        **规范**: 影响性能和显存占用。适配器应支持按此批次大小处理文本。
        """
    )
    
    device: Optional[str] = Field(
        default=None, 
        description="""
        运行模型设备 (e.g., 'cpu', 'cuda', 'cuda:0')。
        **规范**: 如果为 None，库通常会自动选择 (优先 GPU)。适配器应能处理此参数。
        """
    )

    comparison_strategy: Literal["all_pairs", "pre_filtered"] = Field(
        default="pre_filtered",
        description="""
        比较策略。
        - 'all_pairs': 比较所有内容块 (计算成本极高)。
        - 'pre_filtered': (推荐) 仅比较那些未被 MD5 或 SimHash 识别为完全/高度相似的块对。
        **编码要求**: Stage 必须根据此策略决定哪些块对需要进行语义比较。
        """
    )

    class Config: extra = 'forbid'

# ==============================================================================
# 阶段配置: P08 - 决策阶段 (Decision Stage) - 【v4.6 细化】
# ==============================================================================

class DecisionRule(BaseModel):
    """
    定义一条决策规则。
    **规范**: 用于描述当满足某些分析条件时，应采取何种决策。
    """
    md5_score: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=None, 
        description="触发此规则的 MD5 分数 (通常是 1.0)。如果为 None，则不考虑 MD5。"
    )
    simhash_similarity_min: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=None, 
        description="触发此规则的 SimHash 最小相似度。如果为 None，则不考虑。"
    )
    semantic_similarity_min: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=None, 
        description="触发此规则的语义最小相似度。如果为 None，则不考虑。"
    )
    decision_to_apply: DecisionType = Field(
        ..., # 必须指定决策动作
        description="当满足以上所有（非 None）条件时，要应用的决策。"
    )
    rule_priority: int = Field(
        default=0, 
        description="规则优先级。**规范**: 数字越大，优先级越高。用于处理一个块对可能匹配多条规则的情况。"
    )

class DecisionStageSettings(BaseModel):
    """
    P08 - 决策阶段的配置。
    **架构师说明**: 此阶段将根据前面所有分析结果生成决策。
                   其配置将涉及如何组合不同分析分数、设置决策规则等。
    """
    enabled: bool = Field(True, description="是否启用 P08 - 决策阶段。")
    
    rules: List[DecisionRule] = Field(
        default_factory=lambda: [
            # **默认规则示例**:
            # 1. MD5 匹配 -> 标记为 DELETE (高优先级)
            DecisionRule(
                md5_score=1.0, 
                decision_to_apply=DecisionType.DELETE, 
                rule_priority=100
            ),
            # 2. 语义 > 0.95 -> 标记为 DELETE (次高优先级)
            DecisionRule(
                semantic_similarity_min=0.95, 
                decision_to_apply=DecisionType.DELETE, 
                rule_priority=90
            ),
            # 3. SimHash > 0.98 (汉明距离 < 2/64) -> 标记为 UNDECIDED (中优先级)
            DecisionRule(
                simhash_similarity_min=0.97, # 1.0 - (2 / 64) = 0.96875 -> 0.97
                decision_to_apply=DecisionType.UNDECIDED, 
                rule_priority=80
            ),
            # 4. 语义 > 0.85 -> 标记为 UNDECIDED (低优先级)
            DecisionRule(
                semantic_similarity_min=0.85, 
                decision_to_apply=DecisionType.UNDECIDED, 
                rule_priority=70
            ),
        ],
        description="""
        决策规则列表。
        **规范**: DecisionStage 将按优先级从高到低评估这些规则。
                 对于每一对分析结果，将应用第一个匹配的、优先级最高的规则。
        """
    )

    default_decision: DecisionType = Field(
        default=DecisionType.KEEP,
        description="""
        如果没有任何规则匹配分析结果，则应用的默认决策。
        **规范**: 通常设置为 'KEEP' 或 'UNDECIDED'。
        """
    )
    
    process_undecided: bool = Field(
        default=False,
        description="是否为 'UNDECIDED' 的结果创建 UserDecisionDTO。**规范**: 如果为 False，则只有明确的决策会被记录。"
    )

    class Config: 
        extra = 'forbid'
        arbitrary_types_allowed = True

# ==============================================================================
# 阶段配置: P09 - 清理阶段 (Cleanup Stage) - 【v4.7 细化】
# ==============================================================================
class CleanupStageSettings(BaseModel):
    """
    P09 - 清理阶段的配置。
    **架构师说明**: 此阶段将执行 `DecisionStage` 产生的决策。
                   其配置将涉及具体的文件操作（标记、移动、删除）等。
                   **安全第一**: 默认配置应采用最安全的方式 (mark_only)。
    """
    enabled: bool = Field(True, description="是否启用 P09 - 清理阶段。")

    action_map: Dict[DecisionType, Literal['mark_only', 'move_to_trash', 'permanent_delete', 'ignore']] = Field(
        default_factory=lambda: {
            DecisionType.DELETE: 'mark_only', # **默认**: 对标记为 DELETE 的仅做标记
            DecisionType.KEEP: 'ignore',      # 忽略 KEEP
            DecisionType.UNDECIDED: 'ignore', # 忽略 UNDECIDED
            DecisionType.IGNORE_PAIR: 'ignore' # 忽略 IGNORE_PAIR
        },
        description="""
        决策类型到具体清理动作的映射。
        - 'mark_only': (最安全) 仅更新数据库中 FileRecordDTO 的状态为 'MARKED_FOR_DELETION'。
        - 'move_to_trash': 将物理文件移动到指定的 'trash_directory' 并更新状态。
        - 'permanent_delete': (危险!) **永久删除**物理文件并更新状态 (或删除记录)。
        - 'ignore': 对此决策类型不执行任何操作。
        **规范**: 必须为所有 DecisionType 提供映射 (或有默认处理)。
        """
    )

    trash_directory: Optional[Path] = Field( # <-- Path 更通用
        default=None, 
        description="""
        垃圾箱目录的路径。
        **规范**: 如果 `action_map` 中有任何值设为 'move_to_trash'，此字段 **必须** 提供且必须是有效目录。
        """
    )

    # **未来设计**: 可以增加更细粒度的配置，例如是否删除数据库记录等。

    @model_validator(mode='after')
    @classmethod
    def check_trash_dir_if_needed(cls, data: Any) -> Any:
        """**验证器**: 如果需要移动到垃圾箱，确保垃圾箱目录已提供。"""
        if isinstance(data, cls):
            needs_trash = any(action == 'move_to_trash' for action in data.action_map.values())
            if needs_trash and not data.trash_directory:
                raise ValueError("如果 'action_map' 中包含 'move_to_trash'，则 'trash_directory' 必须提供。")
            # **编码要求**: 在实际实现中，还应验证 trash_directory 是否存在且可写。
        return data

    class Config: 
        extra = 'forbid'
        arbitrary_types_allowed = True

# ==============================================================================
# 顶层分析配置 (Analysis Settings)
# ==============================================================================
class AnalysisSettings(BaseModel):
    """
    所有分析阶段参数的顶层配置集合。
    **规范**: 便于集中管理所有分析相关的配置。
    """
    md5: MD5AnalysisStageSettings = Field(default_factory=MD5AnalysisStageSettings) 
    simhash: SimHashAnalysisStageSettings = Field(default_factory=SimHashAnalysisStageSettings) 
    semantic: SemanticAnalysisStageSettings = Field(default_factory=SemanticAnalysisStageSettings) 
    class Config: extra = 'forbid'

# ==============================================================================
# 顶层应用配置 (AppConfig)
# ==============================================================================
class AppConfig(BaseModel):
    """
    KD_Tool v4.0 应用程序的顶层配置模型。
    **规范**: 这是应用程序配置的唯一入口点 (Single Source of Truth)。
             它聚合了所有子模块的配置。
    **编码要求**: `ApplicationBuilder` 将负责加载此配置并分发给各个工厂。
    """
    # 基础服务
    storage: StorageSettingsDTO = Field(default_factory=StorageSettingsDTO, description="存储服务配置。")
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings, description="编排器配置。")
    logging: LoggingSettings = Field(default_factory=LoggingSettings, description="日志系统配置。")

    # 各个阶段的配置
    prefilter: PrefilterStageSettings = Field(default_factory=PrefilterStageSettings, description="P02 预过滤阶段配置。")
    document_processing: DocumentProcessingStageSettings = Field(default_factory=DocumentProcessingStageSettings, description="P03 文档处理阶段配置。")
    block_merging: BlockMergerStageSettings = Field(default_factory=BlockMergerStageSettings, description="P04 块合并阶段配置。")
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings, description="所有分析阶段 (P05, P06, P07) 的配置集合。") 
    decision: DecisionStageSettings = Field(default_factory=DecisionStageSettings, description="P08 决策阶段配置。") 
    cleanup: CleanupStageSettings = Field(default_factory=CleanupStageSettings, description="P09 清理阶段配置。")

    class Config:
        extra = 'forbid' # 再次强调，顶层配置也禁止未知字段。
        arbitrary_types_allowed = True
        # **未来实现**: 可以添加加载和保存配置的方法 (e.g., `load_from_yaml`, `save_to_yaml`)。

        
```