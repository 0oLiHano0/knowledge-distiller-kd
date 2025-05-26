# kd_tool/stages/prefilter/settings_models.py (v4.6 - Prefilter Settings 迁移版)
# -*- coding: utf-8 -*-

"""
=================================================
settings_models.py - Prefilter Stage 配置模型 (v4.6)
=================================================

**模块功能**:

- **核心职责**: 定义 Prefilter Stage (`kd_tool.stages.prefilter`) 所需的配置设置模型。
- **v4.6 核心变更**:
    - **[架构指令]** `CzkawkaSettings` 和 `PrefilterStageSettings` 从原 `schemas` 目录迁移至此。

---
"""

# --- Python 标准库及第三方库导入 ---
from typing import Optional, Literal, Any, List
from pathlib import Path

# --- Pydantic 导入 ---
from pydantic import BaseModel, Field, model_validator


# ==============================================================================
# Czkawka 工具配置 (CzkawkaSettings)
# ==============================================================================
# [架构师说明]: CzkawkaSettings 定义了与 Czkawka 工具交互的具体参数。
#               它之前位于 schemas/settings_models.py。

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
        default=1024, # 默认 1KB
        ge=0, # 确保非负
        description="Czkawka 扫描时要考虑的最小文件大小 (字节)。"
    )
    allowed_extensions: Optional[List[str]] = Field(
        default=None,
        description="只扫描包含这些扩展名的文件 (如果为 None 或空，则由 Czkawka 决定或全扫描)。例如: ['.txt', '.md']"
    )
    output_format: Literal["json"] = Field( # 在 settings 中明确，即使 adapter 内部可能硬编码
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

# ==============================================================================
# 预过滤阶段配置 (PrefilterStageSettings)
# ==============================================================================
# [架构师说明]: PrefilterStageSettings 是 Prefilter Stage 的顶层配置。
#               它之前位于 schemas/settings_models.py。

class PrefilterStageSettings(BaseModel):
    """P02 - 预过滤阶段的配置。"""
    enabled: bool = Field(
        default=True,
        description="是否启用 P02 - 预过滤阶段。"
    )
    tool: Literal["czkawka"] = Field(
        default="czkawka",
        description="当前阶段使用的预过滤工具。**规范**: 未来可扩展支持其他工具。"
    )
    czkawka: Optional[CzkawkaSettings] = Field( # [指令] 类型为上面定义的 CzkawkaSettings
        default=None, # 保持 default=None，因为可能 tool 不是 czkawka (虽然目前只有它)
        description="Czkawka 工具的具体配置。**规范**: 如果 enabled 且 tool 为 'czkawka'，此项必填。"
    )
    register_files_in_storage: bool = Field(
        default=True,
        description="是否在预过滤后将扫描到的文件信息注册到存储服务。"
    )

    @model_validator(mode='after')
    @classmethod
    def check_czkawka_if_enabled_and_tool_is_czkawka(cls, data: Any) -> Any:
        """**验证器**: 确保 'czkawka' 配置在需要时提供。"""
        if isinstance(data, cls): # 确保 data 是 PrefilterStageSettings 的实例
            if data.enabled and data.tool == "czkawka" and not data.czkawka:
                raise ValueError("如果预过滤阶段已启用且工具为 'czkawka', 'czkawka' 配置必须提供。")
        return data

    class Config:
        extra = 'forbid'