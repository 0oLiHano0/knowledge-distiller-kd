"""
====================开发指引======================
kd_tool/core/core_settings_models.py - v4.7
=================================================

**【文件定位】**
- 路径：kd_tool/core/core_settings_models.py
- 所属：核心服务层（core），为 Orchestrator 及其工厂/服务提供配置模型与异常定义。
- 依赖关系：被 Orchestrator、工厂、配置中心等通过依赖注入方式调用。

**【模块职责（SRP）】**
- 唯一职责：定义 Orchestrator 及核心流程调度相关的配置数据模型与专属异常类型，确保配置的类型安全、校验与错误可追踪。

**【依赖关系与注入】**
- 依赖外部：pydantic.BaseModel、kd_tool.core.errors.ConfigError
- 注入方式：仅允许通过工厂/构造函数注入配置对象，严禁全局变量或单例。
- Mock点：如需测试配置异常，可 Mock OrchestratorSettingsError 抛出场景。

**【输入输出规范】**
- OrchestratorSettings
  - 输入：各字段（如 on_pipeline_error_policy: Literal[...]、default_stage_order: List[str]），类型严格限定。
  - 输出：Pydantic 校验后的配置对象。
  - 异常：字段校验失败时，需在工厂/加载逻辑中捕获并转为 OrchestratorSettingsError 抛出。
- OrchestratorSettingsError
  - 输入：Pydantic 校验异常及上下文。
  - 输出：异常对象，供上层捕获。
- DTO/ORM边界：本文件仅定义 DTO（Pydantic），不涉及 ORM。

**【核心架构约束】**
- 禁止直接实例化依赖，所有依赖通过注入。
- 禁止业务逻辑与存储耦合。
- 所有字段/方法必须类型注解。
- 所有配置校验错误必须转为自定义异常（OrchestratorSettingsError），继承自 KDToolError。
- 重要类/方法需三段式注释（WHY/WHAT/HOW）。
- 禁止全局变量、禁止直接读取配置文件。
- 仅允许通过依赖注入获取配置，禁止直接访问全局配置。

**【接口与DTO规范】**
- 暴露接口/DTO/异常：
  - OrchestratorSettings（Pydantic模型，字段类型详见代码）
  - OrchestratorSettingsError（自定义异常，承载校验失败信息）
- 接口定义与实现分离：本文件仅定义数据结构与异常，不含实现逻辑。

**【日志与安全】**
- 本文件不直接产生日志，但要求所有配置加载/校验异常在调用方通过 Loguru 记录（logger.exception），并绑定上下文（如 task_id）。
- 不涉及敏感信息处理，但如有敏感配置字段，需在日志中脱敏。

**【任务清单】**
1. 【已实现】OrchestratorSettings 所有字段类型、默认值、描述，确保与 Orchestrator 及 ApplicationBuilder 阶段注册严格一致。
2. 【已实现】定义 OrchestratorSettingsError，继承自 ConfigError，三段式注释齐全。
3. 【已实现】所有字段/方法类型注解、Pydantic 校验、extra=forbid 配置。
4. 【需补充】在工厂/加载逻辑中实现 Pydantic 校验异常到 OrchestratorSettingsError 的转换（当前文件仅声明，具体转换需在工厂/加载器中实现）。
5. 【需补充】单元测试：覆盖配置校验、异常抛出、DTO边界等场景，确保类型安全与异常可追踪。
6. 【需补充】文档完善：补充/校验所有三段式注释，确保 WHY/WHAT/HOW 齐全，便于后续维护。
7. 【需补充】安全审查：如未来新增敏感配置字段，需在日志与异常中自动脱敏。

**【其他说明】**
- 未来如需扩展配置项，必须保持向后兼容，新增字段需有默认值。
- 若有配置项涉及安全/密钥，需在日志与异常中自动脱敏。
- 本文件为配置模型定义层，严禁包含任何业务逻辑或存储操作。
"""

from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict
from kd_tool.core.errors import ConfigError


class OrchestratorSettingsError(ConfigError):
    """
    WHY: 标识OrchestratorSettings配置校验或加载失败，便于上层精准捕获
    WHAT: 封装Pydantic ValidationError及上下文信息
    HOW: 仅作类型声明，异常转换在工厂/加载逻辑中完成
    """

    pass


class OrchestratorSettings(BaseModel):
    """
    WHY: 统一管理Orchestrator模块的调度与执行配置，支撑流程可控与可扩展
    WHAT: 定义流水线错误处理策略与默认阶段顺序，类型安全、可校验
    HOW: 通过Pydantic模型校验，所有字段类型注解，extra=forbid
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    on_pipeline_error_policy: Literal[
        "HALT_ON_FIRST_ERROR", "CONTINUE_IGNORING_ERROR"
    ] = Field(
        default="HALT_ON_FIRST_ERROR",
        description="""
        流水线错误处理策略。
        - HALT_ON_FIRST_ERROR: 遇到第一个 Stage 错误时，立即停止整个流水线。
        - CONTINUE_IGNORING_ERROR: 记录错误并继续执行下一个 Stage。
        **编码要求**: Orchestrator 的 `run` 方法必须根据此策略进行错误处理。
        """,
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
            "cleanup",
        ],
        description="""
        默认情况下流水线中各个阶段的执行顺序和名称。
        **规范**: 这里的名称 **必须** 与 `ApplicationBuilder` 中注册 Stage 时使用的键名一致。
        **编码要求**: Orchestrator 将按此列表顺序执行 Stage。
        """,
    )
