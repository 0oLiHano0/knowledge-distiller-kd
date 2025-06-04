"""
====================开发指引======================
kd_tool/core/interfaces.py - v0.1
=================================================

**【文件定位】**  
- 所属包结构：kd_tool.core
- 所在层次：核心服务层（Core Layer）
- 主要作用：为所有流水线阶段模块（Stage）定义统一的抽象接口契约，确保各阶段实现解耦、可插拔。

**【模块职责（SRP）】**  
- 唯一职责：定义流水线阶段模块的标准接口，规范其输入、输出及异常处理，保证Orchestrator与各阶段模块的解耦。

**【依赖关系与注入】**  
- 依赖：PipelineContextDTO（Pydantic模型，定义于kd_tool.core.core_dtos）
- 异常：KDToolError（自定义异常，需在实现类中抛出）
- 本接口不直接依赖外部服务，所有实现类的依赖（如Logger、配置等）必须通过构造器注入，禁止内部实例化依赖。
- Mock点：可通过Mock实现StageInterface进行上层Orchestrator等模块的单元测试。

**【输入输出规范】**  
- 方法签名：process(self, context: PipelineContextDTO) -> PipelineContextDTO
    - 输入参数：
        - context: PipelineContextDTO
            - 类型：Pydantic模型
            - 说明：包含所有任务数据的管道上下文
    - 返回值：
        - PipelineContextDTO
            - 说明：处理和更新后的管道上下文
    - 异常：
        - KDToolError或其子类
            - 说明：阶段处理发生错误时抛出，禁止抛出通用Exception
- DTO/ORM边界：仅允许DTO对象在接口间传递，禁止传递ORM对象。

**【核心架构约束】**  
- 禁止直接实例化依赖，所有依赖必须通过构造器注入。
- 禁止业务逻辑与存储耦合，所有数据交互通过DTO完成。
- 必须为所有方法添加类型注解。
- 异常处理需结构化，抛出自定义异常。
- 重要实现点（如process方法）在实现类中需添加三段式注释（WHY/WHAT/HOW）。
- 禁止在接口或实现类中持久化带有动态上下文的logger，日志上下文绑定仅在本地变量中完成。

**【接口与DTO规范】**  
- 关键接口：StageInterface（抽象基类）
    - 方法：process(self, context: PipelineContextDTO) -> PipelineContextDTO
- DTO：PipelineContextDTO（Pydantic模型）
- 异常类：KDToolError及其子类
- 接口定义与实现分离，所有阶段模块必须实现StageInterface。

**【日志与安全】**  
- 日志记录点：本接口不涉及日志，所有实现类需注入Logger并在关键处理点记录日志。
- 日志级别：需根据事件重要性选择合适级别，异常需使用logger.exception。
- 敏感信息处理：实现类需确保日志中不泄露敏感信息。
- 安全约束：如涉及权限或数据安全，需在实现类中明确处理。

**【任务清单】**  
1. [已完成] 定义StageInterface抽象基类
2. [已完成] 明确process方法签名、输入输出、异常
3. [待完成] 单元测试与契约测试

**【其他说明】**  
- 若未来需扩展接口（如增加生命周期钩子），需保持向后兼容。
- 所有实现类必须无状态，依赖通过构造器注入。
- 本文件所有导入必须使用绝对导入路径。
"""
from abc import ABC, abstractmethod
from kd_tool.core.core_dtos import PipelineContextDTO


class StageInterface(ABC):
    """
    抽象基类，定义了流水线阶段模块的契约。
    所有阶段模块都必须实现 `process` 方法。
    """

    @abstractmethod
    def process(self, context: PipelineContextDTO) -> PipelineContextDTO:
        """
        处理管道上下文并返回更新后的上下文。

        Orchestrator 会为流水线中的每个阶段调用此方法。
        每个阶段接收 PipelineContextDTO，对其进行修改（添加数据、
        更新状态等），然后将其返回给下一个阶段。

        参数:
            context (PipelineContextDTO): 包含所有任务数据的管道上下文。

        返回:
            PipelineContextDTO: 处理和更新后的管道上下文。

        抛出:
            KDToolError 或其子类: 当阶段处理发生错误时。
        """
        ...
