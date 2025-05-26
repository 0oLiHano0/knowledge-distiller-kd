# kd_tool/core/interfaces.py （4.1）

from abc import ABC, abstractmethod
# [指令] 必须从同级目录下的 core_dtos.py 导入 PipelineContextDTO
from .core_dtos import PipelineContextDTO

class StageInterface(ABC):
    """
    抽象基类，定义了流水线阶段模块的契约 (v4.x)。
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
        raise NotImplementedError("每个阶段模块都必须实现 'process' 方法。")

# ... StorageInterface 及其它核心接口 ...
