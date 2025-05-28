## 最高规则符合性检查

| 设计规则             | 符合点                           |
| -------------- | ------------------------------- |
| 1 SRP          | Factory 只创建；Service 只记录         |
| 2 高内聚低耦合       | 业务仅依赖 `LoggerProtocol`          |
| 3 接口契约         | `LoggerProtocol` 明确最小方法集        |
| 4 Testability  | 注入 FakeLogger；单元测试无外部依赖         |
| 5 OCP          | 换用 stdlib `logging` 只需改 Factory |
| 6 DI           | 所有模块接受 `logger` 参数              |
| 7 Factory      | `LoggerFactory` 统一装配            |
| 8 Pydantic     | `LoggingSettingsDTO` 管配置        |
| 9 Custom Error | `LoggingError`                  |
| 10 Loguru 遵从   | 绑定 task\_id；唯一耦合点在 Factory      |
| 11 Stateless   | `LoggingService` 无内部可变状态        |
| 12 Type Hint   | 全面类型提示 & `Protocol`             |
| 13 DTO vs ORM  | 不涉及                             |
| 14 集中配置        | 通过 AppConfig 注入 DTO             |
| 15 事务边界        | 日志独立于业务事务                       |
| 16 绝对导入        | 全部示例均为绝对路径                      |


## logging注入示例

```python

# 所有模块只依赖 LoggerProtocol 类型的参数，不导入 loguru。
# 符合 高内聚 / 低耦合、DI、可测试性

# factories/app_factory.py
# factories/app_factory.py
from kd_tool.logging import LoggerFactory
from kd_tool.logging.settings import LoggingSettingsDTO
from kd_tool.orchestrator.main import Orchestrator      # 示例

class AppFactory:
    """
    WHY : 构建应用核心对象  
    WHAT: 创建 Orchestrator 并注入依赖  
    HOW : 采用工厂聚合
    """
    def __init__(self, cfg: AppConfig) -> None:
        self._logger_fac = LoggerFactory(LoggingSettingsDTO(**cfg.logging.dict()))

    def create_orchestrator(self) -> Orchestrator:
        logger = self._logger_fac.get_logger()
        return Orchestrator(logger=logger, ...)   # 其他依赖省略


```