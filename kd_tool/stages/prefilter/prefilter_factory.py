"""
=================================================
prefilter_factory.py - PrefilterStage 工厂 (v4.7)
=================================================

**模块功能**:

- 负责创建并组装 PrefilterStage 实例及其依赖。
- 与 Storage 解耦，仅依赖于 context 和 settings。

"""
from kd_tool.logging.protocols import LoggerProtocol
from kd_tool.core.interfaces import StageInterface
from kd_tool.core.errors import KDToolError
from kd_tool.stages.prefilter.prefilter_stage import PrefilterStage
from kd_tool.stages.prefilter.adapter_interface import CzkawkaAdapterInterface
from kd_tool.stages.prefilter.czkawka_adapter import CzkawkaAdapter
from kd_tool.stages.prefilter.settings_models import PrefilterStageSettings
from typing import Optional

class FactoryConfigurationError(KDToolError):
    """当工厂遇到配置问题时抛出。"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, module='PrefilterStageFactory', **kwargs)


class PrefilterStageFactory:
    """
    WHY: 负责创建PrefilterStage实例。
    WHAT: 依赖注入logger，create方法组装所有依赖。
    HOW: 工厂模式，便于测试和扩展。
    """

    def __init__(self, logger: LoggerProtocol):
        """
        初始化预过滤阶段工厂。
        **参数**:
            logger (LoggerProtocol): **[必须]** 日志记录器实例。
        """
        self._logger = logger.bind(factory_name='PrefilterStageFactory')
        self._logger.info('PrefilterStageFactory 初始化完成.')

    def create(
        self,
        settings: PrefilterStageSettings,
        adapter: Optional[CzkawkaAdapterInterface] = None
    ) -> 'PrefilterStage':
        """
        创建并返回一个配置好的 PrefilterStage 实例。
        **参数**:
            settings (PrefilterStageSettings): **[必须]** Prefilter 阶段的配置 DTO。
            adapter (Optional[CzkawkaAdapterInterface]): **[可选]** CzkawkaAdapter 实例。
        **返回**:
            StageInterface: 一个实现了 `StageInterface` 的 `PrefilterStage` 实例。
        """
        self._logger.info('创建 PrefilterStage 实例...')
        if settings.tool == 'czkawka' and not settings.czkawka:
            self._logger.error(
                "Czkawka 配置缺失。无法创建 CzkawkaAdapter."
                )
            raise FactoryConfigurationError(message=
                "PrefilterStage 配置错误：工具设置为 'czkawka'，但 'czkawka' 配置块缺失。")
        czkawka_adapter_instance: CzkawkaAdapterInterface
        if settings.tool == 'czkawka':
            czkawka_adapter_instance = CzkawkaAdapter(settings=settings.
                czkawka, logger=self._logger.bind(component='CzkawkaAdapter'))
        else:
            self._logger.error(f'Unsupported prefilter tool: {settings.tool}')
            raise FactoryConfigurationError(message=
                f'不支持的预过滤工具: {settings.tool}')
        stage_instance = PrefilterStage(
            logger=self._logger.bind(stage_name='Prefilter'),
            settings=settings,
            adapter=czkawka_adapter_instance
        )
        self._logger.success('PrefilterStage 实例创建成功.')
        return stage_instance
