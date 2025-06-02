"""
测试 storage 层 StorageInterface 的具体实现。
"""

import pytest

"""
测试 storage 层 StorageFactory 的架构符合性。

why: 确保 StorageFactory 遵循工厂模式和依赖注入原则，正确创建 StorageInterface 的实现实例。
what: 检查 StorageFactory 的 __init__ 是否接受依赖注入， create 方法是否存在并返回 StorageInterface 实例。(收集所有错误，一次性报告)
how: 使用 inspect 检查工厂类和方法属性，使用 InMemoryStorage 验证返回类型。收集所有 AssertionError 或 TypeError 并最后报告。
"""

import inspect
import pytest
import sys
from unittest.mock import Mock # 用于模拟依赖

from kd_tool.storage.storage_factory import StorageFactory
from kd_tool.storage.storage_interface import StorageInterface # 确保 StorageInterface 可导入
from tests.storage.in_memory_storage import InMemoryStorage # 导入 InMemoryStorage 工具


from kd_tool.core.core_dtos import PipelineContextDTO
from kd_tool.storage.settings_models import StorageBackend, StorageSettingsDTO
from kd_tool.storage.sqlite_storage import SQLiteStorage

def test_storage_factory_uses_di():
    """
    why: StorageFactory 必须通过构造函数注入依赖。
    what: 验证 StorageFactory 的 __init__ 方法接受多个参数（体现DI）。(收集错误)
    how: 使用 inspect 检查 __init__ 方法签名。收集 AssertionError。
    """
    errors = []

    from kd_tool.storage.storage_factory import StorageFactory # StorageFactory 已在上方导入
    
    # 确保 StorageFactory 已导入或可访问
    if 'StorageFactory' not in globals() and 'StorageFactory' not in locals():
         errors.append("请确保已导入 kd_tool.storage.storage_factory.StorageFactory")
         # 如果无法导入 StorageFactory，后续检查无意义，直接返回
         assert not errors, "\n".join(errors)
         return

    try:
        sig = inspect.signature(StorageFactory.__init__)
        # 至少应该注入 Logger 和 Settings，所以参数数量应大于1 (self)
        if len(sig.parameters) <= 1:
             errors.append("StorageFactory 构造函数参数过少，未体现DI")
    except Exception as e:
        errors.append(f"检查 StorageFactory.__init__ 签名时发生错误: {e}")

    assert not errors, "\n".join(errors)

def test_storage_factory_creates_storage_instance():
    """
    why: StorageFactory 的 create 方法必须返回一个 StorageInterface 的实例。
    what: 调用工厂的 create 方法，验证返回的对象是 StorageInterface 的实例。(收集错误)
    how: 使用 InMemoryStorage 模拟创建，并使用 isinstance 进行检查。收集 AssertionError 或 TypeError。
    """
    errors = []


    
    # 确保必要的类已导入或可访问
    if 'StorageFactory' not in globals() and 'StorageFactory' not in locals():
         errors.append("请确保已导入 kd_tool.storage.storage_factory.StorageFactory")
    if 'InMemoryStorage' not in globals() and 'InMemoryStorage' not in locals():
         errors.append("请确保已导入 tests.storage.in_memory_storage.InMemoryStorage")
    if 'StorageInterface' not in globals() and 'StorageInterface' not in locals() and 'kd_tool.storage.storage_interface' not in sys.modules:
         errors.append("请确保已导入 kd_tool.storage.storage_interface.StorageInterface")
    
    # 如果有导入错误，直接返回
    if errors:
        assert not errors, "\n".join(errors)
        return

    try:
        # 使用 Mock 对象模拟工厂的依赖 (Logger, Settings)
        mock_logger = Mock()
        # TODO: 根据实际 StorageSettingsDTO 结构模拟
        mock_settings = Mock()
        # 模拟 settings 对象的 backend 属性，使其返回一个 StorageBackend 值
        from kd_tool.storage.settings_models import StorageBackend # 需要导入 StorageBackend
        from pathlib import Path # 需要导入 Path
        mock_settings.backend = StorageBackend.SQLITE # 假设工厂会创建 SQLiteStorage
        mock_settings.db_path = Path("mock_db.sqlite") # 模拟 db_path
        mock_settings.echo_sql = False # 模拟 echo_sql
        
        # 实例化工厂，这可能抛出 TypeError
        factory = StorageFactory(mock_logger, mock_settings)
        
        
        instance = factory.create() 
        
        if not isinstance(instance, StorageInterface):
             errors.append("StorageFactory.create 未返回 StorageInterface 实例")
        # 进一步验证返回的是 InMemoryStorage 实例 (如果这是工厂当前唯一支持的实现)
        # if not isinstance(instance, InMemoryStorage):
        #      errors.append("StorageFactory.create 未返回 InMemoryStorage 实例")

    except TypeError as e:
        errors.append(f"实例化 StorageFactory 或调用 create 方法时发生 TypeError: {e}")
    except Exception as e:
        errors.append(f"调用 StorageFactory.create 方法时发生意外错误: {e}")

    # 确保 StorageBackend 已导入，否则上面的模拟会失败
    if 'StorageBackend' not in globals() and 'StorageBackend' not in locals() and 'kd_tool.storage.settings_models' not in sys.modules:
         errors.append("请确保已导入 kd_tool.storage.settings_models.StorageBackend")
    assert not errors, "\n".join(errors)

# TODO: 添加其他测试用例，例如测试工厂根据配置返回不同 Storage 实现的功能 (如果已实现)

# TODO: 添加其他测试用例，覆盖 StorageInterface 的所有方法 

@pytest.mark.parametrize("backend,expected_cls", [
    # 目前工厂仅支持 SQLITE
    (StorageBackend.SQLITE, SQLiteStorage),
])
def test_factory_returns_correct_storage_type(backend, expected_cls):
    """
    why: StorageFactory 应根据 backend 返回正确类型的 Storage 实例。
    what: 验证不同 backend 配置下，工厂产出的实例类型是否正确。
    how: 参数化传入 backend，断言 create() 返回的实例类型。
    """
    mock_logger = Mock()
    settings = StorageSettingsDTO(backend=backend, db_path=":memory:", echo_sql=False, backend_type="sqlite")
    factory = StorageFactory(mock_logger, settings)
    storage = factory.create()
    assert isinstance(storage, expected_cls)

def test_factory_injects_logger_and_settings():
    """
    why: StorageFactory 产出的 Storage 实例应持有 logger 和 settings。
    what: 检查 create() 返回的实例是否正确保存注入的依赖。
    how: 通过 getattr 检查 logger 和 settings 属性。
    """
    mock_logger = Mock()
    settings = StorageSettingsDTO(backend=StorageBackend.SQLITE, db_path=":memory:", echo_sql=False, backend_type="sqlite")
    factory = StorageFactory(mock_logger, settings)
    storage = factory.create()
    assert getattr(storage, "_logger", None) is mock_logger
    assert getattr(storage, "_settings", None) == settings

def test_settings_dto_invalid_backend_raises():
    """
    why: StorageSettingsDTO 应在非法 backend 时抛出校验异常。
    what: 直接实例化 DTO，断言抛出 ValidationError。
    how: 使用 pytest.raises 捕获异常。
    """
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        StorageSettingsDTO(backend="invalid", db_path=":memory:", echo_sql=False, backend_type="invalid") 