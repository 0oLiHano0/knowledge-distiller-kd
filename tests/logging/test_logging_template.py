# logging测试示例：
# 以/test_storage.py为例，测试时，需要创建一个假的LoggerProtocol对象，
# 然后将其传递给Storage对象，这样就可以测试Storage对象的逻辑了。
# 
from kd_tool.storage.memory import InMemoryStorage
from kd_tool.logging.protocols import LoggerProtocol

class _FakeLogger:
    def __getattr__(self, _):
        return lambda *args, **kwargs: None  # noqa: ANN001

def test_save_blocks():
    storage = InMemoryStorage(logger=_FakeLogger())  # 类型兼容
    storage.save_content_blocks(["hello"])
    assert storage.count() == 1
