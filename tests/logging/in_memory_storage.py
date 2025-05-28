from kd_tool.storage.storage_interface import StorageInterface

class InMemoryStorage(StorageInterface):
    def __init__(self, logger=None):
        self._logger = logger
        self._blocks = []
    def save_content_blocks(self, blocks):
        self._blocks.extend(blocks)
    def count(self):
        return len(self._blocks) 