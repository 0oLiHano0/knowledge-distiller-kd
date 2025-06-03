from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any


class ParserAdapterInterface(ABC):
    """
    WHY: 统一文档解析器接口，便于扩展和Mock
    WHAT: 解析文件为原始元素列表
    HOW: 由具体Adapter实现
    """

    @abstractmethod
    def parse_file_to_raw_elements(self, file_path: Path) -> List[Dict[str, Any]]:
        """解析文件为原始元素列表。"""
        pass
