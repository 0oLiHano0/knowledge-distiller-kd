"""
=================================================
adapter_interface.py - 文件系统适配器接口定义 (v4.7)
=================================================

**模块功能**:

- 定义文件系统操作（移动、删除）的抽象接口。
- **规范**: 使得 CleanupStage 可以测试，而无需实际操作磁盘。

---
"""
from abc import ABC, abstractmethod
from pathlib import Path


class FileSystemAdapterInterface(ABC):
    """
    文件系统操作适配器的抽象基类。
    """

    @abstractmethod
    def move_file(self, source_path: Path, target_path: Path) -> None:
        """
        移动文件从源路径到目标路径。
        **规范**: 
        - 必须处理源文件不存在、目标已存在、权限不足等情况。
        - 必须确保目标目录存在。
        - **必须**抛出 `FileOperationError` 如果失败。
        """
        pass

    @abstractmethod
    def delete_file(self, file_path: Path) -> None:
        """
        永久删除指定路径的文件。
        **规范**: 
        - 必须处理文件不存在、权限不足等情况。
        - **警告**: 这是破坏性操作。
        - **必须**抛出 `FileOperationError` 如果失败。
        """
        pass

    @abstractmethod
    def ensure_directory_exists(self, dir_path: Path) -> None:
        """
        确保指定的目录存在，如果不存在则创建它。
        **规范**: 
        - 必须处理创建失败（如权限不足）的情况。
        - **必须**抛出 `FileOperationError` 或类似错误如果失败。
        """
        pass

    @abstractmethod
    def file_exists(self, file_path: Path) -> bool:
        """
        检查文件是否存在。
        """
        pass
