```python

# kd_tool/stages/cleanup/real_file_system_adapter.py
# -*- coding: utf-8 -*-

"""
=================================================
real_file_system_adapter.py - 真实文件系统适配器实现 (v4.7)
=================================================

**模块功能**:

- 实现了 `FileSystemAdapterInterface`，使用 `pathlib` 和 `shutil` 执行真实的文件操作。
- **依赖**: Python 标准库。

---
"""

import shutil
from pathlib import Path

from .adapter_interface import FileSystemAdapterInterface
from .errors import FileOperationError

class RealFileSystemAdapter(FileSystemAdapterInterface):
    """
    使用 Python 标准库实现文件系统操作的适配器。
    """

    def move_file(self, source_path: Path, target_path: Path) -> None:
        """移动文件。"""
        try:
            if not source_path.is_file():
                raise FileNotFoundError(f"源文件不存在: {source_path}")
                
            self.ensure_directory_exists(target_path.parent)
            
            shutil.move(str(source_path), str(target_path))

        except Exception as e:
            raise FileOperationError(source_path, "move", e)

    def delete_file(self, file_path: Path) -> None:
        """永久删除文件。"""
        try:
            if file_path.is_file():
                 file_path.unlink()
            elif file_path.exists():
                 raise IsADirectoryError(f"路径是目录，非文件: {file_path}")
            # 如果文件不存在，可以考虑静默处理或记录警告
        except Exception as e:
            raise FileOperationError(file_path, "delete", e)

    def ensure_directory_exists(self, dir_path: Path) -> None:
        """确保目录存在。"""
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise FileOperationError(dir_path, "mkdir", e)

    def file_exists(self, file_path: Path) -> bool:
        """检查文件是否存在。"""
        return file_path.is_file()

```