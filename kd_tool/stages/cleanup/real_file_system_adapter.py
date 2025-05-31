# kd_tool/stages/cleanup/real_file_system_adapter.py
"""
=================================================
real_file_system_adapter.py - 文件系统适配器实现 (v4.6)
=================================================

**模块功能**:

- 实现了 `FileSystemAdapterInterface`，使用 `pathlib` 和 `shutil` 执行真实的文件操作。
- **依赖**: Python 标准库。

---
"""
import os
import shutil
from pathlib import Path
from kd_tool.logging.protocols import LoggerProtocol # 导入 LoggerProtocol
from kd_tool.stages.cleanup.adapter_interface import FileSystemAdapterInterface
from kd_tool.stages.cleanup.errors import FileOperationError


class RealFileSystemAdapter(FileSystemAdapterInterface):
    """
    使用 Python 标准库实现文件系统操作的适配器。
    """

    def __init__(self, logger: LoggerProtocol): # 添加 logger 参数
        """
        初始化 RealFileSystemAdapter。

        参数:
            logger (LoggerProtocol): 用于记录日志的记录器实例。
        """
        self._logger = logger.bind(component=self.__class__.__name__) # 存储并绑定组件名

    def move_file(self, source_path: Path, target_path: Path) ->None:
        """移动文件。"""
        self._logger.debug(f"尝试移动文件从 '{source_path}' 到 '{target_path}'...") # 添加日志
        try:
            if not source_path.is_file():
                # 在抛出错误前记录
                self._logger.error(f"源文件不存在: {source_path}")
                raise FileNotFoundError(f'源文件不存在: {source_path}')
            self.ensure_directory_exists(target_path.parent) # ensure_directory_exists 内部会有自己的日志
            shutil.move(str(source_path), str(target_path))
            self._logger.info(f"成功移动文件从 '{source_path}' 到 '{target_path}'.") # 成功日志
        except Exception as e:
            self._logger.exception(f"移动文件 '{source_path}' 到 '{target_path}' 时发生错误.") # 异常日志
            raise FileOperationError(source_path, 'move', e)

    def delete_file(self, file_path: Path) ->None:
        """永久删除文件。"""
        self._logger.debug(f"尝试永久删除文件: '{file_path}'...") 
        try:
            if file_path.is_file():
                file_path.unlink()
                self._logger.info(f"成功永久删除文件: '{file_path}'.") 
            elif file_path.exists():
                self._logger.error(f"尝试删除的路径是一个目录，而非文件: {file_path}")
                raise IsADirectoryError(f'路径是目录，非文件: {file_path}')
            else:
                self._logger.warning(f"尝试删除的文件不存在: '{file_path}'. 操作跳过。") # (警告)
        except Exception as e:
            self._logger.exception(f"永久删除文件 '{file_path}' 时发生错误.") # 修改点
            raise FileOperationError(file_path, 'delete', e)

    def ensure_directory_exists(self, dir_path: Path) ->None:
        """确保目录存在。"""
        if dir_path.exists() and dir_path.is_dir():
            # self._logger.trace(f"目录已存在: '{dir_path}'.") # 如果需要更详细的跟踪日志可以取消注释
            return

        self._logger.debug(f"尝试创建目录 (如果不存在): '{dir_path}'...") 
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"成功确保目录存在: '{dir_path}'.") 
        except Exception as e:
            self._logger.exception(f"创建目录 '{dir_path}' 时发生错误.") 
            raise FileOperationError(dir_path, 'mkdir', e)

    def file_exists(self, file_path: Path) ->bool:
        """检查文件是否存在。"""
        # 此方法通常很简单，可能不需要显式日志记录，除非用于调试跟踪
        # self._logger.trace(f"检查文件是否存在: '{file_path}'")
        return file_path.is_file()