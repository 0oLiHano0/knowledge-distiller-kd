# knowledge_distiller_kd/prefilter/czkawka_adapter.py

import subprocess
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
import platform

from knowledge_distiller_kd.core.utils import get_bundled_czkawka_path
from knowledge_distiller_kd.core.models import (
    DuplicateFileInfoDTO,
    DuplicateFileGroupDTO,
)


class CzkawkaAdapter:
    def __init__(
        self,
        czkawka_cli_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        # 如果用户指定了路径则使用，否则加载捆绑在 vendor 下的二进制
        if czkawka_cli_path:
            self.czkawka_cli_path = czkawka_cli_path
        else:
            self.czkawka_cli_path = get_bundled_czkawka_path()

        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

    def _build_command(self, target_directory: Path) -> List[str]:
        """
        构造 czkawka_cli 命令行列表。
        """
        args = self.config.get("czkawka_args", ["duplicates", "--json", "-d"])
        return [self.czkawka_cli_path] + args + [str(target_directory)]

    def _parse_czkawka_json_to_dtos(
        self,
        json_data: Any,
    ) -> List[DuplicateFileGroupDTO]:
        """
        将 Czkawka JSON 输出结构转换为 DTO 列表。
        假定 json_data 是一个列表，每个元素包含 "files" 字段。
        """
        groups: List[DuplicateFileGroupDTO] = []
        for group in json_data:
            files_info = []
            for f in group.get("files", []):
                try:
                    dto = DuplicateFileInfoDTO(
                        path=f["path"],
                        size=int(f["size"]),
                        modified=f.get("modified"),
                    )
                    files_info.append(dto)
                except Exception as e:
                    self.logger.error(f"解析文件信息失败: {e}")
            if files_info:
                groups.append(DuplicateFileGroupDTO(files=files_info))
        return groups

    def scan_directory_for_duplicates(
        self,
        target_directory: Path
    ) -> List[DuplicateFileGroupDTO]:
        """
        调用 Czkawka CLI 扫描目录，解析并返回重复文件组 DTO 列表。
        """
        cmd = self._build_command(target_directory)
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=self.config.get("timeout", 300),
            )
        except FileNotFoundError as e:
            self.logger.error(f"Czkawka 可执行文件未找到: {e}")
            return []
        except subprocess.TimeoutExpired as e:
            self.logger.error(f"Czkawka 执行超时: {e}")
            return []
        except PermissionError as e:
            # 添加对权限错误的处理
            self.logger.error(f"权限错误无法执行 Czkawka: {e}")
            return []
        except Exception as e:
            # 添加通用异常处理，确保健壮性
            self.logger.error(f"执行 Czkawka 时发生意外错误: {e}")
            return []

        if result.returncode != 0:
            self.logger.error(
                f"Czkawka 返回错误码 {result.returncode}: {result.stderr}"
            )
            return []

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            self.logger.error(f"无法解析 Czkawka 输出的 JSON: {e}")
            return []

        return self._parse_czkawka_json_to_dtos(data)

    def filter_unique_files(self, target_directory: Path, patterns: List[str] = None) -> List[Path]:
        """
        扫描目录，识别重复文件组，然后返回最终要解析的文件列表：
        - 所有不在任何重复组里的文件原样保留
        - 每个重复组只保留第一个文件（可以根据需要调整选择策略）
        
        Args:
            target_directory: 要扫描的目录
            patterns: 文件匹配模式列表，默认为 ["*.md", "*.markdown"]
            
        Returns:
            List[Path]: 过滤后的文件路径列表
        """
        # 使用默认模式，如果未提供
        if patterns is None:
            patterns = ["*.md", "*.markdown"]
        
        # 1. 用 Czkawka 找到所有重复组
        groups = self.scan_directory_for_duplicates(target_directory)
        
        # 2. 收集所有重复文件路径和每组要保留的代表文件
        files_in_groups: Set[Path] = set()
        representatives: Set[Path] = set()
        
        for grp in groups:
            # 转换所有文件路径到Path对象
            paths = [Path(f.path) for f in grp.files]
            files_in_groups.update(paths)
            
            # 选择每组的第一个文件作为代表
            if paths:
                representatives.add(paths[0])
        
        # 3. 列出目录下所有匹配的文件
        all_files: Set[Path] = set()
        for pattern in patterns:
            all_files.update(target_directory.rglob(pattern))
        
        # 4. 过滤：保留不在任何重复组的文件和每组的代表文件
        unique_files = [p for p in all_files if (p not in files_in_groups or p in representatives)]
        
        # 5. 返回排序后的列表，确保结果确定性
        return sorted(unique_files)
