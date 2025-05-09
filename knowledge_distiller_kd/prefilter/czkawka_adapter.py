import subprocess
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from knowledge_distiller_kd.core.models import (
    DuplicateFileInfoDTO,
    DuplicateFileGroupDTO,
)

class CzkawkaAdapter:
    def __init__(
        self,
        czkawka_cli_path: str = "czkawka_cli",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.czkawka_cli_path = czkawka_cli_path
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

    def scan_directory_for_duplicates(self, target_directory: Path) -> List[DuplicateFileGroupDTO]:
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
                timeout=300,
            )
        except FileNotFoundError as e:
            self.logger.error(f"Czkawka 可执行文件未找到: {e}")
            return []

        if result.returncode != 0:
            self.logger.error(f"Czkawka 返回错误码 {result.returncode}: {result.stderr}")
            return []

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            self.logger.error(f"无法解析 Czkawka 输出的 JSON: {e}")
            return []

        return self._parse_czkawka_json_to_dtos(data)
