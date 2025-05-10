# knowledge_distiller_kd/prefilter/czkawka_adapter.py

import subprocess
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple
import platform
import time

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

        # 添加性能相关配置
        self.config = {
            "timeout": 300,
            "cache_ttl": 3600,
            "min_file_size": 1024,  # 忽略小于1KB的文件
            "max_file_size": 100 * 1024 * 1024,  # 忽略大于100MB的文件
            "czkawka_args": ["duplicates", "--json", "-d", "--minimal"],
            **(config or {})
        }
        self.logger = logger or logging.getLogger(__name__)

        # 添加扫描结果缓存
        self._scan_cache = {}
        self._cache_ttl = 3600  # 1小时缓存

        self._performance_metrics = {
            "scan_count": 0,
            "scan_time": 0,
            "filter_count": 0,
            "filter_time": 0
        }

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
        target_directory: Path,
        patterns: Optional[List[str]] = None
    ) -> List[DuplicateFileGroupDTO]:
        """
        调用 Czkawka CLI 扫描目录，解析并返回重复文件组 DTO 列表。
        
        Args:
            target_directory: 要扫描的目录
            patterns: 可选的文件类型过滤列表，如["*.md", "*.txt"]
        
        Returns:
            List[DuplicateFileGroupDTO]: 重复文件组列表
        """
        cache_key = str(target_directory)
        if cache_key in self._scan_cache:
            cache_time, result = self._scan_cache[cache_key]
            if time.time() - cache_time < self._cache_ttl:
                return result
                
        # 优化命令行参数
        args = self.config.get("czkawka_args", [
            "duplicates",
            "--json",
            "-d",
            "--minimal",  # 减少不必要的文件系统操作
        ])
        
        # 添加文件类型过滤
        if patterns:
            args.extend(["--type", ",".join(patterns)])
            
        cmd = [self.czkawka_cli_path] + args + [str(target_directory)]
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=self.config.get("timeout", 300),
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                parsed_result = self._parse_czkawka_json_to_dtos(data)
                # 更新缓存
                self._scan_cache[cache_key] = (time.time(), parsed_result)
                self._log_performance("scan", start_time)
                return parsed_result
            else:
                self.logger.error(
                    f"Czkawka 返回错误码 {result.returncode}: {result.stderr}"
                )
                return []
                
        except json.JSONDecodeError as e:
            self.logger.error(f"无法解析 Czkawka 输出的 JSON: {e}")
            return []
        except FileNotFoundError as e:
            self.logger.error(f"Czkawka 可执行文件未找到: {e}")
            return []
        except subprocess.TimeoutExpired as e:
            self.logger.error(f"Czkawka 执行超时: {e}")
            return []
        except PermissionError as e:
            self.logger.error(f"权限错误无法执行 Czkawka: {e}")
            return []
        except Exception as e:
            self.logger.error(f"执行 Czkawka 时发生意外错误: {e}")
            return []

    def filter_unique_files(
        self, 
        input_dir: Path, 
        extensions: List[str] = [".md", ".doc", ".docx"], 
        recursive: bool = True,
        max_depth: Optional[int] = None
    ) -> Tuple[List[Path], List[List[Path]]]:
        """
        扫描给定目录，过滤掉重复文件，只保留指定扩展名的文件。
        
        Args:
            input_dir: 要扫描的目录，例如 Path("input/")
            extensions: 需要处理的文件扩展名列表，默认为 [".md", ".doc", ".docx"]
            recursive: 是否递归扫描子目录，默认为 True
            max_depth: 递归的最大深度，None 表示无限制
            
        Returns:
            Tuple[List[Path], List[List[Path]]]: 
                - 唯一文件路径列表
                - 重复文件组列表，每组包含多个重复文件的路径
        """
        start_time = time.time()
        self.logger.info(f"开始扫描目录 {input_dir} 查找 {', '.join(extensions)} 文件...")
        
        # 1. 转换扩展名为 Czkawka 支持的模式
        patterns = [f"*{ext}" for ext in extensions]
        self.logger.debug(f"使用模式: {patterns}")
        
        # 2. 扫描重复文件
        duplicate_groups_dto = self.scan_directory_for_duplicates(input_dir, patterns)
        self.logger.debug(f"czkawka 发现 {len(duplicate_groups_dto)} 组重复文件")
        for idx, grp in enumerate(duplicate_groups_dto):
            self.logger.debug(f"重复组 #{idx+1}: {len(grp.files)} 个文件")
        
        # 3. 处理结果
        all_files = set()  # 所有匹配文件
        duplicate_files = set()  # 重复文件
        duplicate_groups = []  # 重复文件组
        
        # 收集所有文件和重复文件
        if recursive:
            for ext in extensions:
                glob_pattern = "**/*" + ext
                if max_depth is not None:
                    # 通过手动实现深度限制来扫描
                    for depth in range(1, max_depth + 2):  # +2 因为根目录算一层
                        pattern = "/".join(["*"] * (depth - 1)) + ("/*" + ext if depth > 1 else "*" + ext)
                        found = list(input_dir.glob(pattern))
                        self.logger.debug(f"深度 {depth}, 模式 {pattern}: 找到 {len(found)} 文件")
                        all_files.update(found)
                else:
                    found = list(input_dir.glob(glob_pattern))
                    self.logger.debug(f"模式 {glob_pattern}: 找到 {len(found)} 文件")
                    all_files.update(found)
        else:
            # 不递归，只扫描顶层目录
            for ext in extensions:
                found = list(input_dir.glob("*" + ext))
                self.logger.debug(f"模式 *{ext}: 找到 {len(found)} 文件")
                all_files.update(found)
        
        self.logger.debug(f"总共找到 {len(all_files)} 个匹配扩展名的文件")
        
        # 从DTO转换为路径列表
        for group in duplicate_groups_dto:
            if len(group.files) < 2:
                continue  # 跳过不是真正重复的组
                
            duplicate_group = []
            for file_info in group.files:
                path = Path(file_info.path)
                duplicate_files.add(path)
                duplicate_group.append(path)
            
            if duplicate_group:
                duplicate_groups.append(duplicate_group)
        
        self.logger.debug(f"重复文件组: {len(duplicate_groups)}, 重复文件总数: {len(duplicate_files)}")
        
        # 检查路径是否在 all_files 中
        for dup_file in duplicate_files:
            if dup_file not in all_files:
                self.logger.warning(f"重复文件 {dup_file} 不在扫描到的文件列表中")
        
        # 找出唯一文件 (所有文件中排除重复文件)
        unique_files = sorted(list(all_files - duplicate_files))
        
        self._log_performance("filter", start_time)
        self.logger.info(f"扫描完成: 发现 {len(unique_files)} 个唯一文件，{len(duplicate_groups)} 组重复文件。")
        
        return unique_files, duplicate_groups

    def _log_performance(self, operation: str, start_time: float):
        duration = time.time() - start_time
        self._performance_metrics[f"{operation}_count"] += 1
        self._performance_metrics[f"{operation}_time"] += duration
        self.logger.debug(f"{operation} 耗时: {duration:.2f}秒")
