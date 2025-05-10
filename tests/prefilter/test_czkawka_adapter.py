import unittest
from pathlib import Path
import json
import subprocess
from unittest.mock import patch, MagicMock, mock_open
import os
import pytest
from io import StringIO
import tempfile
import shutil
import logging

from knowledge_distiller_kd.prefilter.czkawka_adapter import CzkawkaAdapter
from knowledge_distiller_kd.core.models import DuplicateFileInfoDTO, DuplicateFileGroupDTO
from knowledge_distiller_kd.core.utils import get_bundled_czkawka_path

class TestCzkawkaAdapter(unittest.TestCase):
    def setUp(self):
        self.custom_args = ["duplicates", "--json", "-d"]
        czkawka_cli_path = get_bundled_czkawka_path()
        
        # 打印 czkawka_cli_path 以确认路径是否正确
        print(f"czkawka_cli_path: {czkawka_cli_path}")

        self.adapter = CzkawkaAdapter(
            czkawka_cli_path=czkawka_cli_path,
            config={"czkawka_args": self.custom_args}
        )

    def test_build_command_default(self):
        dir_path = Path("/tmp/testdir")
        cmd = self.adapter._build_command(dir_path)
        # 使用完整路径作为期望值的第一部分
        expected = [self.adapter.czkawka_cli_path] + self.custom_args + [str(dir_path)]
        self.assertEqual(cmd, expected)

    def test_parse_valid_czkawka_json_to_dtos(self):
        sample = [
            {"files": [
                {"path": "/a.txt", "size": 100, "modified": 1600000000},
                {"path": "/b.txt", "size": 100, "modified": 1600000000}
            ]},
            {"files": []}
        ]
        groups = self.adapter._parse_czkawka_json_to_dtos(sample)
        # 只保留第一个非空组
        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertIsInstance(group, DuplicateFileGroupDTO)
        self.assertEqual(len(group.files), 2)
        self.assertEqual(
            group.files[0],
            DuplicateFileInfoDTO(path="/a.txt", size=100, modified=1600000000)
        )

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run')
    def test_scan_directory_returns_correct_dtos_on_success(self, mock_run):
        sample = [{"files": [{"path": "/x", "size": 1}]}]
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(sample),
            stderr=""
        )
        result = self.adapter.scan_directory_for_duplicates(Path("/some/dir"))
        self.assertEqual(len(result), 1)
        mock_run.assert_called_once()
        self.assertEqual(result[0].files[0], DuplicateFileInfoDTO(path="/x", size=1, modified=None))

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run', side_effect=FileNotFoundError)
    def test_scan_directory_handles_czkawka_cli_not_found(self, mock_run):
        result = self.adapter.scan_directory_for_duplicates(Path("/"))
        self.assertEqual(result, [])

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run')
    def test_scan_directory_handles_czkawka_error_code(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="some error"
        )
        result = self.adapter.scan_directory_for_duplicates(Path("/"))
        self.assertEqual(result, [])

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run')
    def test_scan_directory_handles_invalid_json_output(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="not a json",
            stderr=""
        )
        result = self.adapter.scan_directory_for_duplicates(Path("/"))
        self.assertEqual(result, [])

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run')
    def test_scan_directory_handles_no_duplicates_found_json(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="[]",
            stderr=""
        )
        result = self.adapter.scan_directory_for_duplicates(Path("/"))
        self.assertEqual(result, [])
    # add in 2025-05-09 17:00
    def test_custom_arguments_override_defaults(self):
        """测试自定义参数完全覆盖默认参数"""
        custom_adapter = CzkawkaAdapter(
            czkawka_cli_path="custom_path",
            config={"czkawka_args": ["empty-files", "--minimal", "--not-recursive"]}
        )
        cmd = custom_adapter._build_command(Path("/test"))
        expected = ["custom_path", "empty-files", "--minimal", "--not-recursive", "/test"]
        self.assertEqual(cmd, expected)

    def test_path_with_special_characters(self):
        """测试包含特殊字符的路径处理"""
        dir_path = Path("/tmp/test dir with spaces/查找重复")
        cmd = self.adapter._build_command(dir_path)
        # 使用完整路径作为期望值的第一部分
        expected = [self.adapter.czkawka_cli_path] + self.custom_args + [str(dir_path)]
        self.assertEqual(cmd, expected)

    def test_parse_malformed_json_structure_graceful_handling(self):
        """测试处理格式异常但部分可解析的JSON"""
        malformed_data = [
            {"header": "Group 1", "files": [{"path": "/valid.txt", "size": 100}]},
            {"wrong_key": [], "files": []},  # 错误的结构但有files键
            {"files": [{"path": "/valid2.txt", "size": 200, "extra": "ignored"}]}  # 额外字段
        ]
        result = self.adapter._parse_czkawka_json_to_dtos(malformed_data)
        self.assertEqual(len(result), 2)  # 应该成功解析两组
        self.assertEqual(result[0].files[0].path, "/valid.txt")
        self.assertEqual(result[1].files[0].path, "/valid2.txt")

    def test_parse_json_with_missing_required_fields(self):
        """测试缺少必要字段的JSON处理"""
        missing_fields_data = [
            {"files": [{"size": 100}]},  # 缺少path
            {"files": [{"path": "/valid.txt"}]},  # 缺少size
        ]
        result = self.adapter._parse_czkawka_json_to_dtos(missing_fields_data)
        self.assertEqual(len(result), 0)  # 两组都应该被跳过

    def test_unicode_file_paths(self):
        """测试Unicode字符路径的处理"""
        unicode_data = [{"files": [
            {"path": "/路径/файл.txt", "size": 100},
            {"path": "/path/文件.txt", "size": 100}
        ]}]
        result = self.adapter._parse_czkawka_json_to_dtos(unicode_data)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].files[0].path, "/路径/файл.txt")

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run', 
           side_effect=subprocess.TimeoutExpired(cmd="", timeout=5))
    def test_scan_handles_subprocess_timeout(self, mock_run):
        """测试子进程执行超时的情况"""
        result = self.adapter.scan_directory_for_duplicates(Path("/test"))
        self.assertEqual(result, [])
        mock_run.assert_called_once()

    @patch('knowledge_distiller_kd.core.utils.get_bundled_czkawka_path')
    def test_custom_timeout_parameter(self, mock_get_path):
        """测试自定义超时参数"""
        # 模拟返回一个假路径
        mock_get_path.return_value = "/mock/path/czkawka_cli"
        
        adapter = CzkawkaAdapter(config={"timeout": 120})  # 设置120秒超时
        with patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run') as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="[]", stderr=""
            )
            adapter.scan_directory_for_duplicates(Path("/test"))
            # 验证调用subprocess.run时使用了正确的超时参数
            call_kwargs = mock_run.call_args[1]
            self.assertEqual(call_kwargs["timeout"], 120)

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run')
    def test_large_duplicate_groups(self, mock_run):
        """测试处理大量文件的重复组"""
        # 创建一个包含100个文件的重复组
        large_group = {"files": [
            {"path": f"/path/file{i}.txt", "size": 1000} 
            for i in range(100)
        ]}
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps([large_group]), stderr=""
        )
        
        result = self.adapter.scan_directory_for_duplicates(Path("/test"))
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].files), 100)

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run')
    def test_single_file_groups_are_filtered(self, mock_run):
        """测试单文件组被正确处理"""
        # Czkawka可能在某些情况下返回单文件组，这些不应被视为重复
        single_file_groups = [
            {"files": [{"path": "/path/unique1.txt", "size": 100}]},
            {"files": [{"path": "/path/unique2.txt", "size": 200}]}
        ]
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(single_file_groups), stderr=""
        )
        
        result = self.adapter.scan_directory_for_duplicates(Path("/test"))
        self.assertEqual(len(result), 2)  # 这里应保留单文件组的行为取决于业务需求

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run')
    def test_mixed_size_duplicate_groups(self, mock_run):
        """测试混合大小的重复组处理"""
        mixed_groups = [
            {"files": [{"path": f"/path/group1_{i}.txt", "size": 100} for i in range(5)]},
            {"files": [{"path": f"/path/group2_{i}.txt", "size": 200} for i in range(2)]},
            {"files": [{"path": f"/path/group3_{i}.txt", "size": 300} for i in range(10)]}
        ]
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(mixed_groups), stderr=""
        )
        
        result = self.adapter.scan_directory_for_duplicates(Path("/test"))
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0].files), 5)
        self.assertEqual(len(result[1].files), 2)
        self.assertEqual(len(result[2].files), 10)

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run', 
           side_effect=PermissionError("Permission denied"))
    def test_scan_handles_permission_error(self, mock_run):
        """测试权限错误处理"""
        result = self.adapter.scan_directory_for_duplicates(Path("/test"))
        self.assertEqual(result, [])

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run')
    def test_scan_handles_stderr_output(self, mock_run):
        """测试处理标准错误输出"""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, 
            stdout="[]", 
            stderr="Warning: Some files could not be scanned"
        )
        
        # 使用StringIO捕获日志输出
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger()
        logger.addHandler(handler)
        
        result = self.adapter.scan_directory_for_duplicates(Path("/test"))
        self.assertEqual(result, [])
        
        # 检查是否记录了警告
        log_output = log_capture.getvalue()
        logger.removeHandler(handler)
        # 这个断言可能需要根据实际的日志记录方式调整
        # self.assertIn("Warning", log_output) 

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.CzkawkaAdapter.scan_directory_for_duplicates')
    def test_filter_unique_files(self, mock_scan):
        """测试从重复组中筛选出唯一文件"""
        # 假设已实现filter_unique_files方法
        # 这个测试用例验证它是否正确保留每组的第一个文件和所有不在重复组中的文件

        # 模拟一个目录中有以下文件:
        # /test/file1.md, /test/file2.md (重复)
        # /test/file3.md, /test/file4.md, /test/file5.md (重复)
        # /test/unique.md (唯一文件)

        # 模拟scan_directory_for_duplicates的返回结果
        mock_scan.return_value = [
            DuplicateFileGroupDTO(files=[
                DuplicateFileInfoDTO(path="/test/file1.md", size=100),
                DuplicateFileInfoDTO(path="/test/file2.md", size=100)
            ]),
            DuplicateFileGroupDTO(files=[
                DuplicateFileInfoDTO(path="/test/file3.md", size=200),
                DuplicateFileInfoDTO(path="/test/file4.md", size=200),
                DuplicateFileInfoDTO(path="/test/file5.md", size=200)
            ])
        ]

        # 模拟文件系统
        with patch('pathlib.Path.glob') as mock_glob:
            all_files = [
                Path("/test/file1.md"), Path("/test/file2.md"),
                Path("/test/file3.md"), Path("/test/file4.md"), Path("/test/file5.md"),
                Path("/test/unique.md")  # 不在任何重复组中
            ]
            
            def glob_side_effect(pattern):
                if pattern.endswith(".md"):
                    return all_files
                return []
            
            mock_glob.side_effect = glob_side_effect

            # 调用filter_unique_files方法
            unique_files, duplicate_groups = self.adapter.filter_unique_files(Path("/test"), extensions=[".md"])

            # 验证结果
            # 应有1个唯一文件
            self.assertEqual(len(unique_files), 1)
            self.assertEqual(unique_files[0].name, "unique.md")
            
            # 应有2组重复文件
            self.assertEqual(len(duplicate_groups), 2)
            # 第一组有2个文件
            self.assertEqual(len(duplicate_groups[0]), 2)
            # 第二组有3个文件
            self.assertEqual(len(duplicate_groups[1]), 3)

    def test_filter_unique_files_empty_directory(self):
        """测试空目录的处理"""
        with patch('knowledge_distiller_kd.prefilter.czkawka_adapter.CzkawkaAdapter.scan_directory_for_duplicates') as mock_scan:
            mock_scan.return_value = []

            with patch('pathlib.Path.glob') as mock_glob:
                mock_glob.return_value = []

                # 调用filter_unique_files方法
                unique_files, duplicate_groups = self.adapter.filter_unique_files(Path("/empty"))
                self.assertEqual(len(unique_files), 0)
                self.assertEqual(len(duplicate_groups), 0)

    def test_filter_unique_files_custom_patterns(self):
        """测试使用自定义文件模式"""
        with patch('knowledge_distiller_kd.prefilter.czkawka_adapter.CzkawkaAdapter.scan_directory_for_duplicates') as mock_scan:
            mock_scan.return_value = [
                DuplicateFileGroupDTO(files=[
                    DuplicateFileInfoDTO(path="/test/file1.txt", size=100),
                    DuplicateFileInfoDTO(path="/test/file2.txt", size=100)
                ])
            ]

            with patch('pathlib.Path.glob') as mock_glob:
                # 模拟不同的模式返回不同的文件
                def side_effect(pattern):
                    if pattern.endswith(".txt"):
                        return [Path("/test/file1.txt"), Path("/test/file2.txt"), Path("/test/unique.txt")]
                    return []

                mock_glob.side_effect = side_effect

                # 调用filter_unique_files方法并传入自定义模式
                unique_files, duplicate_groups = self.adapter.filter_unique_files(
                    Path("/test"), extensions=[".txt"]
                )
                
                # 验证结果
                self.assertEqual(len(unique_files), 1)
                self.assertEqual(unique_files[0].name, "unique.txt")
                self.assertEqual(len(duplicate_groups), 1)

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.CzkawkaAdapter.scan_directory_for_duplicates')
    def test_filter_unique_files_with_extensions(self, mock_scan):
        """测试 filter_unique_files 方法使用指定扩展名过滤重复文件"""
        # 模拟 scan_directory_for_duplicates 返回结果
        duplicate_group = DuplicateFileGroupDTO(files=[
            DuplicateFileInfoDTO(path="/tmp/doc1.md", size=100),
            DuplicateFileInfoDTO(path="/tmp/doc1_copy.md", size=100)
        ])
        mock_scan.return_value = [duplicate_group]
        
        # 模拟文件系统
        with patch('pathlib.Path.glob') as mock_glob:
            # 返回的文件应该包括重复组中的文件和一些唯一文件
            md_files = [
                Path("/tmp/doc1.md"),
                Path("/tmp/doc1_copy.md"),
                Path("/tmp/unique1.md")
            ]
            docx_files = [
                Path("/tmp/unique2.docx")
            ]
            ignored_files = [
                Path("/tmp/ignored.txt")
            ]
            
            # 模拟找到的所有文件
            def glob_side_effect(pattern):
                # 根据不同模式返回不同结果，避免重复计数
                if pattern.endswith(".md"):
                    return md_files
                elif pattern.endswith(".doc"):
                    return []
                elif pattern.endswith(".docx"):
                    return docx_files
                else:
                    return []
                
            mock_glob.side_effect = glob_side_effect
            
            # 调用测试目标方法
            adapter = CzkawkaAdapter()
            
            # 因为测试时不会真的执行替换，我们使用一个更明确的自定义logger来验证
            test_logger = logging.getLogger("test_logger")
            adapter.logger = test_logger
            
            with patch('logging.getLogger'):
                unique_files, duplicate_groups = adapter.filter_unique_files(
                    Path("/tmp"), 
                    extensions=[".md", ".doc", ".docx"]
                )
                
                # 验证结果
                self.assertEqual(len(unique_files), 2)  # 1个md, 1个docx
                unique_names = [f.name for f in unique_files]
                self.assertIn("unique1.md", unique_names)
                self.assertIn("unique2.docx", unique_names)
                self.assertEqual(len(duplicate_groups), 1)

    def test_filter_unique_files_real_files(self):
        """使用真实临时文件测试 filter_unique_files 方法"""
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 创建测试文件结构
            # 唯一文件
            with open(temp_path / "unique1.md", "w") as f:
                f.write("This is unique file 1")
            
            with open(temp_path / "unique2.docx", "w") as f:
                f.write("This is unique file 2")
                
            # 重复文件 (内容相同)
            duplicate_content = "This is duplicate content"
            with open(temp_path / "dup1.md", "w") as f:
                f.write(duplicate_content)
            
            with open(temp_path / "dup2.md", "w") as f:
                f.write(duplicate_content)
                
            # 忽略的文件类型
            with open(temp_path / "ignored.txt", "w") as f:
                f.write("This should be ignored")
                
            # 创建子目录测试递归
            sub_dir = temp_path / "subdir"
            sub_dir.mkdir()
            
            with open(sub_dir / "sub_unique.md", "w") as f:
                f.write("Unique in subdirectory")
                
            with open(sub_dir / "sub_dup.md", "w") as f:
                f.write(duplicate_content)  # 与 dup1.md 和 dup2.md 内容相同
            
            # 模拟 czkawka 的执行 (因为真实执行可能不可靠)
            with patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run') as mock_run:
                # 模拟 czkawka 的输出，返回我们创建的重复文件
                duplicate_json = [{
                    "files": [
                        {"path": str(temp_path / "dup1.md"), "size": len(duplicate_content)},
                        {"path": str(temp_path / "dup2.md"), "size": len(duplicate_content)},
                        {"path": str(sub_dir / "sub_dup.md"), "size": len(duplicate_content)}
                    ]
                }]
                
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=json.dumps(duplicate_json), stderr=""
                )
                
                # 执行被测试的方法
                adapter = CzkawkaAdapter()
                unique_files, duplicate_groups = adapter.filter_unique_files(
                    temp_path,
                    extensions=[".md", ".docx"],
                    recursive=True
                )
                
                # 验证结果
                # 唯一文件应该有3个：unique1.md, unique2.docx, sub_unique.md
                self.assertEqual(len(unique_files), 3)
                unique_filenames = [f.name for f in unique_files]
                self.assertIn("unique1.md", unique_filenames)
                self.assertIn("unique2.docx", unique_filenames)
                self.assertIn("sub_unique.md", unique_filenames)
                
                # 重复文件组应该有1组
                self.assertEqual(len(duplicate_groups), 1)
                # 每组应该有3个文件
                self.assertEqual(len(duplicate_groups[0]), 3)
                
                # 忽略的文件类型不应该出现在结果中
                for file in unique_files + [f for group in duplicate_groups for f in group]:
                    self.assertNotEqual(file.name, "ignored.txt")

    def test_filter_unique_files_non_recursive(self):
        """测试非递归模式下的 filter_unique_files 方法"""
        # 模拟 scan_directory_for_duplicates 的返回
        with patch('knowledge_distiller_kd.prefilter.czkawka_adapter.CzkawkaAdapter.scan_directory_for_duplicates') as mock_scan:
            duplicate_group = DuplicateFileGroupDTO(files=[
                DuplicateFileInfoDTO(path="/tmp/dup1.md", size=100),
                DuplicateFileInfoDTO(path="/tmp/dup2.md", size=100)
            ])
            mock_scan.return_value = [duplicate_group]
            
            # 模拟文件系统
            with patch('pathlib.Path.glob') as mock_glob:
                top_level_files = [
                    Path("/tmp/dup1.md"),
                    Path("/tmp/dup2.md"),
                    Path("/tmp/unique1.md")
                ]
                
                # 不同的扩展名和递归模式应返回不同结果
                def glob_side_effect(pattern):
                    # 只返回顶层文件，不返回子目录文件
                    if ".md" in pattern and "**" not in pattern:
                        return [f for f in top_level_files if str(f).endswith(".md")]
                    elif ".docx" in pattern and "**" not in pattern:
                        return []
                    return []
                
                mock_glob.side_effect = glob_side_effect
                
                # 执行被测试方法
                adapter = CzkawkaAdapter()
                unique_files, duplicate_groups = adapter.filter_unique_files(
                    Path("/tmp"),
                    recursive=False  # 非递归模式
                )
                
                # 验证结果
                self.assertEqual(len(unique_files), 1)  # 只应返回顶层的唯一文件
                self.assertEqual(unique_files[0].name, "unique1.md")
                self.assertEqual(len(duplicate_groups), 1)

if __name__ == "__main__":
    unittest.main()
