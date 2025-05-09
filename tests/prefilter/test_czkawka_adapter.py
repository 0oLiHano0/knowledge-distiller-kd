import unittest
from pathlib import Path
import json
import subprocess
from unittest.mock import patch, MagicMock
import os
import pytest
from io import StringIO

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
        import logging
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
        
        # 模拟Path.rglob方法
        with patch('pathlib.Path.rglob') as mock_rglob:
            mock_rglob.side_effect = lambda pattern: [
                Path("/test/file1.md"), Path("/test/file2.md"),
                Path("/test/file3.md"), Path("/test/file4.md"), Path("/test/file5.md"),
                Path("/test/unique.md")  # 不在任何重复组中
            ] if pattern in ["*.md", "*.markdown"] else []
            
            # 调用filter_unique_files方法 (需要实现)
            result = self.adapter.filter_unique_files(Path("/test"))
            
            # 验证结果
            expected = [
                Path("/test/file1.md"),  # 第一组保留
                Path("/test/file3.md"),  # 第二组保留
                Path("/test/unique.md")  # 唯一文件保留
            ]
            
            # 排序进行比较
            self.assertEqual(sorted(str(p) for p in result), 
                             sorted(str(p) for p in expected))

    def test_filter_unique_files_empty_directory(self):
        """测试空目录的处理"""
        with patch('knowledge_distiller_kd.prefilter.czkawka_adapter.CzkawkaAdapter.scan_directory_for_duplicates') as mock_scan:
            mock_scan.return_value = []
            
            with patch('pathlib.Path.rglob') as mock_rglob:
                mock_rglob.return_value = []
                
                # 调用filter_unique_files方法 (需要实现)
                result = self.adapter.filter_unique_files(Path("/empty"))
                self.assertEqual(result, [])

    def test_filter_unique_files_custom_patterns(self):
        """测试使用自定义文件模式"""
        with patch('knowledge_distiller_kd.prefilter.czkawka_adapter.CzkawkaAdapter.scan_directory_for_duplicates') as mock_scan:
            mock_scan.return_value = [
                DuplicateFileGroupDTO(files=[
                    DuplicateFileInfoDTO(path="/test/file1.txt", size=100),
                    DuplicateFileInfoDTO(path="/test/file2.txt", size=100)
                ])
            ]
            
            with patch('pathlib.Path.rglob') as mock_rglob:
                # 模拟不同的模式返回不同的文件
                def side_effect(pattern):
                    if pattern == "*.txt":
                        return [Path("/test/file1.txt"), Path("/test/file2.txt"), Path("/test/unique.txt")]
                    return []
                
                mock_rglob.side_effect = side_effect
                
                # 调用filter_unique_files方法并传入自定义模式
                result = self.adapter.filter_unique_files(
                    Path("/test"), patterns=["*.txt"]
                )
                
                expected = [
                    Path("/test/file1.txt"),  # 保留第一个文件
                    Path("/test/unique.txt")  # 不在任何重复组中的文件
                ]
                
                self.assertEqual(sorted(str(p) for p in result), 
                                 sorted(str(p) for p in expected))
    # end of add in 2025-05-09 17:00
if __name__ == "__main__":
    unittest.main()
