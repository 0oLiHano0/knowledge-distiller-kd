import unittest
from pathlib import Path
import json
import subprocess
from unittest.mock import patch

from knowledge_distiller_kd.prefilter.czkawka_adapter import CzkawkaAdapter
from knowledge_distiller_kd.core.models import DuplicateFileInfoDTO, DuplicateFileGroupDTO

class TestCzkawkaAdapter(unittest.TestCase):
    def setUp(self):
        self.custom_args = ["duplicates", "--json", "-d"]
        self.adapter = CzkawkaAdapter(
            czkawka_cli_path="czkawka_cli",
            config={"czkawka_args": self.custom_args}
        )

    def test_build_command_default(self):
        dir_path = Path("/tmp/testdir")
        cmd = self.adapter._build_command(dir_path)
        expected = ["czkawka_cli"] + self.custom_args + [str(dir_path)]
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

if __name__ == "__main__":
    unittest.main()
