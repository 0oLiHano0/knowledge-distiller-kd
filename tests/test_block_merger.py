# tests/prefilter/test_czkawka_adapter.py

import unittest
from pathlib import Path
import json
import subprocess
from unittest.mock import patch

from knowledge_distiller_kd.prefilter.czkawka_adapter import CzkawkaAdapter
from knowledge_distiller_kd.core.models import (
    DuplicateFileInfoDTO,
    DuplicateFileGroupDTO,
)


class TestCzkawkaAdapter(unittest.TestCase):
    def setUp(self):
        self.custom_args = ["duplicates", "--json", "-d"]
        self.adapter = CzkawkaAdapter(
            czkawka_cli_path="czkawka_cli",
            config={"czkawka_args": self.custom_args}
        )

    def test_build_command_default_when_no_config(self):
        adapter = CzkawkaAdapter(
            czkawka_cli_path="czkawka_cli",
            config=None
        )
        cmd = adapter._build_command(Path("/test/dir"))
        self.assertEqual(
            cmd,
            ["czkawka_cli", "duplicates", "--json", "-d", "/test/dir"]
        )

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run')
    def test_scan_directory_returns_correct_dtos_on_success(self, mock_run):
        sample_json = [
            {
                "files": [
                    {"path": "/file1.txt", "size": 100, "modified": 123456},
                    {"path": "/file2.txt", "size": 100, "modified": 123456}
                ]
            }
        ]
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(sample_json),
            stderr=""
        )
        mock_run.return_value = mock_result

        result = self.adapter.scan_directory_for_duplicates(Path("/some_dir"))

        expected = [
            DuplicateFileGroupDTO(
                files=[
                    DuplicateFileInfoDTO(
                        path="/file1.txt",
                        size=100,
                        modified=123456
                    ),
                    DuplicateFileInfoDTO(
                        path="/file2.txt",
                        size=100,
                        modified=123456
                    )
                ]
            )
        ]
        self.assertEqual(result, expected)
        mock_run.assert_called_once_with(
            ["czkawka_cli"] + self.custom_args + [str(Path("/some_dir"))],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=300,
        )

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run',
           side_effect=FileNotFoundError)
    def test_scan_directory_handles_czkawka_cli_not_found(self, mock_run):
        result = self.adapter.scan_directory_for_duplicates(Path("/"))
        self.assertEqual(result, [])

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run')
    def test_scan_directory_handles_czkawka_error_code(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="error"
        )
        result = self.adapter.scan_directory_for_duplicates(Path("/"))
        self.assertEqual(result, [])

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run')
    def test_scan_directory_handles_invalid_json_output(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="invalid json",
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

    @patch('knowledge_distiller_kd.prefilter.czkawka_adapter.subprocess.run',
           side_effect=subprocess.TimeoutExpired(cmd=["czkawka_cli"], timeout=300))
    def test_scan_directory_handles_timeout_expired(self, mock_run):
        result = self.adapter.scan_directory_for_duplicates(Path("/"))
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
