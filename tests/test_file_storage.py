# tests/storage/test_file_storage.py
"""
Tests for the FileStorage class.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, List

# 导入需要测试的类
from knowledge_distiller_kd.storage.file_storage import FileStorage
from knowledge_distiller_kd.core.error_handler import FileOperationError # 导入可能需要的错误类型
# 导入常量，如果需要的话
from knowledge_distiller_kd.core import constants


@pytest.fixture
def sample_decisions_data() -> List[Dict]:
    """Provides sample decision data for saving."""
    return [
        {
            "file": "file1.md",
            "block_id": "id1",
            "type": "NarrativeText",
            "decision": "keep"
        },
        {
            "file": "relative/path/file2.md",
            "block_id": "id2",
            "type": "CodeSnippet",
            "decision": "delete"
        }
    ]

def test_save_decisions_creates_file(tmp_path: Path, sample_decisions_data: List[Dict]):
    """
    Test that save_decisions creates the specified file.
    """
    decision_file_path = tmp_path / "test_save_decisions.json"
    assert not decision_file_path.exists()

    storage = FileStorage()
    save_success = storage.save_decisions(decision_file_path, sample_decisions_data)

    assert save_success is True # Check return value
    assert decision_file_path.exists()
    try:
        with open(decision_file_path, 'r', encoding=constants.DEFAULT_ENCODING) as f:
            loaded_data = json.load(f)
        assert loaded_data == sample_decisions_data # Verify content matches
    except json.JSONDecodeError:
        pytest.fail(f"File {decision_file_path} does not contain valid JSON.")
    except Exception as e:
        pytest.fail(f"Error reading created file {decision_file_path}: {e}")

# --- 新增测试用例 ---
def test_load_decisions_success(tmp_path: Path, sample_decisions_data: List[Dict]):
    """
    Test successfully loading decisions from an existing file.
    """
    decision_file_path = tmp_path / "test_load_decisions.json"

    # 1. Setup: Create a decision file first using save_decisions
    storage_writer = FileStorage()
    storage_writer.save_decisions(decision_file_path, sample_decisions_data)
    assert decision_file_path.exists()

    # 2. Test: Load the decisions using a new instance
    storage_reader = FileStorage()
    loaded_data = storage_reader.load_decisions(decision_file_path)

    # 3. Assert: Loaded data should match the original saved data
    assert isinstance(loaded_data, list)
    assert loaded_data == sample_decisions_data

def test_load_decisions_file_not_found(tmp_path: Path):
    """
    Test loading decisions when the file does not exist.
    It should return an empty list without raising an error.
    """
    non_existent_file = tmp_path / "non_existent_decisions.json"
    storage = FileStorage()
    loaded_data = storage.load_decisions(non_existent_file)
    assert loaded_data == []

def test_load_decisions_invalid_json(tmp_path: Path):
    """
    Test loading decisions from a file with invalid JSON content.
    It should return an empty list and log an error.
    """
    invalid_json_file = tmp_path / "invalid_decisions.json"
    invalid_json_file.write_text("{invalid json content", encoding=constants.DEFAULT_ENCODING)

    storage = FileStorage()
    # Capture logs if needed to verify error logging, or just check return value
    loaded_data = storage.load_decisions(invalid_json_file)
    assert loaded_data == []

def test_load_decisions_not_a_list(tmp_path: Path):
    """
    Test loading decisions from a file where the top-level JSON is not a list.
    It should return an empty list and log an error.
    """
    not_a_list_file = tmp_path / "not_a_list_decisions.json"
    not_a_list_file.write_text('{"key": "value"}', encoding=constants.DEFAULT_ENCODING) # Save a dictionary

    storage = FileStorage()
    loaded_data = storage.load_decisions(not_a_list_file)
    assert loaded_data == []


# --- Add more tests later for error handling during save (e.g., permissions) ---

