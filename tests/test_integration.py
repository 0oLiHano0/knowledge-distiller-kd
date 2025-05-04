# tests/test_integration.py
"""
Integration tests for the Knowledge Distiller Engine workflow.
"""

import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock # 导入 mock 工具
from typing import Dict # 导入 Dict

# Import components from their new locations
from knowledge_distiller_kd.core.engine import KnowledgeDistillerEngine
from knowledge_distiller_kd.storage.file_storage import FileStorage
from knowledge_distiller_kd.analysis.semantic_analyzer import SemanticAnalyzer # Import for mocking
from knowledge_distiller_kd.core import constants
from knowledge_distiller_kd.core.utils import create_decision_key, parse_decision_key # 导入 utils

# --- Test Data ---

# Adjusted content to better reflect potential block splitting
FILE1_CONTENT = """# 文件1 Title

这是文件1的普通段落。

```python
def func1():
    print("Hello from file 1")
```

这是另一段普通文本。

又一个段落。
"""

FILE2_CONTENT = """# 文件2 Title

这是文件1的普通段落。

```javascript
function func2() {
    console.log("Hello from file 2");
}
```

这是文件2独有的文本。

又一个段落。
"""

FILE3_CONTENT = """# 文件3 Title

完全不同的内容。

又一个段落。
"""

# --- Fixtures ---

@pytest.fixture(scope="function") # Use function scope for tmp_path isolation
def temp_dirs(tmp_path: Path) -> Dict[str, Path]:
    """Creates temporary input, output, and decision directories."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    decision_dir = tmp_path / "decisions"
    input_dir.mkdir()
    output_dir.mkdir()
    decision_dir.mkdir()
    # Populate input files
    (input_dir / "file1.md").write_text(FILE1_CONTENT, encoding="utf-8")
    (input_dir / "file2.md").write_text(FILE2_CONTENT, encoding="utf-8")
    (input_dir / "file3.md").write_text(FILE3_CONTENT, encoding="utf-8")
    return {
        "input": input_dir,
        "output": output_dir,
        "decisions": decision_dir,
        "decision_file": decision_dir / "decisions.json" # Default decision file path
    }

@pytest.fixture
def file_storage() -> FileStorage:
    """Provides a FileStorage instance."""
    return FileStorage()

@pytest.fixture
def mock_semantic_analyzer(mocker) -> MagicMock:
    """Mocks the SemanticAnalyzer class."""
    mock = MagicMock(spec=SemanticAnalyzer)
    # Mock methods to avoid real model loading/computation
    mock.load_semantic_model.return_value = True # Simulate successful (mock) load
    mock.find_semantic_duplicates.return_value = [] # Simulate finding no semantic duplicates
    # Patch the class in the engine module where it's imported/used
    mocker.patch("knowledge_distiller_kd.core.engine.SemanticAnalyzer", return_value=mock)
    return mock

@pytest.fixture
def engine_instance(temp_dirs: Dict[str, Path], file_storage: FileStorage, mock_semantic_analyzer: MagicMock) -> KnowledgeDistillerEngine:
    """
    Creates a KnowledgeDistillerEngine instance configured for integration tests,
    with SemanticAnalyzer mocked.
    """
    # Engine needs configuration passed to __init__
    engine = KnowledgeDistillerEngine(
        storage=file_storage,
        input_dir=temp_dirs["input"],
        output_dir=temp_dirs["output"],
        decision_file=temp_dirs["decision_file"],
        similarity_threshold=0.95, # Keep high to further avoid accidental semantic matches if mock fails
        skip_semantic=False # We mock it, so don't explicitly skip
    )
    return engine

# --- Test Cases ---

def test_integration_full_workflow(engine_instance: KnowledgeDistillerEngine, temp_dirs: Dict[str, Path]):
    """
    Test the core workflow: analyze -> check decisions -> apply -> check output.
    Uses mocked SemanticAnalyzer.
    """
    input_dir = temp_dirs["input"]
    output_dir = temp_dirs["output"]

    # 1. Run Analysis
    analysis_success = engine_instance.run_analysis()
    assert analysis_success is True
    assert engine_instance._analysis_completed is True
    assert len(engine_instance.blocks_data) > 0 # Should have processed blocks

    # 2. Check MD5 Results and Default Decisions
    md5_duplicates = engine_instance.get_md5_duplicates()
    # Expecting 2 groups: "这是文件1的普通段落。" and "又一个段落。"
    assert len(md5_duplicates) == 2

    # Find the specific duplicate paragraph group ("这是文件1的普通段落。") and check decisions
    target_text_1 = "这是文件1的普通段落。"
    found_md5_para_1 = False
    kept_key_abs_1 = None
    deleted_key_abs_1 = None

    for group in md5_duplicates:
        # Check based on normalized text as used by MD5 analyzer
        group_texts = {block.analysis_text for block in group}
        if target_text_1 in group_texts:
            found_md5_para_1 = True
            assert len(group) == 2
            # Determine which block is kept/deleted based on file order (assuming file1 comes first)
            sorted_group = sorted(group, key=lambda b: Path(b.file_path).name)
            kept_block = sorted_group[0]
            deleted_block = sorted_group[1]
            kept_key_abs_1 = create_decision_key(str(Path(kept_block.file_path).resolve()), kept_block.block_id, kept_block.block_type)
            deleted_key_abs_1 = create_decision_key(str(Path(deleted_block.file_path).resolve()), deleted_block.block_id, deleted_block.block_type)
            break # Found the target group

    assert found_md5_para_1, f"MD5 did not find the expected duplicate paragraph: '{target_text_1}'"
    assert kept_key_abs_1 is not None
    assert deleted_key_abs_1 is not None
    # Check decisions made automatically by MD5 analysis (first keep, rest delete)
    assert engine_instance.block_decisions.get(kept_key_abs_1) == constants.DECISION_KEEP
    assert engine_instance.block_decisions.get(deleted_key_abs_1) == constants.DECISION_DELETE

    # Check the second duplicate paragraph group ("又一个段落。")
    target_text_2 = "又一个段落。"
    found_md5_para_2 = False
    kept_key_abs_2 = None
    deleted_keys_abs_2 = []
    for group in md5_duplicates:
        group_texts = {block.analysis_text for block in group}
        if target_text_2 in group_texts:
            found_md5_para_2 = True
            assert len(group) == 3 # Should be in file1, file2, file3
            sorted_group = sorted(group, key=lambda b: Path(b.file_path).name) # Sort by filename
            kept_block = sorted_group[0] # Assume file1 is kept
            kept_key_abs_2 = create_decision_key(str(Path(kept_block.file_path).resolve()), kept_block.block_id, kept_block.block_type)
            deleted_keys_abs_2 = [
                create_decision_key(str(Path(b.file_path).resolve()), b.block_id, b.block_type)
                for b in sorted_group[1:]
            ]
            break
    assert found_md5_para_2, f"MD5 did not find the second duplicate paragraph: '{target_text_2}'"
    assert kept_key_abs_2 is not None
    assert len(deleted_keys_abs_2) == 2
    assert engine_instance.block_decisions.get(kept_key_abs_2) == constants.DECISION_KEEP
    for key in deleted_keys_abs_2:
        assert engine_instance.block_decisions.get(key) == constants.DECISION_DELETE


    # Check semantic results (should be empty due to mocking)
    semantic_duplicates = engine_instance.get_semantic_duplicates()
    assert semantic_duplicates == []

    # 3. Apply Decisions
    apply_success = engine_instance.apply_decisions()
    assert apply_success is True

    # 4. Verify Output Files
    output_file1 = output_dir / f"file1{constants.DEFAULT_OUTPUT_SUFFIX}.md"
    output_file2 = output_dir / f"file2{constants.DEFAULT_OUTPUT_SUFFIX}.md"
    output_file3 = output_dir / f"file3{constants.DEFAULT_OUTPUT_SUFFIX}.md"

    assert output_file1.exists(), f"Output file not found: {output_file1}"
    assert output_file2.exists(), f"Output file not found: {output_file2}"
    assert output_file3.exists(), f"Output file not found: {output_file3}"

    # Check content (adjust expectations based on actual block splitting and content)
    content1 = output_file1.read_text(encoding="utf-8")
    assert "文件1 Title" in content1 # Title text should be kept
    assert "这是文件1的普通段落。" in content1 # Kept
    assert "def func1():" in content1
    assert "这是另一段普通文本。" in content1
    assert "又一个段落。" in content1 # Kept (first occurrence)

    content2 = output_file2.read_text(encoding="utf-8")
    assert "文件2 Title" in content2
    assert "这是文件1的普通段落。" not in content2 # Deleted by MD5 auto-decision
    assert "function func2()" in content2
    assert "这是文件2独有的文本。" in content2
    assert "又一个段落。" not in content2 # Deleted (second occurrence)

    content3 = output_file3.read_text(encoding="utf-8")
    assert "文件3 Title" in content3
    assert "完全不同的内容。" in content3
    assert "又一个段落。" not in content3 # Deleted (third occurrence)

# --- Updated test_integration_save_load_decisions ---
def test_integration_save_load_decisions(engine_instance: KnowledgeDistillerEngine, temp_dirs: Dict[str, Path]):
    """
    Test saving and loading decisions integrates correctly with the engine
    by comparing the decision dictionary before saving and after loading.
    """
    input_dir = temp_dirs["input"]
    decision_file = temp_dirs["decision_file"]

    # 1. Run analysis to populate blocks and initial decisions
    analysis_success = engine_instance.run_analysis()
    assert analysis_success
    assert len(engine_instance.block_decisions) > 0

    # 2. (Optional) Manually change a decision to ensure save/load handles changes
    key_to_modify = None
    for key in engine_instance.block_decisions: # Find any key
        key_to_modify = key
        break
    assert key_to_modify is not None, "No decisions found to modify"
    original_decision = engine_instance.block_decisions[key_to_modify]
    new_decision = constants.DECISION_DELETE if original_decision != constants.DECISION_DELETE else constants.DECISION_KEEP
    engine_instance.block_decisions[key_to_modify] = new_decision

    # 3. Capture the expected decision state *before* saving
    # Create a copy to avoid modification issues
    expected_decisions_after_load = engine_instance.block_decisions.copy()

    # 4. Save decisions
    save_success = engine_instance.save_decisions()
    assert save_success is True
    assert decision_file.exists()
    assert decision_file.stat().st_size > 0

    # 5. Create a new engine instance
    new_engine = KnowledgeDistillerEngine(
        storage=engine_instance.storage,
        input_dir=input_dir,
        output_dir=engine_instance.output_dir,
        decision_file=decision_file,
        similarity_threshold=engine_instance.similarity_threshold,
        skip_semantic=engine_instance.skip_semantic
    )
    # Ensure the new engine starts with empty decisions
    assert len(new_engine.block_decisions) == 0

    # 6. Load decisions into the new engine
    load_success = new_engine.load_decisions()
    assert load_success is True

    # 7. Verify the loaded decision dictionary matches the saved one
    # Compare the entire dictionaries
    assert new_engine.block_decisions == expected_decisions_after_load

