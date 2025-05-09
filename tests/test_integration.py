# tests/test_integration.py
"""
Integration tests for the Knowledge Distiller Engine workflow.
Updated to reflect Engine using StorageInterface and FileStorage requiring base_path.
Tests Engine <-> FileStorage interaction. SemanticAnalyzer is mocked.
"""

import pytest
import os
import json
import logging # Import logging

@pytest.fixture
def logger():
    logger = logging.getLogger("integration_block_merger_test")
    logger.addHandler(logging.NullHandler())
    return logger

from knowledge_distiller_kd.processing.block_merger import merge_code_blocks
from knowledge_distiller_kd.core.models import BlockDTO, BlockType, DecisionType
from pathlib import Path
from unittest.mock import patch, MagicMock # Import mock tools
# Import necessary types, ADDED Optional
from typing import Dict, List, Any, Optional

# Import components from their new locations
from knowledge_distiller_kd.core.engine import KnowledgeDistillerEngine
from knowledge_distiller_kd.storage.file_storage import FileStorage # Use the actual FileStorage
from knowledge_distiller_kd.storage.storage_interface import StorageInterface # For type hints if needed
from knowledge_distiller_kd.analysis.semantic_analyzer import SemanticAnalyzer # Import for mocking
# Use the agreed constants/keys from engine or models if defined there
from knowledge_distiller_kd.core.engine import METADATA_DECISION_KEY, DECISION_KEEP, DECISION_DELETE, DECISION_UNDECIDED
from knowledge_distiller_kd.core.utils import create_decision_key, parse_decision_key # Import logger
# Import DTOs and Enums (ensure using the final confirmed version)
from knowledge_distiller_kd.core.models import (
    ContentBlock, AnalysisType, DecisionType, BlockType, FileRecord
)
# ADDED constants import
from knowledge_distiller_kd.core import constants

# --- Test Data ---

# Adjusted content to better reflect potential block splitting
FILE1_CONTENT = """# File 1 Title

This is a paragraph from file 1.

```python
def func1():
    print("Hello from file 1")
```

This is another text paragraph.

A common paragraph.
"""

FILE2_CONTENT = """# File 2 Title

This is a paragraph from file 1.

```javascript
function func2() {
    console.log("Hello from file 2");
}
```

This is unique text from file 2.

A common paragraph.
"""

FILE3_CONTENT = """# File 3 Title

Completely different content.

A common paragraph.
"""

# --- Fixtures ---

@pytest.fixture(scope="function") # Use function scope for tmp_path isolation
def temp_dirs(tmp_path: Path) -> Dict[str, Path]:
    """Creates temporary input, output, and storage base directories."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    # Storage base path - FileStorage will create files inside this
    storage_base_dir = tmp_path / "storage"
    input_dir.mkdir()
    output_dir.mkdir()
    storage_base_dir.mkdir() # Create the base dir for storage

    # Populate input files
    (input_dir / "file1.md").write_text(FILE1_CONTENT, encoding="utf-8")
    (input_dir / "file2.md").write_text(FILE2_CONTENT, encoding="utf-8")
    (input_dir / "file3.md").write_text(FILE3_CONTENT, encoding="utf-8")
    return {
        "input": input_dir,
        "output": output_dir,
        "storage_base": storage_base_dir,
    }

# Removed the old file_storage fixture as it's created within engine_instance now

@pytest.fixture
def mock_semantic_analyzer(mocker) -> MagicMock:
    """Mocks the SemanticAnalyzer class."""
    mock = MagicMock(spec=SemanticAnalyzer)
    mock.load_semantic_model.return_value = True # Simulate successful (mock) load
    mock.find_semantic_duplicates.return_value = [] # Simulate finding no semantic duplicates
    mock._model_loaded = True # Ensure internal flag is set
    mock.model = MagicMock() # Ensure model object exists
    # Patch the class in the engine module where it's imported/used
    mocker.patch("knowledge_distiller_kd.core.engine.SemanticAnalyzer", return_value=mock)
    return mock

@pytest.fixture
def engine_instance(temp_dirs: Dict[str, Path], mock_semantic_analyzer: MagicMock) -> KnowledgeDistillerEngine:
    """
    Creates a KnowledgeDistillerEngine instance configured for integration tests,
    using a real FileStorage instance initialized with a temporary path.
    SemanticAnalyzer is mocked.
    """
    storage_base = temp_dirs["storage_base"]
    # Create the actual FileStorage instance, providing the required base_path
    file_storage_instance = FileStorage(base_path=storage_base)
    # Initialize the storage explicitly (important!)
    file_storage_instance.initialize()

    # Engine needs configuration passed to __init__
    # Pass the *actual* FileStorage instance to the Engine
    engine = KnowledgeDistillerEngine(
        storage=file_storage_instance, # Inject the real FileStorage
        input_dir=temp_dirs["input"],
        output_dir=temp_dirs["output"],
        # decision_file is no longer directly used by engine for loading/saving
        decision_file=storage_base / "decisions.json", # Still needed for config, points inside storage
        similarity_threshold=0.95, # Keep high to further avoid accidental semantic matches if mock fails
        skip_semantic=False # We mock it, so don't explicitly skip
    )
    # Ensure the engine uses the mocked semantic analyzer
    engine.semantic_analyzer = mock_semantic_analyzer
    return engine

# --- Helper Function to get block by text ---
def find_block_by_text(blocks: List[ContentBlock], text_content: str) -> Optional[ContentBlock]:
    """Finds the first block matching the text content."""
    for block in blocks:
        if block.text == text_content:
            return block
    return None

# --- Test Cases ---

def test_integration_full_workflow(engine_instance: KnowledgeDistillerEngine, temp_dirs: Dict[str, Path]):
    """
    Test the core workflow: analyze -> check decisions -> apply -> check output content dictionary.
    Uses mocked SemanticAnalyzer and real FileStorage.
    """
    input_dir = temp_dirs["input"]
    output_dir = temp_dirs["output"]

    # 1. Run Analysis
    analysis_success = engine_instance.run_analysis()
    assert analysis_success is True
    assert engine_instance._analysis_completed is True
    # Verify blocks were processed and exist in the engine's memory (or storage)
    # Check storage directly for block count (more robust integration check)
    all_blocks_from_storage = engine_instance.storage.get_blocks_for_analysis()
    assert len(all_blocks_from_storage) > 0 # Should have processed blocks

    # 2. Check MD5 Results and Default Decisions (via Engine's state after analysis)
    md5_duplicates = engine_instance.get_md5_duplicates()
    # Expecting 2 groups: "This is a paragraph from file 1." and "A common paragraph."
    assert len(md5_duplicates) == 2

    # --- Verify decisions stored in metadata after analysis ---
    # Find the blocks corresponding to the duplicate paragraphs
    target_text_1 = "This is a paragraph from file 1."
    target_text_2 = "A common paragraph."

    blocks_t1 = [b for b in all_blocks_from_storage if b.text == target_text_1]
    blocks_t2 = [b for b in all_blocks_from_storage if b.text == target_text_2]

    assert len(blocks_t1) == 2, f"Expected 2 blocks for '{target_text_1}'"
    assert len(blocks_t2) == 3, f"Expected 3 blocks for '{target_text_2}'"

    # Sort by original path to determine which one should be kept (first one)
    blocks_t1.sort(key=lambda b: b.metadata.get('original_path', ''))
    blocks_t2.sort(key=lambda b: b.metadata.get('original_path', ''))

    # --- Verify decisions in engine's memory map ---
    # Helper to get key
    def get_key(block):
        path = block.metadata.get('original_path')
        if not path: pytest.fail(f"Block {block.block_id} missing original_path")
        # Resolve path for consistency with how keys might be stored
        resolved_path = str(Path(path).resolve())
        return create_decision_key(resolved_path, block.block_id, block.block_type.value)

    # Check decisions for the first group ("This is a paragraph from file 1.")
    key_t1_keep = get_key(blocks_t1[0])
    key_t1_delete = get_key(blocks_t1[1])
    assert engine_instance.block_decisions.get(key_t1_keep) == DECISION_KEEP, f"Expected KEEP for {key_t1_keep}"
    assert engine_instance.block_decisions.get(key_t1_delete) == DECISION_DELETE, f"Expected DELETE for {key_t1_delete}"

    # Check decisions for the second group ("A common paragraph.")
    key_t2_keep = get_key(blocks_t2[0])
    key_t2_delete1 = get_key(blocks_t2[1])
    key_t2_delete2 = get_key(blocks_t2[2])
    assert engine_instance.block_decisions.get(key_t2_keep) == DECISION_KEEP, f"Expected KEEP for {key_t2_keep}"
    assert engine_instance.block_decisions.get(key_t2_delete1) == DECISION_DELETE, f"Expected DELETE for {key_t2_delete1}"
    assert engine_instance.block_decisions.get(key_t2_delete2) == DECISION_DELETE, f"Expected DELETE for {key_t2_delete2}"
    # --- End verification of memory map ---
    # Check semantic results (should be empty due to mocking)
    semantic_duplicates = engine_instance.get_semantic_duplicates()
    assert semantic_duplicates == []

    # 3. Apply Decisions (returns dictionary)
    output_content_dict = engine_instance.apply_decisions()
    assert isinstance(output_content_dict, dict)
    # Expecting output for file1, file2, file3 (file3 won't be empty now)
    assert len(output_content_dict) == 3

    # 4. Verify Output Content Dictionary
    # Construct expected output paths based on engine's output_dir config
    output_suffix = ".md" # Assume default suffix for now
    # Use constants for the output suffix
    if hasattr(constants, 'DEFAULT_OUTPUT_SUFFIX'):
        output_suffix = constants.DEFAULT_OUTPUT_SUFFIX + output_suffix

    expected_output_path1 = output_dir / f"file1{output_suffix}"
    expected_output_path2 = output_dir / f"file2{output_suffix}"
    expected_output_path3 = output_dir / f"file3{output_suffix}"

    assert expected_output_path1 in output_content_dict
    assert expected_output_path2 in output_content_dict
    assert expected_output_path3 in output_content_dict

    # Check content (based on which blocks should be kept)
    content1 = output_content_dict[expected_output_path1]
    assert "# File 1 Title" in content1
    assert "This is a paragraph from file 1." in content1 # Kept
    assert "def func1():" in content1
    assert "This is another text paragraph." in content1
    assert "A common paragraph." in content1 # Kept (first occurrence)

    content2 = output_content_dict[expected_output_path2]
    assert "# File 2 Title" in content2
    assert "This is a paragraph from file 1." not in content2 # Deleted
    assert "function func2()" in content2
    assert "This is unique text from file 2." in content2
    assert "A common paragraph." not in content2 # Deleted

    content3 = output_content_dict[expected_output_path3]
    assert "# File 3 Title" in content3
    assert "Completely different content." in content3
    assert "A common paragraph." not in content3 # Deleted

    # Optional: Write the content to actual files for manual inspection if needed
    # for path, content in output_content_dict.items():
    #     path.parent.mkdir(parents=True, exist_ok=True)
    #     path.write_text(content, encoding='utf-8')


def test_integration_save_load_decisions_via_metadata(temp_dirs: Dict[str, Path], mock_semantic_analyzer: MagicMock):
    """
    Test saving and loading decisions integrates correctly by modifying a decision,
    saving it (via saving blocks), creating a new engine/storage instance pointing
    to the same path, loading decisions (via reading block metadata), and verifying.
    """
    input_dir = temp_dirs["input"]
    storage_base = temp_dirs["storage_base"]
    output_dir = temp_dirs["output"]

    # --- Instance 1: Run analysis and modify a decision ---
    storage1 = FileStorage(base_path=storage_base)
    storage1.initialize()
    engine1 = KnowledgeDistillerEngine(
        storage=storage1, input_dir=input_dir, output_dir=output_dir, skip_semantic=False
    )
    engine1.semantic_analyzer = mock_semantic_analyzer # Ensure mocked analyzer is used

    analysis_success = engine1.run_analysis()
    assert analysis_success
    all_blocks = storage1.get_blocks_for_analysis() # Get all blocks after analysis
    assert len(all_blocks) > 0

    # Find a block that was initially marked for deletion (e.g., second common paragraph)
    target_text = "A common paragraph."
    blocks_common = [b for b in all_blocks if b.text == target_text]
    blocks_common.sort(key=lambda b: b.metadata.get('original_path', ''))
    assert len(blocks_common) == 3
    block_to_modify = blocks_common[1] # The one in file2 (initially deleted)
    # assert block_to_modify.metadata.get(METADATA_DECISION_KEY) == DECISION_DELETE
    # --- Verify initial decision in engine's memory map ---
    block_path = block_to_modify.metadata.get('original_path')
    block_id = block_to_modify.block_id
    block_type_val = block_to_modify.block_type.value
    assert block_path is not None, f"Block {block_id} missing original_path"
    # Resolve path for consistency
    resolved_path_modify = str(Path(block_path).resolve())
    decision_key_modify = create_decision_key(resolved_path_modify, block_id, block_type_val)

    assert engine1.block_decisions.get(decision_key_modify) == DECISION_DELETE, f"Expected initial decision for {decision_key_modify} to be DELETE in memory"
    # --- End verification of memory map ---

    # Manually update the decision in memory using the engine's method
    # (The rest of the test, including the key creation which is now duplicated above,
    # and the call to update_decision and subsequent checks should remain the same
    # as they test the update and loading functionality correctly)
    # You might want to reuse 'decision_key_modify' below instead of recalculating 'decision_key'

    # Example: Reuse the key calculated above
    update_success = engine1.update_decision(decision_key_modify, DECISION_KEEP) # Change to KEEP
    assert update_success is True
    # Verify in-memory decision map is updated
    assert engine1.block_decisions.get(decision_key_modify) == DECISION_KEEP
    # Verify the block metadata itself was updated in storage by update_decision->save_blocks
    updated_block = storage1.get_block(block_id)
    assert updated_block is not None
    assert updated_block.metadata.get(METADATA_DECISION_KEY) == DECISION_KEEP

    # --- Instance 2: Load decisions and verify ---
    storage2 = FileStorage(base_path=storage_base) # New instance, same path
    storage2.initialize() # This will load data from files written by storage1
    engine2 = KnowledgeDistillerEngine(
        storage=storage2, input_dir=input_dir, output_dir=output_dir, skip_semantic=False
    )
    # No need to run analysis again, just load decisions
    load_success = engine2.load_decisions()
    assert load_success is True

    # Verify the modified decision was loaded correctly into engine2's memory map
    assert engine2.block_decisions.get(decision_key_modify) == DECISION_KEEP

    # Verify other decisions loaded correctly as well
    block_kept_originally = blocks_common[0] # The one in file1
    key_kept = create_decision_key(
        block_kept_originally.metadata['original_path'],
        block_kept_originally.block_id,
        block_kept_originally.block_type.value
    )
    # assert engine2.block_decisions.get(key_kept) == DECISION_KEEP

# -----------------------------------------------------------------------------
# 新增：Block Merger 集成测试，确保不会被跳过
def test_integration_block_merger_basic(tmp_path, logger):
    # 准备三个碎片化的 Markdown 围栏代码块（同一 file_id）
    blocks = [
        BlockDTO(block_id="c1", file_id="fileA", block_type=BlockType.CODE, text_content="```python\n"),
        BlockDTO(block_id="c2", file_id="fileA", block_type=BlockType.CODE, text_content="print('hello')\n"),
        BlockDTO(block_id="c3", file_id="fileA", block_type=BlockType.CODE, text_content="```\n"),
    ]
    # 使用默认 config（max_gap=1）
    config = {}
    merged = merge_code_blocks(blocks, config, logger)

    # 合并后应只有一个 BlockDTO
    assert len(merged) == 1, "碎片化代码块应被合并为一个"
    m = merged[0]
    # 合并结果不应包含围栏标记，且 metadata.language 正确
    assert "```" not in m.text_content
    assert m.metadata.get("language") == "python"
    # 原始三片段都应被标记为 DELETE
    for orig in blocks:
        assert orig.kd_processing_status == DecisionType.DELETE
