# tests/test_md5_analyzer.py
"""
Unit tests for the MD5Analyzer class.
"""

import pytest
from typing import List, Dict, Any
from pathlib import Path # 导入 Path

# Corrected imports based on new structure
from knowledge_distiller_kd.analysis.md5_analyzer import MD5Analyzer
from knowledge_distiller_kd.processing.document_processor import ContentBlock
from knowledge_distiller_kd.core import constants
from knowledge_distiller_kd.core.utils import create_decision_key

# Import necessary element types from unstructured
from unstructured.documents.elements import NarrativeText, Title, CodeSnippet, Text

# --- Fixtures ---

@pytest.fixture
def md5_analyzer() -> MD5Analyzer:
    """Creates a new MD5Analyzer instance for each test."""
    return MD5Analyzer()

@pytest.fixture
def create_content_block_md5():
    """
    Factory fixture to create ContentBlock instances specifically for MD5 tests.
    Uses simple text and basic element types. Need to use unique IDs for blocks
    unless testing identical blocks across files.
    Resolves file_path to absolute path for consistent key generation.
    """
    _block_counter = 0
    # Use tmp_path fixture provided by pytest for unique absolute paths per test run
    base_dir = Path(f"test_md5_run_{_block_counter}") # Simple base dir name

    def _create(
        text: str,
        file_path: str = "test.md",
        block_type_cls: type = NarrativeText,
        element_id: str | None = None
    ) -> ContentBlock:
        nonlocal _block_counter
        if element_id is None:
            _block_counter += 1
            element_id = f"md5_test_{_block_counter}"

        # --- Use absolute path ---
        abs_file_path = (base_dir / file_path).resolve()

        # Create a basic unstructured element
        element = block_type_cls(text=text, element_id=element_id)
        # Pass the absolute path string to ContentBlock
        cb = ContentBlock(element, str(abs_file_path))
        # Ensure original_text and analysis_text are populated for the test
        cb.original_text = text
        cb.analysis_text = cb._normalize_text() # Use the actual normalization
        return cb
    return _create

# Helper function to get absolute key
def get_abs_key(block: ContentBlock) -> str:
    # Ensure the path used for the key is absolute
    abs_path_str = str(Path(block.file_path).resolve())
    return create_decision_key(abs_path_str, block.block_id, block.block_type)

# --- Test Cases ---

def test_initialization(md5_analyzer: MD5Analyzer):
    """Test if MD5Analyzer initializes without errors."""
    assert isinstance(md5_analyzer, MD5Analyzer)

def test_find_md5_duplicates_empty_blocks(md5_analyzer: MD5Analyzer):
    """Test with no content blocks."""
    input_blocks: List[ContentBlock] = []
    input_decisions: Dict[str, str] = {}
    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)
    assert groups == []
    assert suggestions == {}

def test_find_md5_duplicates_single_block(md5_analyzer: MD5Analyzer, create_content_block_md5):
    """Test with only one content block."""
    block1 = create_content_block_md5("Single block content")
    input_blocks = [block1]
    key1_abs = get_abs_key(block1)
    input_decisions = {key1_abs: constants.DECISION_UNDECIDED}
    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)
    assert groups == []
    assert suggestions == {}

def test_find_md5_duplicates_identical_blocks(md5_analyzer: MD5Analyzer, create_content_block_md5):
    """Test with two identical content blocks."""
    content = "Identical content"
    block1 = create_content_block_md5(content, file_path="file1.md", element_id="id1")
    block2 = create_content_block_md5(content, file_path="file2.md", element_id="id2")
    input_blocks = [block1, block2]
    key1_abs = get_abs_key(block1)
    key2_abs = get_abs_key(block2)
    input_decisions = {key1_abs: constants.DECISION_UNDECIDED, key2_abs: constants.DECISION_UNDECIDED}

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert len(groups) == 1
    assert set(groups[0]) == {block1, block2}
    assert len(suggestions) == 2
    # Assert using absolute keys now
    assert suggestions[key1_abs] == constants.DECISION_KEEP
    assert suggestions[key2_abs] == constants.DECISION_DELETE

def test_find_md5_duplicates_different_blocks(md5_analyzer: MD5Analyzer, create_content_block_md5):
    """Test with two different content blocks."""
    block1 = create_content_block_md5("Content One", element_id="id1")
    block2 = create_content_block_md5("Content Two", element_id="id2")
    input_blocks = [block1, block2]
    key1_abs = get_abs_key(block1)
    key2_abs = get_abs_key(block2)
    input_decisions = {key1_abs: constants.DECISION_UNDECIDED, key2_abs: constants.DECISION_UNDECIDED}

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert groups == []
    assert suggestions == {}

def test_find_md5_duplicates_mixed_blocks(md5_analyzer: MD5Analyzer, create_content_block_md5):
    """Test with a mix of unique and duplicate blocks."""
    content_a = "Content A (Duplicate)"
    content_b = "Content B (Unique)"
    block1 = create_content_block_md5(content_a, file_path="file1.md", element_id="id1")
    block2 = create_content_block_md5(content_b, file_path="file2.md", element_id="id2")
    block3 = create_content_block_md5(content_a, file_path="file3.md", element_id="id3")
    input_blocks = [block1, block2, block3]
    key1_abs = get_abs_key(block1)
    key2_abs = get_abs_key(block2)
    key3_abs = get_abs_key(block3)
    input_decisions = {
        key1_abs: constants.DECISION_UNDECIDED,
        key2_abs: constants.DECISION_UNDECIDED,
        key3_abs: constants.DECISION_UNDECIDED,
    }

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert len(groups) == 1
    assert set(groups[0]) == {block1, block3}
    assert len(suggestions) == 2
    assert suggestions[key1_abs] == constants.DECISION_KEEP
    assert suggestions[key3_abs] == constants.DECISION_DELETE
    assert key2_abs not in suggestions

def test_find_md5_duplicates_identical_titles_skipped(md5_analyzer: MD5Analyzer, create_content_block_md5):
    """Test that identical Title blocks are skipped by default."""
    block1 = create_content_block_md5("# Same Title", block_type_cls=Title, element_id="id1")
    block2 = create_content_block_md5("## Same Title", block_type_cls=Title, element_id="id2")
    input_blocks = [block1, block2]
    key1_abs = get_abs_key(block1)
    key2_abs = get_abs_key(block2)
    input_decisions = {key1_abs: constants.DECISION_UNDECIDED, key2_abs: constants.DECISION_UNDECIDED}

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert groups == []
    assert suggestions == {}

def test_find_md5_duplicates_empty_content(md5_analyzer: MD5Analyzer, create_content_block_md5):
    """Test with blocks having empty or whitespace-only content."""
    # Need unique IDs if file paths are the same in the fixture base dir
    block1 = create_content_block_md5("", file_path="file1.md", block_type_cls=Text, element_id="id_empty")
    block2 = create_content_block_md5("   \n\t  ", file_path="file2.md", block_type_cls=Text, element_id="id_space")
    input_blocks = [block1, block2]
    key1_abs = get_abs_key(block1)
    key2_abs = get_abs_key(block2)
    input_decisions = {key1_abs: constants.DECISION_UNDECIDED, key2_abs: constants.DECISION_UNDECIDED}

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert len(groups) == 1
    assert set(groups[0]) == {block1, block2}
    assert len(suggestions) == 2
    # Use absolute keys for assertion
    assert suggestions[key1_abs] == constants.DECISION_KEEP
    assert suggestions[key2_abs] == constants.DECISION_DELETE

def test_find_md5_duplicates_normalize_text(md5_analyzer: MD5Analyzer, create_content_block_md5):
    """Test that normalization (whitespace, markdown) happens before hashing."""
    block1 = create_content_block_md5("Text   with\n\n extra  space", element_id="id1")
    block2 = create_content_block_md5("Text with extra space", element_id="id2")
    input_blocks = [block1, block2]
    key1_abs = get_abs_key(block1)
    key2_abs = get_abs_key(block2)
    input_decisions = {key1_abs: constants.DECISION_UNDECIDED, key2_abs: constants.DECISION_UNDECIDED}

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert len(groups) == 1
    assert set(groups[0]) == {block1, block2}
    assert len(suggestions) == 2
    # Use absolute keys for assertion
    assert suggestions[key1_abs] == constants.DECISION_KEEP
    assert suggestions[key2_abs] == constants.DECISION_DELETE

def test_find_md5_duplicates_pre_deleted_blocks(md5_analyzer: MD5Analyzer, create_content_block_md5):
    """Test that blocks already marked as DELETE are ignored in suggestions."""
    content = "Identical content"
    block1 = create_content_block_md5(content, file_path="file1.md", element_id="id1")
    block2 = create_content_block_md5(content, file_path="file2.md", element_id="id2")
    block3 = create_content_block_md5(content, file_path="file3.md", element_id="id3")
    input_blocks = [block1, block2, block3]
    key1_abs = get_abs_key(block1)
    key2_abs = get_abs_key(block2)
    key3_abs = get_abs_key(block3)
    # Mark block2 as already deleted using absolute key
    input_decisions = {
        key1_abs: constants.DECISION_UNDECIDED,
        key2_abs: constants.DECISION_DELETE, # Pre-deleted
        key3_abs: constants.DECISION_UNDECIDED,
    }

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert len(groups) == 1
    # 修改为:
    assert set(groups[0]) == {block1, block3}
    # Suggestions should only be generated for non-deleted blocks in the group
    # Analyzer should now respect the pre-deleted status using the absolute key
    assert len(suggestions) == 2 # Expecting suggestions only for key1 and key3
    assert suggestions[key1_abs] == constants.DECISION_KEEP # block1 is kept
    assert key2_abs not in suggestions # No suggestion for already deleted block2
    assert suggestions[key3_abs] == constants.DECISION_DELETE # block3 is suggested for deletion

def test_find_md5_duplicates_multiple_groups(md5_analyzer: MD5Analyzer, create_content_block_md5):
    """Test finding multiple distinct groups of duplicates."""
    block_a1 = create_content_block_md5("Content A", file_path="fA1.md", element_id="idA1")
    block_a2 = create_content_block_md5("Content A", file_path="fA2.md", element_id="idA2")
    block_b1 = create_content_block_md5("Content B", file_path="fB1.md", element_id="idB1")
    block_b2 = create_content_block_md5("Content B", file_path="fB2.md", element_id="idB2")
    block_c = create_content_block_md5("Content C", file_path="fC.md", element_id="idC") # Unique
    input_blocks = [block_a1, block_b1, block_c, block_a2, block_b2] # Mix order
    key_a1_abs = get_abs_key(block_a1)
    key_a2_abs = get_abs_key(block_a2)
    key_b1_abs = get_abs_key(block_b1)
    key_b2_abs = get_abs_key(block_b2)
    key_c_abs = get_abs_key(block_c)
    input_decisions = {
        key_a1_abs: constants.DECISION_UNDECIDED, key_a2_abs: constants.DECISION_UNDECIDED,
        key_b1_abs: constants.DECISION_UNDECIDED, key_b2_abs: constants.DECISION_UNDECIDED,
        key_c_abs: constants.DECISION_UNDECIDED,
    }

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert len(groups) == 2
    groups_set = {frozenset(group) for group in groups}
    assert frozenset([block_a1, block_a2]) in groups_set
    assert frozenset([block_b1, block_b2]) in groups_set

    assert len(suggestions) == 4
    # Use absolute keys for assertion
    assert suggestions[key_a1_abs] == constants.DECISION_KEEP
    assert suggestions[key_a2_abs] == constants.DECISION_DELETE
    assert suggestions[key_b1_abs] == constants.DECISION_KEEP
    assert suggestions[key_b2_abs] == constants.DECISION_DELETE
    assert key_c_abs not in suggestions

def test_find_md5_duplicates_three_identical(md5_analyzer: MD5Analyzer, create_content_block_md5):
    """Test a group with three identical blocks."""
    content = "Identical x3"
    block1 = create_content_block_md5(content, file_path="f1.md", element_id="id1")
    block2 = create_content_block_md5(content, file_path="f2.md", element_id="id2")
    block3 = create_content_block_md5(content, file_path="f3.md", element_id="id3")
    input_blocks = [block1, block2, block3]
    key1_abs = get_abs_key(block1)
    key2_abs = get_abs_key(block2)
    key3_abs = get_abs_key(block3)
    input_decisions = {key1_abs: 'undecided', key2_abs: 'undecided', key3_abs: 'undecided'}

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert len(groups) == 1
    assert set(groups[0]) == {block1, block2, block3}
    assert len(suggestions) == 3
    # Use absolute keys for assertion
    assert suggestions[key1_abs] == constants.DECISION_KEEP
    assert suggestions[key2_abs] == constants.DECISION_DELETE
    assert suggestions[key3_abs] == constants.DECISION_DELETE