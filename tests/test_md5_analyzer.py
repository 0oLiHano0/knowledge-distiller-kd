# tests/test_md5_analyzer.py
"""
Unit tests for the MD5Analyzer class.
Updated to use ContentBlockDTO from core.models correctly.
Version 2: Fixed fixture to create correct DTO type.
"""

import pytest
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import uuid

# Module to test
from knowledge_distiller_kd.analysis.md5_analyzer import MD5Analyzer
# Import the correct DTO and Enum from core.models
from knowledge_distiller_kd.core.models import ContentBlock as ContentBlockDTO, BlockType
# Import utils needed for tests
from knowledge_distiller_kd.core.utils import create_decision_key
# Use constants consistent with the analyzer/engine
from knowledge_distiller_kd.analysis.md5_analyzer import DECISION_KEEP, DECISION_DELETE, DECISION_UNDECIDED


# --- Test Fixtures ---

@pytest.fixture
def md5_analyzer() -> MD5Analyzer:
    """Provides an instance of the MD5Analyzer."""
    return MD5Analyzer()

# --- Fixture to create ContentBlockDTO for tests ---
@pytest.fixture
def create_content_block_dto_md5():
    """
    Factory fixture to create ContentBlockDTO instances from core.models
    for MD5 tests.
    """
    def _create(
        text: str,
        file_path: str = "test.md", # Default file path for simplicity
        block_id: Optional[str] = None,
        block_type: Union[BlockType, str] = BlockType.TEXT, # Default to TEXT
        metadata: Optional[Dict[str, Any]] = None,
        file_id: str = "test_file_id" # Add default file_id
    ) -> ContentBlockDTO: # Ensure return type is ContentBlockDTO
        # Ensure block_id is unique if not provided
        if block_id is None:
            block_id = str(uuid.uuid4())

        # Ensure block_type is an Enum member
        if isinstance(block_type, str):
             try:
                 # Use the BlockType enum from core.models
                 block_type_enum = BlockType(block_type.lower()) # Use lower for robustness
             except ValueError:
                 # Fallback for common names if direct mapping fails
                 if block_type.lower() == "narrativetext": block_type_enum = BlockType.TEXT
                 elif block_type.lower() == "title": block_type_enum = BlockType.HEADING
                 elif block_type.lower() == "codesnippet": block_type_enum = BlockType.CODE
                 else: raise ValueError(f"Invalid BlockType string '{block_type}' in test setup")
        elif isinstance(block_type, BlockType):
            block_type_enum = block_type
        else:
            raise TypeError(f"block_type must be BlockType enum or string, not {type(block_type)}")

        # Add original_path to metadata, crucial for decision key generation
        if metadata is None:
            metadata = {}
        # Use resolve() to ensure absolute path for key consistency
        metadata['original_path'] = str(Path(file_path).resolve())

        # *** Create the ContentBlockDTO from core.models ***
        return ContentBlockDTO(
            file_id=file_id, # Use provided or default file_id
            text=text,
            block_type=block_type_enum,
            block_id=block_id,
            metadata=metadata
        )
    return _create

# Helper to create absolute decision keys for testing
def get_abs_key_md5(block: ContentBlockDTO) -> str:
    """Helper to generate the decision key using absolute path from metadata."""
    path_str = block.metadata.get('original_path')
    if not path_str:
        raise ValueError(f"Block {block.block_id} missing 'original_path' in metadata for key generation.")
    # Ensure path is resolved before creating key
    return create_decision_key(str(Path(path_str).resolve()), block.block_id, block.block_type.value)


# --- Test Cases ---

def test_find_md5_duplicates_no_blocks(md5_analyzer: MD5Analyzer):
    """Test with an empty list of blocks."""
    groups, suggestions = md5_analyzer.find_md5_duplicates([], {})
    assert groups == []
    assert suggestions == {}

def test_find_md5_duplicates_unique_blocks(md5_analyzer: MD5Analyzer, create_content_block_dto_md5):
    """Test with all unique content blocks."""
    block1 = create_content_block_dto_md5("Unique content 1", file_path="f1.md")
    block2 = create_content_block_dto_md5("Unique content 2", file_path="f2.md")
    input_blocks = [block1, block2]
    key1 = get_abs_key_md5(block1)
    key2 = get_abs_key_md5(block2)
    input_decisions = {key1: DECISION_UNDECIDED, key2: DECISION_UNDECIDED}

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert groups == []
    assert suggestions == {}

def test_find_md5_duplicates_identical_blocks(md5_analyzer: MD5Analyzer, create_content_block_dto_md5):
    """Test with two identical content blocks."""
    content = "Identical content"
    # Ensure different file paths or block IDs for sorting stability
    block1 = create_content_block_dto_md5(content, file_path="file1.md", block_id="id1")
    block2 = create_content_block_dto_md5(content, file_path="file2.md", block_id="id2")
    input_blocks = [block1, block2]
    key1 = get_abs_key_md5(block1)
    key2 = get_abs_key_md5(block2)
    input_decisions = {key1: DECISION_UNDECIDED, key2: DECISION_UNDECIDED}

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert len(groups) == 1
    assert len(groups[0]) == 2
    # Verify the group contains the correct blocks (order depends on sorting)
    assert block1 in groups[0]
    assert block2 in groups[0]

    # Verify suggestions (block1 should be kept, block2 deleted due to sorting)
    assert suggestions.get(key1) == DECISION_KEEP
    assert suggestions.get(key2) == DECISION_DELETE
    assert len(suggestions) == 2

def test_find_md5_duplicates_mixed_blocks(md5_analyzer: MD5Analyzer, create_content_block_dto_md5):
    """Test with a mix of unique and duplicate blocks."""
    content_a = "Content A (Duplicate)"
    content_b = "Content B (Unique)"
    block1 = create_content_block_dto_md5(content_a, file_path="file1.md", block_id="id1")
    block2 = create_content_block_dto_md5(content_b, file_path="file2.md", block_id="id2")
    block3 = create_content_block_dto_md5(content_a, file_path="file3.md", block_id="id3")
    input_blocks = [block1, block2, block3]
    key1 = get_abs_key_md5(block1)
    key2 = get_abs_key_md5(block2)
    key3 = get_abs_key_md5(block3)
    input_decisions = {
        key1: DECISION_UNDECIDED,
        key2: DECISION_UNDECIDED,
        key3: DECISION_UNDECIDED,
    }

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert len(groups) == 1
    assert len(groups[0]) == 2
    assert block1 in groups[0]
    assert block3 in groups[0]
    assert block2 not in groups[0] # Unique block should not be in the group

    # Verify suggestions (block1 kept, block3 deleted)
    assert suggestions.get(key1) == DECISION_KEEP
    assert suggestions.get(key3) == DECISION_DELETE
    assert key2 not in suggestions # No suggestion for unique block
    assert len(suggestions) == 2

def test_find_md5_duplicates_empty_content(md5_analyzer: MD5Analyzer, create_content_block_dto_md5):
    """Test with blocks having empty or whitespace-only content (should match)."""
    block1 = create_content_block_dto_md5("", file_path="file1.md", block_id="id_empty")
    block2 = create_content_block_dto_md5("   \n\t  ", file_path="file2.md", block_id="id_space") # Whitespace only
    input_blocks = [block1, block2]
    key1 = get_abs_key_md5(block1)
    key2 = get_abs_key_md5(block2)
    input_decisions = {key1: DECISION_UNDECIDED, key2: DECISION_UNDECIDED}

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert len(groups) == 1 # Empty/whitespace should hash the same after strip()
    assert len(groups[0]) == 2
    assert block1 in groups[0]
    assert block2 in groups[0]

    assert suggestions.get(key1) == DECISION_KEEP
    assert suggestions.get(key2) == DECISION_DELETE
    assert len(suggestions) == 2

def test_find_md5_duplicates_normalize_text(md5_analyzer: MD5Analyzer, create_content_block_dto_md5):
    """Test that normalization (whitespace) happens before hashing."""
    # MD5Analyzer uses strip(), so only leading/trailing whitespace matters for identity
    block1 = create_content_block_dto_md5("  Text with space  \n", block_id="id1", file_path="f1.md")
    block2 = create_content_block_dto_md5("Text with space", block_id="id2", file_path="f2.md") # Should match after strip
    input_blocks = [block1, block2]
    key1 = get_abs_key_md5(block1)
    key2 = get_abs_key_md5(block2)
    input_decisions = {key1: DECISION_UNDECIDED, key2: DECISION_UNDECIDED}

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert len(groups) == 1
    assert len(groups[0]) == 2
    assert block1 in groups[0]
    assert block2 in groups[0]

    assert suggestions.get(key1) == DECISION_KEEP # Assuming f1.md sorts before f2.md
    assert suggestions.get(key2) == DECISION_DELETE
    assert len(suggestions) == 2

def test_find_md5_duplicates_pre_deleted_blocks(md5_analyzer: MD5Analyzer, create_content_block_dto_md5):
    """Test that blocks already marked as DELETE are ignored in suggestions."""
    content = "Identical content"
    block1 = create_content_block_dto_md5(content, file_path="file1.md", block_id="id1")
    block2 = create_content_block_dto_md5(content, file_path="file2.md", block_id="id2")
    block3 = create_content_block_dto_md5(content, file_path="file3.md", block_id="id3")
    input_blocks = [block1, block2, block3]
    key1 = get_abs_key_md5(block1)
    key2 = get_abs_key_md5(block2)
    key3 = get_abs_key_md5(block3)
    input_decisions = {
        key1: DECISION_UNDECIDED,
        key2: DECISION_DELETE, # Pre-deleted
        key3: DECISION_UNDECIDED,
    }

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    # The group should contain only non-deleted blocks with the same hash
    assert len(groups) == 1
    assert len(groups[0]) == 2 # Only block1 and block3 should be hashed and grouped
    assert block1 in groups[0]
    assert block3 in groups[0]
    assert block2 not in groups[0] # Because it was skipped before hashing

    # Suggestions: block1 kept, block3 deleted. block2 is untouched.
    assert suggestions.get(key1) == DECISION_KEEP
    assert suggestions.get(key3) == DECISION_DELETE
    assert key2 not in suggestions # No suggestion for pre-deleted block
    assert len(suggestions) == 2

def test_find_md5_duplicates_multiple_groups(md5_analyzer: MD5Analyzer, create_content_block_dto_md5):
    """Test finding multiple distinct groups of duplicates."""
    block_a1 = create_content_block_dto_md5("Content A", file_path="fA1.md", block_id="idA1")
    block_a2 = create_content_block_dto_md5("Content A", file_path="fA2.md", block_id="idA2")
    block_b1 = create_content_block_dto_md5("Content B", file_path="fB1.md", block_id="idB1")
    block_b2 = create_content_block_dto_md5("Content B", file_path="fB2.md", block_id="idB2")
    block_c = create_content_block_dto_md5("Content C", file_path="fC.md", block_id="idC") # Unique
    input_blocks = [block_a1, block_b1, block_c, block_a2, block_b2] # Mix order
    key_a1 = get_abs_key_md5(block_a1); key_a2 = get_abs_key_md5(block_a2)
    key_b1 = get_abs_key_md5(block_b1); key_b2 = get_abs_key_md5(block_b2)
    key_c = get_abs_key_md5(block_c)
    input_decisions = {
        key_a1: DECISION_UNDECIDED, key_a2: DECISION_UNDECIDED,
        key_b1: DECISION_UNDECIDED, key_b2: DECISION_UNDECIDED,
        key_c: DECISION_UNDECIDED,
    }

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert len(groups) == 2 # Expecting two groups (A and B)
    # Find group A
    group_a = next((g for g in groups if block_a1 in g), None)
    assert group_a is not None and len(group_a) == 2 and block_a2 in group_a
    # Find group B
    group_b = next((g for g in groups if block_b1 in g), None)
    assert group_b is not None and len(group_b) == 2 and block_b2 in group_b

    # Verify suggestions
    assert suggestions.get(key_a1) == DECISION_KEEP # Assuming fA1 sorts before fA2
    assert suggestions.get(key_a2) == DECISION_DELETE
    assert suggestions.get(key_b1) == DECISION_KEEP # Assuming fB1 sorts before fB2
    assert suggestions.get(key_b2) == DECISION_DELETE
    assert key_c not in suggestions
    assert len(suggestions) == 4

def test_find_md5_duplicates_three_identical(md5_analyzer: MD5Analyzer, create_content_block_dto_md5):
    """Test a group with three identical blocks."""
    content = "Identical x3"
    block1 = create_content_block_dto_md5(content, file_path="f1.md", block_id="id1")
    block2 = create_content_block_dto_md5(content, file_path="f2.md", block_id="id2")
    block3 = create_content_block_dto_md5(content, file_path="f3.md", block_id="id3")
    input_blocks = [block1, block2, block3]
    key1 = get_abs_key_md5(block1); key2 = get_abs_key_md5(block2); key3 = get_abs_key_md5(block3)
    input_decisions = {key1: DECISION_UNDECIDED, key2: DECISION_UNDECIDED, key3: DECISION_UNDECIDED}

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert len(groups) == 1
    assert len(groups[0]) == 3
    assert block1 in groups[0] and block2 in groups[0] and block3 in groups[0]

    # Verify suggestions (keep first, delete others)
    assert suggestions.get(key1) == DECISION_KEEP
    assert suggestions.get(key2) == DECISION_DELETE
    assert suggestions.get(key3) == DECISION_DELETE
    assert len(suggestions) == 3

def test_find_md5_duplicates_skip_headings(md5_analyzer: MD5Analyzer, create_content_block_dto_md5):
    """Test that heading blocks are skipped."""
    block1 = create_content_block_dto_md5("Heading Content", block_type=BlockType.HEADING, block_id="id1")
    block2 = create_content_block_dto_md5("Heading Content", block_type=BlockType.HEADING, block_id="id2")
    block3 = create_content_block_dto_md5("Regular Text", block_type=BlockType.TEXT, block_id="id3")
    input_blocks = [block1, block2, block3]
    key1 = get_abs_key_md5(block1); key2 = get_abs_key_md5(block2); key3 = get_abs_key_md5(block3)
    input_decisions = {key1: DECISION_UNDECIDED, key2: DECISION_UNDECIDED, key3: DECISION_UNDECIDED}

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert groups == [] # Headings should be skipped, no duplicates found
    assert suggestions == {}

def test_find_md5_duplicates_block_type_matters(md5_analyzer: MD5Analyzer, create_content_block_dto_md5):
    """Test that blocks with same text but different types are not duplicates."""
    content = "Same text"
    block1 = create_content_block_dto_md5(content, block_type=BlockType.TEXT, block_id="id1")
    block2 = create_content_block_dto_md5(content, block_type=BlockType.CODE, block_id="id2") # Different type
    input_blocks = [block1, block2]
    key1 = get_abs_key_md5(block1); key2 = get_abs_key_md5(block2)
    input_decisions = {key1: DECISION_UNDECIDED, key2: DECISION_UNDECIDED}

    groups, suggestions = md5_analyzer.find_md5_duplicates(input_blocks, input_decisions)

    assert groups == [] # Should not be duplicates due to different type
    assert suggestions == {}

