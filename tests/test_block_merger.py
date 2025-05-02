# tests/test_block_merger.py
"""
Tests for the block merging logic, specifically for code blocks.
"""

import pytest
from typing import List, Type

# Import necessary classes from the core modules
# Assuming ContentBlock and Element types are needed for creating test data
from knowledge_distiller_kd.core.document_processor import ContentBlock
from unstructured.documents.elements import Element, NarrativeText, Text, CodeSnippet, Title

# Import the function to be tested
from knowledge_distiller_kd.core.block_merger import merge_code_blocks


# Helper fixture to create ContentBlock instances easily in tests
@pytest.fixture
def create_block():
    """Factory fixture to create ContentBlock instances."""
    def _create(text: str, file_path: str = "test.md", element_type: Type[Element] = NarrativeText, element_id_suffix: str = "") -> ContentBlock:
        # Use a simple counter or random string for unique IDs in tests if needed
        import uuid
        element_id = f"test_{element_id_suffix}_{uuid.uuid4()}"
        element = element_type(text=text, element_id=element_id)
        # Instantiate ContentBlock - it will call _infer_block_type and _normalize_text
        cb = ContentBlock(element=element, file_path=file_path)
        # Note: We let ContentBlock infer its type. Tests should check the state *after* inference.
        return cb
    return _create

# --- Test Cases ---

def test_merge_no_code_blocks(create_block):
    """Test merging when there are no code blocks."""
    input_blocks = [
        create_block("# Title 1", element_type=Title, element_id_suffix="title1"),
        create_block("Paragraph 1.", element_type=NarrativeText, element_id_suffix="p1"),
        create_block("Paragraph 2.", element_type=NarrativeText, element_id_suffix="p2"),
    ]
    expected_blocks = input_blocks # Expect no changes
    merged_blocks = merge_code_blocks(input_blocks)
    assert len(merged_blocks) == len(expected_blocks)
    # Check if the objects are the same (or have the same relevant attributes)
    for i in range(len(merged_blocks)):
        assert merged_blocks[i].original_text == expected_blocks[i].original_text
        assert merged_blocks[i].block_id == expected_blocks[i].block_id
        assert merged_blocks[i].block_type == expected_blocks[i].block_type

def test_merge_single_fragmented_code_block(create_block):
    """Test merging a typical fragmented code block."""
    input_blocks = [
        create_block("# Title", element_type=Title, element_id_suffix="title"),
        create_block("```python", element_type=NarrativeText, element_id_suffix="cb_start"), # Intentionally NarrativeText
        create_block("print('Hello')", element_type=NarrativeText, element_id_suffix="cb_content"),
        create_block("```", element_type=Text, element_id_suffix="cb_end"), # Intentionally Text
        create_block("After code.", element_type=NarrativeText, element_id_suffix="after"),
    ]
    merged_blocks = merge_code_blocks(input_blocks)

    assert len(merged_blocks) == 3 # Title, Merged Code Block, After code
    assert merged_blocks[0].block_type == "Title"
    assert merged_blocks[2].block_type == "NarrativeText"

    # Check the merged code block
    merged_code = merged_blocks[1]
    assert merged_code.block_type == "CodeSnippet" # Should be forced to CodeSnippet
    # Check original text joining (assuming newline)
    expected_original_text = "```python\nprint('Hello')\n```"
    assert merged_code.original_text == expected_original_text

    # Check the analysis text (should be pure code)
    expected_analysis_text = "print('Hello')"
    assert merged_code.analysis_text == expected_analysis_text
    # Check ID and file path (should likely take from the starting block)
    assert merged_code.block_id == input_blocks[1].block_id
    assert merged_code.file_path == input_blocks[1].file_path

def test_merge_multiple_code_blocks(create_block):
    """Test merging multiple code blocks sequentially."""
    input_blocks = [
        create_block("```python", element_type=Text, element_id_suffix="cb1_start"),
        create_block("code1", element_type=NarrativeText, element_id_suffix="cb1_content"),
        create_block("```", element_type=Text, element_id_suffix="cb1_end"),
        create_block("Some text", element_type=NarrativeText, element_id_suffix="p1"),
        create_block("```javascript", element_type=NarrativeText, element_id_suffix="cb2_start"),
        create_block("code2", element_type=NarrativeText, element_id_suffix="cb2_content"),
        create_block("```", element_type=Text, element_id_suffix="cb2_end"),
    ]
    merged_blocks = merge_code_blocks(input_blocks)

    assert len(merged_blocks) == 3 # Code1, Text, Code2
    assert merged_blocks[0].block_type == "CodeSnippet"
    assert merged_blocks[0].analysis_text == "code1"
    assert merged_blocks[1].block_type == "NarrativeText"
    assert merged_blocks[2].block_type == "CodeSnippet"
    assert merged_blocks[2].analysis_text == "code2"

def test_merge_unclosed_code_block(create_block, caplog):
    """Test merging when a code block is not properly closed at the end."""
    input_blocks = [
        create_block("Paragraph.", element_type=NarrativeText, element_id_suffix="p1"),
        create_block("```python", element_type=NarrativeText, element_id_suffix="cb_start"),
        create_block("unclosed_code", element_type=NarrativeText, element_id_suffix="cb_content"),
        # No closing ```
    ]
    merged_blocks = merge_code_blocks(input_blocks)

    # Expect the unclosed parts to be returned as original blocks, with a warning
    assert len(merged_blocks) == 3
    assert merged_blocks[0].block_type == "NarrativeText"
    # The original types of the unclosed parts should be preserved *after* ContentBlock init
    # ==================== Fix Assertion ====================
    # ContentBlock._infer_block_type correctly identifies ```python as CodeSnippet
    assert merged_blocks[1].block_type == "CodeSnippet"
    # ======================================================
    assert merged_blocks[1].original_text == "```python"
    assert merged_blocks[2].block_type == "NarrativeText"
    assert merged_blocks[2].original_text == "unclosed_code"
    # Check for warning log
    assert "Unclosed code block detected" in caplog.text

def test_merge_only_start_fence(create_block, caplog):
    """Test when only a start fence exists."""
    input_blocks = [
        create_block("```python", element_type=NarrativeText, element_id_suffix="cb_start"),
    ]
    merged_blocks = merge_code_blocks(input_blocks)
    assert len(merged_blocks) == 1
    # ==================== Fix Assertion ====================
    # Check attributes, as the object identity might change if ContentBlock modified type
    assert merged_blocks[0].original_text == input_blocks[0].original_text
    assert merged_blocks[0].block_type == "CodeSnippet" # Type inference will make it CodeSnippet
    # ======================================================
    assert "Unclosed code block detected" in caplog.text

def test_merge_start_and_end_fence_only(create_block):
    """Test merging a code block with only start and end fences (empty content)."""
    input_blocks = [
        create_block("```", element_type=Text, element_id_suffix="cb_start"),
        create_block("```", element_type=Text, element_id_suffix="cb_end"),
    ]
    merged_blocks = merge_code_blocks(input_blocks)
    # ==================== Fix Assertion ====================
    assert len(merged_blocks) == 1 # Should now merge correctly
    # ======================================================
    merged_code = merged_blocks[0]
    assert merged_code.block_type == "CodeSnippet"
    assert merged_code.original_text == "```\n```" # Check original text joining
    assert merged_code.analysis_text == "" # Analysis text should be empty

def test_merge_ignores_fences_within_text(create_block):
    """Test that fences within regular text are not treated as code blocks."""
    input_blocks = [
        create_block("Text with ``` inside.", element_type=NarrativeText, element_id_suffix="p1"),
        create_block("Another ```python example.", element_type=NarrativeText, element_id_suffix="p2"),
    ]
    expected_blocks = input_blocks
    merged_blocks = merge_code_blocks(input_blocks)
    assert len(merged_blocks) == len(expected_blocks)
    for i in range(len(merged_blocks)):
        assert merged_blocks[i].original_text == expected_blocks[i].original_text
        assert merged_blocks[i].block_type == expected_blocks[i].block_type

# Add more tests as needed:
# - Code blocks with different language identifiers
# - Nested structures (though Markdown doesn't typically nest code blocks like this)
# - Edge cases with whitespace around fences
