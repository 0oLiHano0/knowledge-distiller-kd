import pytest
import logging

from knowledge_distiller_kd.processing.block_merger import merge_code_blocks
from knowledge_distiller_kd.core.models import BlockDTO, BlockType, DecisionType

# Helper to create a dummy logger
@pytest.fixture
def logger():
    logger = logging.getLogger("block_merger_test")
    # prevent printing to console
    logger.addHandler(logging.NullHandler())
    return logger

# Helper to create BlockDTO
def make_block(block_id, file_id, block_type, text_content, token_count=1):
    return BlockDTO(
        block_id=block_id,
        file_id=file_id,
        block_type=block_type,
        text_content=text_content,
        analysis_text=text_content,
        token_count=token_count
    )

def test_empty_list_returns_empty(logger):
    result = merge_code_blocks([], {}, logger)
    assert result == [], "Expected empty result for empty input"

def test_no_code_blocks_pass_through(logger):
    b1 = make_block("t1", "fileA", BlockType.TEXT, "just text")
    b2 = make_block("t2", "fileA", BlockType.HEADING, "# title")
    result = merge_code_blocks([b1, b2], {}, logger)
    assert result == [b1, b2], "Non-code blocks should pass through unchanged"

def test_single_complete_code_block_no_merge(logger):
    # A single block containing full fenced code
    code = "```python\nprint('hello')\n```"
    b = make_block("c1", "fileA", BlockType.CODE, code)
    result = merge_code_blocks([b], {}, logger)
    # Should not merge: original block is preserved
    assert len(result) == 1
    out = result[0]
    assert out is b, "Original block should be returned"
    assert out.kd_processing_status == DecisionType.UNDECIDED

def test_fragmented_code_block_merge(logger):
    # Start, content, end fragments
    start = make_block("c1", "fileA", BlockType.CODE, "```python")
    mid = make_block("c2", "fileA", BlockType.CODE, "print(1)")
    end = make_block("c3", "fileA", BlockType.CODE, "```")
    result = merge_code_blocks([start, mid, end], {}, logger)
    # Expect one merged block
    assert len(result) == 1
    merged = result[0]
    # Merged content should not include fences
    assert "```" not in merged.text_content
    # Language metadata
    assert merged.metadata.get("language") == "python"
    # Original fragments marked DELETE
    assert start.kd_processing_status == DecisionType.DELETE
    assert mid.kd_processing_status == DecisionType.DELETE
    assert end.kd_processing_status == DecisionType.DELETE
    # duplicate_of_block_id set correctly
    assert start.duplicate_of_block_id == merged.block_id
    assert mid.duplicate_of_block_id == merged.block_id
    assert end.duplicate_of_block_id == merged.block_id

def test_multiple_independent_fragments_same_file(logger):
    # Two separate code blocks
    s1 = make_block("s1", "fileA", BlockType.CODE, "```js")
    m1 = make_block("m1", "fileA", BlockType.CODE, "console.log('a');")
    e1 = make_block("e1", "fileA", BlockType.CODE, "```")
    s2 = make_block("s2", "fileA", BlockType.CODE, "```python")
    m2 = make_block("m2", "fileA", BlockType.CODE, "print('b')")
    e2 = make_block("e2", "fileA", BlockType.CODE, "```")
    result = merge_code_blocks([s1, m1, e1, s2, m2, e2], {}, logger)
    # Expect two merged blocks
    assert len(result) == 2
    langs = [blk.metadata.get("language") for blk in result]
    assert set(langs) == {"js", "python"}

def test_fragments_different_files_not_merged(logger):
    # Fragments in different files
    s1 = make_block("1", "A", BlockType.CODE, "```python")
    c1 = make_block("2", "A", BlockType.CODE, "a=1")
    e1 = make_block("3", "A", BlockType.CODE, "```")
    s2 = make_block("4", "B", BlockType.CODE, "```python")
    c2 = make_block("5", "B", BlockType.CODE, "b=2")
    e2 = make_block("6", "B", BlockType.CODE, "```")
    result = merge_code_blocks([s1, c1, e1, s2, c2, e2], {}, logger)
    assert len(result) == 2
    assert result[0].file_id == "A"
    assert result[1].file_id == "B"

def test_allowed_gap_merges_across_non_code(logger):
    # One non-code allowed between fragments
    s = make_block("s", "fileA", BlockType.CODE, "```python")
    t = make_block("t", "fileA", BlockType.TEXT, "# comment")
    e = make_block("e", "fileA", BlockType.CODE, "```")
    result = merge_code_blocks([s, t, e], {}, logger)
    assert len(result) == 1
    merged = result[0]
    assert "# comment" in merged.text_content

def test_exceeding_gap_breaks_merge(logger):
    # No non-code allowed (gap=0)
    config = {"processing.merging.max_consecutive_non_code_lines_to_break_merge": 0}
    s = make_block("s", "fileA", BlockType.CODE, "```python")
    t = make_block("t", "fileA", BlockType.TEXT, "# comment")
    e = make_block("e", "fileA", BlockType.CODE, "```")
    result = merge_code_blocks([s, t, e], config, logger)
    # Should not merge: expect three blocks
    assert len(result) == 3
    # None should be marked DELETE
    assert all(b.kd_processing_status == DecisionType.UNDECIDED for b in result)

def test_unclosed_fence_at_end_merges(logger, caplog):
    caplog.set_level(logging.WARNING)
    s = make_block("s", "fileA", BlockType.CODE, "```python")
    c = make_block("c", "fileA", BlockType.CODE, "x=1")
    # No closing fence
    result = merge_code_blocks([s, c], {}, logger)
    assert len(result) == 1
    assert "Unclosed code block detected" in caplog.text

def test_empty_code_block_only_fences(logger):
    # start and end only
    s = make_block("s", "fileA", BlockType.CODE, "```")
    e = make_block("e", "fileA", BlockType.CODE, "```")
    result = merge_code_blocks([s, e], {}, logger)
    assert len(result) == 1
    merged = result[0]
    assert merged.text_content == ""

def test_default_config_key_missing_uses_default(logger):
    # config without key should use max_gap=1
    s = make_block("s", "fileA", BlockType.CODE, "```python")
    t1 = make_block("t1", "fileA", BlockType.TEXT, "a")
    t2 = make_block("t2", "fileA", BlockType.TEXT, "b")
    e = make_block("e", "fileA", BlockType.CODE, "```")
    # Two non-code in between -> exceeds default=1 -> no merge
    result = merge_code_blocks([s, t1, t2, e], {}, logger)
    assert len(result) == 4
    # All original remain
    assert [b.block_id for b in result] == ["s", "t1", "t2", "e"]
