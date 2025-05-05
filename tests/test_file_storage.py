# tests/test_file_storage.py
"""
Unit tests for the refactored FileStorage implementation using pytest.
Focuses on testing the adherence to the StorageInterface contract
and correct handling of data persistence to JSON files using DTOs.
(Version corresponding to Task KD-REFACTOR-001)
"""

import datetime
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Generator

import pytest

# Assume DTOs and Interface are in the correct path relative to the project root
# Adjust imports based on your project structure if necessary
# Note: Assuming 'knowledge_distiller_kd' is the root package accessible in PYTHONPATH
from knowledge_distiller_kd.core.models import (
    AnalysisResult, AnalysisType, BlockType, ContentBlock, DecisionType,
    FileRecord, UserDecision
)
# Import the class to be tested
from knowledge_distiller_kd.storage.file_storage import FileStorage
# Import the interface for type checking and conformity tests
from knowledge_distiller_kd.storage.storage_interface import StorageInterface
# Import potential custom errors if defined
# from knowledge_distiller_kd.core.exceptions import StorageError

# --- Test Fixtures ---

@pytest.fixture
def base_test_path(tmp_path: Path) -> Path:
    """Provides a temporary directory for test file storage."""
    storage_dir = tmp_path / "test_storage"
    # FileStorage.initialize() should create it
    return storage_dir

@pytest.fixture
def storage(base_test_path: Path) -> Generator[FileStorage, None, None]:
    """Provides an initialized FileStorage instance for testing."""
    # *** Correctly initializes FileStorage with base_path ***
    fs = FileStorage(base_path=base_test_path)
    fs.initialize() # Ensure storage is set up before tests run
    yield fs
    # No explicit finalize action needed for basic file storage unless specified
    # fs.finalize()

@pytest.fixture
def sample_filepath() -> str:
    """Provides a sample file path string."""
    # Doesn't need to exist for registration testing
    return "/path/to/document1.txt" # Use string as per interface

@pytest.fixture
def sample_filepath_obj(sample_filepath: str) -> Path:
    """Provides the sample file path as a Path object (for internal checks)."""
    # Note: The interface takes str, but internally Path might be used.
    return Path(sample_filepath)


@pytest.fixture
def sample_file_id(storage: FileStorage, sample_filepath: str) -> str:
    """Registers a sample file and returns its ID."""
    return storage.register_file(sample_filepath)

@pytest.fixture
def sample_block_1(sample_file_id: str) -> ContentBlock:
    """Provides a sample ContentBlock."""
    return ContentBlock(
        file_id=sample_file_id,
        text="This is the first block.",
        block_type=BlockType.TEXT,
        metadata={"page": 1}
        # block_id is auto-generated
    )

@pytest.fixture
def sample_block_2(sample_file_id: str) -> ContentBlock:
    """Provides another sample ContentBlock."""
    return ContentBlock(
        file_id=sample_file_id,
        text="This is the second block, slightly different.",
        block_type=BlockType.TEXT,
        metadata={"page": 1, "style": "bold"}
        # block_id is auto-generated
    )

@pytest.fixture
def sample_block_3(sample_file_id: str) -> ContentBlock:
    """Provides a third sample ContentBlock of a different type."""
    return ContentBlock(
        file_id=sample_file_id,
        text="## Section Heading",
        block_type=BlockType.HEADING,
        metadata={"level": 2}
        # block_id is auto-generated
    )

@pytest.fixture
def sample_analysis_result_1(sample_block_1: ContentBlock, sample_block_2: ContentBlock) -> AnalysisResult:
    """Provides a sample AnalysisResult."""
    return AnalysisResult(
        block_id_1=sample_block_1.block_id,
        block_id_2=sample_block_2.block_id,
        analysis_type=AnalysisType.SEMANTIC_SIMILARITY,
        score=0.85
        # result_id is auto-generated
    )

@pytest.fixture
def sample_analysis_result_2(sample_block_1: ContentBlock, sample_block_3: ContentBlock) -> AnalysisResult:
    """Provides another sample AnalysisResult of a different type."""
    # Assume block 1 and 3 are identical for MD5 testing
    # In a real scenario, their text would be identical
    return AnalysisResult(
        block_id_1=sample_block_1.block_id,
        block_id_2=sample_block_3.block_id, # Using block 3 for variety
        analysis_type=AnalysisType.MD5_DUPLICATE,
        score=1.0 # Typically score is not used for MD5, but can be 1.0 for match
        # result_id is auto-generated
    )


@pytest.fixture
def sample_decision_1(sample_analysis_result_1: AnalysisResult) -> UserDecision:
    """Provides a sample UserDecision."""
    # Ensure timestamp is fixed for predictable testing if needed, otherwise use default
    now = datetime.datetime.now(datetime.timezone.utc)
    return UserDecision(
        block_id_1=sample_analysis_result_1.block_id_1,
        block_id_2=sample_analysis_result_1.block_id_2,
        analysis_type=sample_analysis_result_1.analysis_type,
        decision=DecisionType.MARK_SIMILAR,
        notes="Looks very similar indeed.",
        timestamp=now
        # decision_id is auto-generated
    )

@pytest.fixture
def sample_decision_2_undecided(sample_analysis_result_2: AnalysisResult) -> UserDecision:
    """Provides a sample UserDecision that is undecided."""
    # Note: We typically wouldn't explicitly save UNDECIDED unless overriding
    # a previous decision. This fixture is for testing retrieval logic.
    return UserDecision(
        block_id_1=sample_analysis_result_2.block_id_1,
        block_id_2=sample_analysis_result_2.block_id_2,
        analysis_type=sample_analysis_result_2.analysis_type,
        decision=DecisionType.UNDECIDED
        # decision_id is auto-generated
    )


# --- Helper Functions ---

def read_json_file(path: Path) -> Any:
    """Reads and parses a JSON file."""
    if not path.exists():
        return None # Or raise error, depending on expected test behavior
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content:
                return {} # Return empty dict for empty file, consistent with initial state
            return json.loads(content)
    except json.JSONDecodeError:
        pytest.fail(f"Failed to decode JSON from {path}")
    except Exception as e:
        pytest.fail(f"Failed to read file {path}: {e}")


# --- Test Cases (Aligned with StorageInterface) ---

def test_storage_conforms_to_interface():
    """Check if FileStorage correctly implements StorageInterface."""
    assert issubclass(FileStorage, StorageInterface)
    # Check for presence of all methods (less critical with ABC)
    required_methods = [
        '__init__', 'initialize', 'register_file', 'get_file_record', 'list_files',
        'save_blocks', 'get_block', 'get_blocks_by_file', 'get_blocks_for_analysis',
        'save_analysis_result', 'get_analysis_results', 'save_user_decision',
        'get_user_decisions', 'get_undecided_pairs', 'finalize'
    ]
    for method in required_methods:
        assert hasattr(FileStorage, method) and callable(getattr(FileStorage, method))

def test_initialize_creates_directory_and_files(base_test_path: Path):
    """Test if initialize creates the base directory and expected JSON files."""
    assert not base_test_path.exists() # Ensure it doesn't exist initially

    # *** Initialize WITHOUT using the fixture to test initialize itself ***
    storage_init_test = FileStorage(base_path=base_test_path)
    storage_init_test.initialize()

    assert base_test_path.is_dir()
    # Check if essential JSON files are created (use internal names for robustness)
    assert (base_test_path / storage_init_test._metadata_filename).is_file()
    assert (base_test_path / storage_init_test._blocks_filename).is_file()
    assert (base_test_path / storage_init_test._results_filename).is_file()
    assert (base_test_path / storage_init_test._decisions_filename).is_file()

    # Check initial content (should be empty or default structure)
    meta_content = read_json_file(base_test_path / storage_init_test._metadata_filename)
    assert meta_content == {"files": {}, "path_to_id": {}} # Check against expected default

    blocks_content = read_json_file(base_test_path / storage_init_test._blocks_filename)
    assert blocks_content == {}

    results_content = read_json_file(base_test_path / storage_init_test._results_filename)
    assert results_content == {}

    decisions_content = read_json_file(base_test_path / storage_init_test._decisions_filename)
    assert decisions_content == {}

def test_initialize_is_idempotent(storage: FileStorage, base_test_path: Path):
    """Test if calling initialize multiple times is safe."""
    # First initialization happened in the 'storage' fixture
    metadata_path = base_test_path / storage._metadata_filename
    assert metadata_path.exists()
    initial_mtime = metadata_path.stat().st_mtime
    initial_content = read_json_file(metadata_path)

    # Call initialize again on the already initialized instance
    try:
        storage.initialize() # Should do nothing if already initialized
    except Exception as e:
        pytest.fail(f"Second call to initialize failed: {e}")

    # Check if directory still exists and files weren't necessarily overwritten
    assert base_test_path.is_dir()
    assert metadata_path.exists()
    # Content should remain the same
    assert read_json_file(metadata_path) == initial_content


def test_register_file_new(storage: FileStorage, sample_filepath: str, base_test_path: Path):
    """Test registering a new file path."""
    # *** Use the 'storage' fixture which is already initialized ***
    file_id = storage.register_file(sample_filepath)

    assert isinstance(file_id, str)
    assert len(file_id) > 0 # Basic check for non-empty string

    # Verify persistence in metadata.json
    meta_content = read_json_file(base_test_path / storage._metadata_filename)
    assert "files" in meta_content
    assert file_id in meta_content["files"]
    saved_record_dict = meta_content["files"][file_id]
    assert saved_record_dict["original_path"] == sample_filepath
    assert "registration_time" in saved_record_dict
    # Check if time is a valid ISO format string
    try:
        datetime.datetime.fromisoformat(saved_record_dict["registration_time"])
    except ValueError:
        pytest.fail("registration_time is not a valid ISO format string")

    # Verify get_file_record
    record = storage.get_file_record(file_id)
    assert record is not None
    assert isinstance(record, FileRecord)
    assert record.file_id == file_id
    assert record.original_path == sample_filepath
    assert isinstance(record.registration_time, datetime.datetime)
    # Ensure timezone awareness (expecting UTC from implementation)
    assert record.registration_time.tzinfo is not None

def test_register_file_existing(storage: FileStorage, sample_filepath: str):
    """Test registering the same file path again returns the same ID."""
    file_id_1 = storage.register_file(sample_filepath)
    file_id_2 = storage.register_file(sample_filepath)

    assert file_id_1 == file_id_2

def test_get_file_record_not_found(storage: FileStorage):
    """Test getting a file record for a non-existent file ID."""
    non_existent_id = str(uuid.uuid4())
    record = storage.get_file_record(non_existent_id)
    assert record is None

def test_list_files_empty(storage: FileStorage):
    """Test listing files when none are registered."""
    files = storage.list_files()
    assert isinstance(files, list)
    assert len(files) == 0

def test_list_files_multiple(storage: FileStorage):
    """Test listing files after registering multiple."""
    path1 = "/path/doc1.txt"
    path2 = "/path/doc2.md"
    id1 = storage.register_file(path1)
    id2 = storage.register_file(path2)

    files = storage.list_files()
    assert isinstance(files, list)
    assert len(files) == 2

    # Check if the returned objects are FileRecords and contain the correct data
    found_ids = {record.file_id for record in files}
    assert found_ids == {id1, id2}
    for record in files:
        assert isinstance(record, FileRecord)
        if record.file_id == id1:
            assert record.original_path == path1
        else:
            assert record.original_path == path2


def test_save_blocks_new(storage: FileStorage, sample_file_id: str, sample_block_1: ContentBlock, base_test_path: Path):
    """Test saving a single new block."""
    storage.save_blocks(sample_file_id, [sample_block_1])

    # Verify persistence in blocks.json
    blocks_content = read_json_file(base_test_path / storage._blocks_filename)
    assert sample_block_1.block_id in blocks_content
    saved_block_dict = blocks_content[sample_block_1.block_id]

    # Compare serialized data (DTO's to_dict handles internal serialization)
    expected_dict = sample_block_1.to_dict()
    assert saved_block_dict == expected_dict

    # Verify retrieval via get_block
    retrieved_block = storage.get_block(sample_block_1.block_id)
    assert retrieved_block is not None
    # Dataclasses compare equal if fields are equal
    assert retrieved_block == sample_block_1

def test_save_blocks_multiple(storage: FileStorage, sample_file_id: str, sample_block_1: ContentBlock, sample_block_2: ContentBlock):
    """Test saving multiple blocks at once."""
    blocks_to_save = [sample_block_1, sample_block_2]
    storage.save_blocks(sample_file_id, blocks_to_save)

    retrieved_1 = storage.get_block(sample_block_1.block_id)
    retrieved_2 = storage.get_block(sample_block_2.block_id)

    assert retrieved_1 == sample_block_1
    assert retrieved_2 == sample_block_2

def test_save_blocks_update(storage: FileStorage, sample_file_id: str, sample_block_1: ContentBlock):
    """Test updating an existing block by saving it again."""
    storage.save_blocks(sample_file_id, [sample_block_1])

    # Modify the block in memory (create a new object with same ID)
    updated_block = ContentBlock(
        block_id=sample_block_1.block_id, # Same ID
        file_id=sample_file_id,
        text="This is the updated first block.",
        block_type=BlockType.TEXT, # Can change type too if needed
        metadata={"page": 2, "status": "updated"}
    )
    storage.save_blocks(sample_file_id, [updated_block])

    # Retrieve and verify update
    retrieved_block = storage.get_block(sample_block_1.block_id)
    assert retrieved_block is not None
    assert retrieved_block.block_id == updated_block.block_id # Same ID
    assert retrieved_block.text == updated_block.text
    assert retrieved_block.metadata == updated_block.metadata
    assert retrieved_block.block_type == updated_block.block_type
    # Ensure it's different from the original object state
    assert retrieved_block != sample_block_1

def test_get_block_not_found(storage: FileStorage):
    """Test getting a block that doesn't exist."""
    non_existent_id = str(uuid.uuid4())
    block = storage.get_block(non_existent_id)
    assert block is None

def test_get_blocks_by_file(storage: FileStorage, sample_file_id: str, sample_block_1: ContentBlock, sample_block_2: ContentBlock, sample_block_3: ContentBlock):
    """Test retrieving all blocks for a specific file."""
    # Register another file and block to ensure filtering works
    other_filepath = "/path/other_doc.txt"
    other_file_id = storage.register_file(other_filepath)
    other_block = ContentBlock(file_id=other_file_id, text="Other file content", block_type=BlockType.TEXT)

    storage.save_blocks(sample_file_id, [sample_block_1, sample_block_2, sample_block_3])
    storage.save_blocks(other_file_id, [other_block])

    blocks = storage.get_blocks_by_file(sample_file_id)
    assert isinstance(blocks, list)
    # Order might not be guaranteed, compare sets of IDs or sort before comparing objects
    assert len(blocks) == 3
    block_ids_retrieved = {b.block_id for b in blocks}
    expected_ids = {sample_block_1.block_id, sample_block_2.block_id, sample_block_3.block_id}
    assert block_ids_retrieved == expected_ids

    # Verify content by comparing sorted lists of objects
    retrieved_blocks_sorted = sorted(blocks, key=lambda b: b.block_id)
    expected_blocks_sorted = sorted([sample_block_1, sample_block_2, sample_block_3], key=lambda b: b.block_id)
    assert retrieved_blocks_sorted == expected_blocks_sorted


    # Check retrieval for the other file
    other_blocks = storage.get_blocks_by_file(other_file_id)
    assert len(other_blocks) == 1
    assert other_blocks[0] == other_block

def test_get_blocks_by_file_not_found(storage: FileStorage):
    """Test retrieving blocks for a non-existent file ID."""
    non_existent_id = str(uuid.uuid4())
    blocks = storage.get_blocks_by_file(non_existent_id)
    assert isinstance(blocks, list)
    assert len(blocks) == 0

def test_get_blocks_for_analysis_no_filter(storage: FileStorage, sample_file_id: str, sample_block_1: ContentBlock, sample_block_2: ContentBlock):
    """Test retrieving blocks for analysis with no filter (should return all)."""
    storage.save_blocks(sample_file_id, [sample_block_1, sample_block_2])
    blocks = storage.get_blocks_for_analysis() # No filter
    assert len(blocks) == 2
    block_ids_retrieved = {b.block_id for b in blocks}
    expected_ids = {sample_block_1.block_id, sample_block_2.block_id}
    assert block_ids_retrieved == expected_ids

def test_get_blocks_for_analysis_filter_by_file(storage: FileStorage, sample_file_id: str, sample_block_1: ContentBlock, sample_block_2: ContentBlock):
    """Test filtering blocks for analysis by file_id."""
     # Register another file and block
    other_filepath = "/path/other_doc.txt"
    other_file_id = storage.register_file(other_filepath)
    other_block = ContentBlock(file_id=other_file_id, text="Other file content", block_type=BlockType.TEXT)

    storage.save_blocks(sample_file_id, [sample_block_1, sample_block_2])
    storage.save_blocks(other_file_id, [other_block])

    blocks = storage.get_blocks_for_analysis(filter_criteria={"file_id": sample_file_id})
    assert len(blocks) == 2
    block_ids_retrieved = {b.block_id for b in blocks}
    expected_ids = {sample_block_1.block_id, sample_block_2.block_id}
    assert block_ids_retrieved == expected_ids

def test_get_blocks_for_analysis_filter_by_type(storage: FileStorage, sample_file_id: str, sample_block_1: ContentBlock, sample_block_3: ContentBlock):
    """Test filtering blocks for analysis by block_type."""
    storage.save_blocks(sample_file_id, [sample_block_1, sample_block_3]) # TEXT and HEADING
    # Use the Enum member for filtering
    blocks = storage.get_blocks_for_analysis(filter_criteria={"block_type": BlockType.HEADING})
    assert len(blocks) == 1
    assert blocks[0] == sample_block_3

    # Test filtering by string value (should also work if implemented)
    blocks_str = storage.get_blocks_for_analysis(filter_criteria={"block_type": "heading"})
    assert len(blocks_str) == 1
    assert blocks_str[0] == sample_block_3


def test_get_blocks_for_analysis_filter_combined(storage: FileStorage, sample_file_id: str, sample_block_1: ContentBlock, sample_block_2: ContentBlock, sample_block_3: ContentBlock):
    """Test filtering blocks for analysis by multiple criteria."""
    # Add another text block to a different file
    other_file_id = storage.register_file("/path/other.txt")
    other_text_block = ContentBlock(file_id=other_file_id, text="Another text block", block_type=BlockType.TEXT)

    storage.save_blocks(sample_file_id, [sample_block_1, sample_block_2, sample_block_3])
    storage.save_blocks(other_file_id, [other_text_block])

    # Filter for TEXT blocks only in the sample_file_id
    blocks = storage.get_blocks_for_analysis(filter_criteria={
        "file_id": sample_file_id,
        "block_type": BlockType.TEXT # Use Enum
    })
    assert len(blocks) == 2
    block_ids_retrieved = {b.block_id for b in blocks}
    expected_ids = {sample_block_1.block_id, sample_block_2.block_id}
    assert block_ids_retrieved == expected_ids
    for block in blocks:
        assert block.block_type == BlockType.TEXT
        assert block.file_id == sample_file_id


def test_save_analysis_result_new(storage: FileStorage, sample_analysis_result_1: AnalysisResult, base_test_path: Path):
    """Test saving a new analysis result."""
    analysis_type = sample_analysis_result_1.analysis_type
    storage.save_analysis_result(analysis_type, [sample_analysis_result_1])

    # Verify persistence in results.json
    results_content = read_json_file(base_test_path / storage._results_filename)
    analysis_type_str = analysis_type.value
    assert analysis_type_str in results_content
    assert sample_analysis_result_1.result_id in results_content[analysis_type_str]

    saved_result_dict = results_content[analysis_type_str][sample_analysis_result_1.result_id]
    # Compare serialized data
    expected_dict = sample_analysis_result_1.to_dict()
    assert saved_result_dict == expected_dict

    # Verify retrieval
    retrieved_results = storage.get_analysis_results(analysis_type)
    assert len(retrieved_results) == 1
    retrieved = retrieved_results[0]
    assert isinstance(retrieved, AnalysisResult)
    assert retrieved == sample_analysis_result_1

def test_save_analysis_result_multiple_types(storage: FileStorage, sample_analysis_result_1: AnalysisResult, sample_analysis_result_2: AnalysisResult):
    """Test saving results for different analysis types."""
    storage.save_analysis_result(sample_analysis_result_1.analysis_type, [sample_analysis_result_1])
    storage.save_analysis_result(sample_analysis_result_2.analysis_type, [sample_analysis_result_2])

    results_semantic = storage.get_analysis_results(AnalysisType.SEMANTIC_SIMILARITY)
    assert len(results_semantic) == 1
    assert results_semantic[0] == sample_analysis_result_1

    results_md5 = storage.get_analysis_results(AnalysisType.MD5_DUPLICATE)
    assert len(results_md5) == 1
    assert results_md5[0] == sample_analysis_result_2

def test_save_analysis_result_update(storage: FileStorage, sample_analysis_result_1: AnalysisResult):
    """Test updating an existing analysis result."""
    storage.save_analysis_result(sample_analysis_result_1.analysis_type, [sample_analysis_result_1])

    updated_result = AnalysisResult(
        block_id_1=sample_analysis_result_1.block_id_1,
        block_id_2=sample_analysis_result_1.block_id_2,
        analysis_type=sample_analysis_result_1.analysis_type,
        score=0.99 # Updated score
    )
    # Ensure it has the same result_id
    assert updated_result.result_id == sample_analysis_result_1.result_id

    storage.save_analysis_result(updated_result.analysis_type, [updated_result])

    retrieved_results = storage.get_analysis_results(sample_analysis_result_1.analysis_type)
    assert len(retrieved_results) == 1
    assert retrieved_results[0].score == 0.99
    assert retrieved_results[0] == updated_result # Compare full object


def test_get_analysis_results_empty(storage: FileStorage):
    """Test getting results when none exist for that type."""
    results = storage.get_analysis_results(AnalysisType.SEMANTIC_SIMILARITY)
    assert isinstance(results, list)
    assert len(results) == 0

def test_get_analysis_results_filter_by_block(storage: FileStorage, sample_analysis_result_1: AnalysisResult, sample_analysis_result_2: AnalysisResult):
    """Test filtering analysis results by block_id."""
    storage.save_analysis_result(sample_analysis_result_1.analysis_type, [sample_analysis_result_1])
    storage.save_analysis_result(sample_analysis_result_2.analysis_type, [sample_analysis_result_2])

    block_id_to_find = sample_analysis_result_1.block_id_1 # Belongs to both results

    # Get SEMANTIC results involving block_id_to_find
    results_semantic = storage.get_analysis_results(
        AnalysisType.SEMANTIC_SIMILARITY,
        filter_criteria={"block_id": block_id_to_find}
    )
    assert len(results_semantic) == 1
    assert results_semantic[0] == sample_analysis_result_1

    # Get MD5 results involving block_id_to_find
    results_md5 = storage.get_analysis_results(
        AnalysisType.MD5_DUPLICATE,
        filter_criteria={"block_id": block_id_to_find}
    )
    assert len(results_md5) == 1
    assert results_md5[0] == sample_analysis_result_2

    # Get results involving sample_block_2 (only in result_1)
    results_b2 = storage.get_analysis_results(
        AnalysisType.SEMANTIC_SIMILARITY,
        filter_criteria={"block_id": sample_analysis_result_1.block_id_2}
    )
    assert len(results_b2) == 1
    assert results_b2[0] == sample_analysis_result_1

    # Get results involving a block not present in SEMANTIC results
    results_b3_semantic = storage.get_analysis_results(
        AnalysisType.SEMANTIC_SIMILARITY,
        filter_criteria={"block_id": sample_analysis_result_2.block_id_2} # block_3
    )
    assert len(results_b3_semantic) == 0


def test_get_analysis_results_filter_by_score(storage: FileStorage, sample_analysis_result_1: AnalysisResult):
    """Test filtering analysis results by minimum score."""
    storage.save_analysis_result(sample_analysis_result_1.analysis_type, [sample_analysis_result_1]) # score=0.85

    # Filter for score >= 0.9 (should be empty)
    results_high = storage.get_analysis_results(
        sample_analysis_result_1.analysis_type,
        filter_criteria={"min_score": 0.9}
    )
    assert len(results_high) == 0

    # Filter for score >= 0.8 (should find it)
    results_low = storage.get_analysis_results(
        sample_analysis_result_1.analysis_type,
        filter_criteria={"min_score": 0.8}
    )
    assert len(results_low) == 1
    assert results_low[0] == sample_analysis_result_1

    # Filter for score >= 0.85 (should find it)
    results_exact = storage.get_analysis_results(
        sample_analysis_result_1.analysis_type,
        filter_criteria={"min_score": 0.85}
    )
    assert len(results_exact) == 1
    assert results_exact[0] == sample_analysis_result_1


def test_save_user_decision_new(storage: FileStorage, sample_decision_1: UserDecision, base_test_path: Path):
    """Test saving a new user decision."""
    storage.save_user_decision(sample_decision_1)

    # Verify persistence in decisions.json
    decisions_content = read_json_file(base_test_path / storage._decisions_filename)
    assert sample_decision_1.decision_id in decisions_content
    saved_decision_dict = decisions_content[sample_decision_1.decision_id]

    # Compare serialized data
    expected_dict = sample_decision_1.to_dict()
    # Compare timestamps separately due to potential minor precision differences on save/load
    saved_ts_str = saved_decision_dict.pop("timestamp", None)
    expected_ts_str = expected_dict.pop("timestamp", None)
    assert saved_decision_dict == expected_dict
    assert saved_ts_str is not None and expected_ts_str is not None
    # Parse back to datetime for comparison (or compare ISO strings)
    assert datetime.datetime.fromisoformat(saved_ts_str) == datetime.datetime.fromisoformat(expected_ts_str)


    # Verify retrieval using a filter that should match
    retrieved_decisions = storage.get_user_decisions(filter_criteria={
        "decision_id": sample_decision_1.decision_id
    })
    assert len(retrieved_decisions) == 1
    retrieved = retrieved_decisions[0]
    assert isinstance(retrieved, UserDecision)
    # Compare objects (accounts for datetime precision differences if minor)
    assert retrieved == sample_decision_1

def test_save_user_decision_update(storage: FileStorage, sample_decision_1: UserDecision):
    """Test updating an existing user decision."""
    storage.save_user_decision(sample_decision_1) # Initial decision: MARK_SIMILAR

    # Wait a bit to ensure timestamp changes
    import time
    time.sleep(0.01)

    updated_decision = UserDecision(
        block_id_1=sample_decision_1.block_id_1,
        block_id_2=sample_decision_1.block_id_2,
        analysis_type=sample_decision_1.analysis_type,
        decision=DecisionType.IGNORE, # Changed decision
        notes="Changed my mind.",
        # Timestamp will be different (auto-generated or manually set later)
    )
    # Manually set timestamp if needed for exact comparison, or rely on auto-update
    updated_decision.timestamp = datetime.datetime.now(datetime.timezone.utc)

    assert updated_decision.decision_id == sample_decision_1.decision_id

    storage.save_user_decision(updated_decision)

    retrieved_decisions = storage.get_user_decisions(filter_criteria={
        "decision_id": sample_decision_1.decision_id
    })
    assert len(retrieved_decisions) == 1
    retrieved = retrieved_decisions[0]
    assert retrieved.decision == DecisionType.IGNORE
    assert retrieved.notes == "Changed my mind."
    # Check timestamp is updated (should be later than original)
    assert retrieved.timestamp > sample_decision_1.timestamp
    assert retrieved == updated_decision


def test_get_user_decisions_empty(storage: FileStorage):
    """Test getting decisions when none exist."""
    decisions = storage.get_user_decisions()
    assert isinstance(decisions, list)
    assert len(decisions) == 0

def test_get_user_decisions_filter_by_type(storage: FileStorage, sample_decision_1: UserDecision):
    """Test filtering decisions by analysis type."""
    # Save another decision of a different type
    decision_md5 = UserDecision(
        block_id_1="block_a", block_id_2="block_b",
        analysis_type=AnalysisType.MD5_DUPLICATE, decision=DecisionType.MARK_DUPLICATE
    )
    storage.save_user_decision(sample_decision_1) # SEMANTIC_SIMILARITY
    storage.save_user_decision(decision_md5)    # MD5_DUPLICATE

    decisions_semantic = storage.get_user_decisions(filter_criteria={
        "analysis_type": AnalysisType.SEMANTIC_SIMILARITY
    })
    assert len(decisions_semantic) == 1
    assert decisions_semantic[0] == sample_decision_1

    decisions_md5_enum = storage.get_user_decisions(filter_criteria={
        "analysis_type": AnalysisType.MD5_DUPLICATE
    })
    assert len(decisions_md5_enum) == 1
    assert decisions_md5_enum[0] == decision_md5

    # Test filtering by string value
    decisions_md5_str = storage.get_user_decisions(filter_criteria={
        "analysis_type": "md5_duplicate"
    })
    assert len(decisions_md5_str) == 1
    assert decisions_md5_str[0] == decision_md5


def test_get_user_decisions_filter_by_decision(storage: FileStorage, sample_decision_1: UserDecision):
    """Test filtering decisions by the decision value."""
    # Save another decision with a different outcome
    decision_ignore = UserDecision(
        block_id_1="block_x", block_id_2="block_y",
        analysis_type=AnalysisType.SEMANTIC_SIMILARITY, decision=DecisionType.IGNORE
    )
    storage.save_user_decision(sample_decision_1) # MARK_SIMILAR
    storage.save_user_decision(decision_ignore) # IGNORE

    decisions_similar = storage.get_user_decisions(filter_criteria={
        "decision": DecisionType.MARK_SIMILAR # Use Enum
    })
    assert len(decisions_similar) == 1
    assert decisions_similar[0] == sample_decision_1

    decisions_ignore_enum = storage.get_user_decisions(filter_criteria={
        "decision": DecisionType.IGNORE
    })
    assert len(decisions_ignore_enum) == 1
    assert decisions_ignore_enum[0] == decision_ignore

    # Test filtering by string value
    decisions_ignore_str = storage.get_user_decisions(filter_criteria={
        "decision": "ignore"
    })
    assert len(decisions_ignore_str) == 1
    assert decisions_ignore_str[0] == decision_ignore


def test_get_undecided_pairs_no_results(storage: FileStorage):
    """Test getting undecided pairs when no analysis results exist."""
    undecided = storage.get_undecided_pairs(AnalysisType.SEMANTIC_SIMILARITY)
    assert isinstance(undecided, list)
    assert len(undecided) == 0

def test_get_undecided_pairs_no_decisions(storage: FileStorage, sample_analysis_result_1: AnalysisResult, sample_analysis_result_2: AnalysisResult):
    """Test getting undecided pairs when results exist but no decisions have been made."""
    storage.save_analysis_result(sample_analysis_result_1.analysis_type, [sample_analysis_result_1])
    storage.save_analysis_result(sample_analysis_result_2.analysis_type, [sample_analysis_result_2])

    undecided_semantic = storage.get_undecided_pairs(AnalysisType.SEMANTIC_SIMILARITY)
    assert len(undecided_semantic) == 1
    assert undecided_semantic[0] == sample_analysis_result_1

    undecided_md5 = storage.get_undecided_pairs(AnalysisType.MD5_DUPLICATE)
    assert len(undecided_md5) == 1
    assert undecided_md5[0] == sample_analysis_result_2

def test_get_undecided_pairs_with_some_decisions(storage: FileStorage, sample_analysis_result_1: AnalysisResult, sample_analysis_result_2: AnalysisResult, sample_decision_1: UserDecision):
    """Test getting undecided pairs when some pairs have definitive decisions."""
    storage.save_analysis_result(sample_analysis_result_1.analysis_type, [sample_analysis_result_1])
    storage.save_analysis_result(sample_analysis_result_2.analysis_type, [sample_analysis_result_2])
    # sample_decision_1 is MARK_SIMILAR (definitive)
    storage.save_user_decision(sample_decision_1) # Decision made for result 1 (SEMANTIC)

    # Should NOT return the SEMANTIC pair now
    undecided_semantic = storage.get_undecided_pairs(AnalysisType.SEMANTIC_SIMILARITY)
    assert len(undecided_semantic) == 0

    # Should still return the MD5 pair
    undecided_md5 = storage.get_undecided_pairs(AnalysisType.MD5_DUPLICATE)
    assert len(undecided_md5) == 1
    assert undecided_md5[0] == sample_analysis_result_2

def test_get_undecided_pairs_with_undecided_decision(storage: FileStorage, sample_analysis_result_2: AnalysisResult, sample_decision_2_undecided: UserDecision):
    """Test if pairs explicitly marked as UNDECIDED are still returned as undecided."""
    storage.save_analysis_result(sample_analysis_result_2.analysis_type, [sample_analysis_result_2])
    # Save an explicit UNDECIDED decision
    storage.save_user_decision(sample_decision_2_undecided)

    undecided_md5 = storage.get_undecided_pairs(AnalysisType.MD5_DUPLICATE)
    # The pair should still be returned as it requires a *definitive* decision (not UNDECIDED)
    assert len(undecided_md5) == 1
    assert undecided_md5[0] == sample_analysis_result_2


def test_finalize_method_exists(storage: FileStorage):
    """Test that the finalize method exists (though it might do nothing)."""
    try:
        storage.finalize()
    except Exception as e:
        pytest.fail(f"storage.finalize() raised an exception: {e}")


def test_error_handling_file_not_found_on_load(base_test_path: Path):
    """Test behavior when JSON files are missing during load (e.g., on init)."""
    # Simulate missing files *before* initialization
    storage_err = FileStorage(base_path=base_test_path)
    # Ensure the path is empty before calling initialize on a new instance.
    base_test_path.mkdir(parents=True, exist_ok=True) # Create dir but not files
    # Make one file missing, e.g., blocks.json
    (base_test_path / storage_err._metadata_filename).touch()
    (base_test_path / storage_err._results_filename).touch()
    (base_test_path / storage_err._decisions_filename).touch()


    try:
        storage_err.initialize() # Should recreate or handle missing file gracefully
        assert (base_test_path / storage_err._blocks_filename).exists() # Check if recreated

        # Try loading data - should be empty
        block = storage_err.get_block("some_id")
        assert block is None
    except Exception as e:
        pytest.fail(f"Error handling missing file during initialize failed: {e}")


def test_error_handling_json_decode_error(base_test_path: Path):
    """Test behavior when a JSON file is corrupted."""
    storage_err = FileStorage(base_path=base_test_path)
    storage_err.initialize() # Creates valid files first

    # Write invalid JSON to a file
    blocks_path = base_test_path / storage_err._blocks_filename
    with open(blocks_path, "w", encoding='utf-8') as f:
        f.write("{invalid json")

    # Re-initialize - should log error and reset the file to default
    try:
        # Create a new instance to force reloading from the corrupted file
        storage_reinit = FileStorage(base_path=base_test_path)
        storage_reinit.initialize() # Should handle corrupt file
    except Exception as e:
        # Depending on implementation, init might raise or just log
        pytest.fail(f"Initialize failed unexpectedly on corrupt JSON: {e}")

    # Verify the file was reset by reading it again
    blocks_content = read_json_file(blocks_path)
    assert blocks_content == {} # Assert it was reset to empty dict

    # Verify operations still work (return empty/default) on the reinitialized instance
    block = storage_reinit.get_block("some_id")
    assert block is None


# Add more tests for edge cases:
# - Empty lists passed to save methods
# - Filters that match nothing
# - Special characters in text content or file paths (ensure UTF-8 handling)
# - Very large numbers of blocks/results (basic check, not stress test)
# - Concurrency (not expected to handle, but good to note)

