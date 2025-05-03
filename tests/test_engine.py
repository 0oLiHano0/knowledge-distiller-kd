# tests/core/test_engine.py
"""
Unit tests for the KnowledgeDistillerEngine class.
"""

import pytest
import json
from pathlib import Path
from typing import Type, Generator, List, Tuple, Any # Import List, Tuple, Any
from unittest.mock import MagicMock, patch, call, mock_open, DEFAULT # Import DEFAULT


# Modules to test
from knowledge_distiller_kd.core.engine import KnowledgeDistillerEngine
from knowledge_distiller_kd.storage.file_storage import FileStorage # Needed for mocking
# Import analyzers for mocking their methods
from knowledge_distiller_kd.analysis.md5_analyzer import MD5Analyzer
from knowledge_distiller_kd.analysis.semantic_analyzer import SemanticAnalyzer
from knowledge_distiller_kd.core import constants
from knowledge_distiller_kd.core.error_handler import ConfigurationError, FileOperationError, AnalysisError
from knowledge_distiller_kd.core.utils import create_decision_key
# Import ContentBlock and element types for creating test data
from knowledge_distiller_kd.processing.document_processor import ContentBlock
from unstructured.documents.elements import Element, NarrativeText, Title, CodeSnippet

# --- Fixtures ---

@pytest.fixture
def mock_storage() -> MagicMock:
    """Creates a mock FileStorage object."""
    storage = MagicMock(spec=FileStorage)
    storage.save_decisions.return_value = True
    # Mock load_decisions to return an empty list by default
    storage.load_decisions.return_value = []
    return storage

# Fixture for mock MD5 Analyzer instance
@pytest.fixture
def mock_md5_analyzer() -> MagicMock:
    analyzer = MagicMock(spec=MD5Analyzer)
    analyzer.find_md5_duplicates.return_value = ([], {})
    return analyzer

# Fixture for mock Semantic Analyzer instance
@pytest.fixture
def mock_semantic_analyzer() -> MagicMock:
    analyzer = MagicMock(spec=SemanticAnalyzer)
    analyzer.load_semantic_model.return_value = True
    analyzer.find_semantic_duplicates.return_value = []
    analyzer._model_loaded = True
    analyzer.model = MagicMock()
    # Add the 'similarity_threshold' attribute expected by the engine
    analyzer.similarity_threshold = constants.DEFAULT_SIMILARITY_THRESHOLD
    return analyzer


@pytest.fixture
def tmp_decision_file(tmp_path: Path) -> Path:
    """Provides a path for a temporary decision file."""
    return tmp_path / "test_decisions.json"

@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Provides a path for a temporary output directory."""
    return tmp_path / "test_output"

# Fixture to create Engine instance, now patching analyzers during init
@pytest.fixture
def engine_instance(mock_storage: MagicMock, tmp_decision_file: Path, tmp_output_dir: Path) -> Generator[KnowledgeDistillerEngine, None, None]:
    """Creates a KnowledgeDistillerEngine instance with mocked storage and analyzers."""
    # Use patch as a context manager to ensure analyzers are mocked during Engine init
    with patch('knowledge_distiller_kd.core.engine.MD5Analyzer', return_value=MagicMock(spec=MD5Analyzer)) as MockMD5, \
         patch('knowledge_distiller_kd.core.engine.SemanticAnalyzer', return_value=MagicMock(spec=SemanticAnalyzer)) as MockSemantic:

        # Configure the return values *after* patching
        mock_md5_instance = MockMD5.return_value
        mock_md5_instance.find_md5_duplicates.return_value = ([], {}) # Default return

        mock_semantic_instance = MockSemantic.return_value
        mock_semantic_instance.load_semantic_model.return_value = True
        mock_semantic_instance.find_semantic_duplicates.return_value = []
        mock_semantic_instance._model_loaded = True
        mock_semantic_instance.model = MagicMock()
        mock_semantic_instance.similarity_threshold = constants.DEFAULT_SIMILARITY_THRESHOLD

        # Instantiate the engine *within* the patch context
        engine = KnowledgeDistillerEngine(
            storage=mock_storage,
            decision_file=tmp_decision_file,
            output_dir=tmp_output_dir
        )
        # Assign the mocked instances back to the engine instance if needed by tests
        engine.md5_analyzer = mock_md5_instance
        engine.semantic_analyzer = mock_semantic_instance
        yield engine # Use yield for fixture teardown if needed


# Helper to create ContentBlock for tests
@pytest.fixture
def create_content_block():
    """Factory fixture to create ContentBlock instances for testing."""
    def _create(text: str, file_path: str, block_id: str, block_type: str = "NarrativeText") -> ContentBlock:
        element_map = {"NarrativeText": NarrativeText, "Title": Title, "CodeSnippet": CodeSnippet}
        # Use Type[Element] for the type hint
        element_cls: Type[Element] = element_map.get(block_type, NarrativeText)
        # Create the element instance
        element = element_cls(text=text, element_id=block_id)
        # Create the ContentBlock instance
        cb = ContentBlock(element, file_path)
        # Manually ensure the element type is correct if ContentBlock's inference might change it
        cb.element.__class__ = element_cls
        return cb
    return _create

# --- Test Cases ---

# --- Initialization Tests ---
def test_engine_initialization(engine_instance: KnowledgeDistillerEngine, mock_storage: MagicMock, tmp_decision_file: Path, tmp_output_dir: Path):
    """Test basic engine initialization."""
    assert engine_instance.storage == mock_storage
    assert engine_instance.input_dir is None
    assert engine_instance.decision_file == tmp_decision_file.resolve()
    assert engine_instance.output_dir == tmp_output_dir.resolve()
    assert not engine_instance.skip_semantic
    assert engine_instance.similarity_threshold == constants.DEFAULT_SIMILARITY_THRESHOLD
    assert engine_instance.blocks_data == []
    assert engine_instance.block_decisions == {}
    assert engine_instance.md5_duplicates == []
    assert engine_instance.semantic_duplicates == []
    assert not engine_instance._decisions_loaded
    assert not engine_instance._analysis_completed
    # Check that the analyzers are the mocked instances
    assert isinstance(engine_instance.md5_analyzer, MagicMock)
    assert isinstance(engine_instance.semantic_analyzer, MagicMock)

def test_engine_initialization_with_input_dir(mock_storage: MagicMock, tmp_path: Path):
    """Test initialization with a valid input directory."""
    input_dir = tmp_path / "input_init"
    input_dir.mkdir()
    # Need to patch analyzers here too, as they are created in __init__
    with patch('knowledge_distiller_kd.core.engine.MD5Analyzer'), \
         patch('knowledge_distiller_kd.core.engine.SemanticAnalyzer'):
        engine = KnowledgeDistillerEngine(storage=mock_storage, input_dir=input_dir)
        assert engine.input_dir == input_dir.resolve()

def test_engine_initialization_with_invalid_input_dir(mock_storage: MagicMock, tmp_path: Path):
    """Test initialization fails with a non-existent input directory."""
    invalid_input = tmp_path / "non_existent_dir"
    with pytest.raises(ConfigurationError, match="Engine initialization failed"):
         # Patch analyzers even for failing init tests if __init__ tries to create them
         with patch('knowledge_distiller_kd.core.engine.MD5Analyzer'), \
              patch('knowledge_distiller_kd.core.engine.SemanticAnalyzer'):
            KnowledgeDistillerEngine(storage=mock_storage, input_dir=invalid_input)

def test_engine_initialization_with_file_as_input_dir(mock_storage: MagicMock, tmp_path: Path):
    """Test initialization fails when input path is a file, not a directory."""
    input_file = tmp_path / "input_file.txt"
    input_file.touch()
    with pytest.raises(ConfigurationError, match="Input path is not a directory"):
         with patch('knowledge_distiller_kd.core.engine.MD5Analyzer'), \
              patch('knowledge_distiller_kd.core.engine.SemanticAnalyzer'):
            KnowledgeDistillerEngine(storage=mock_storage, input_dir=input_file)

# --- set_input_dir Tests ---
def test_set_input_dir_success(engine_instance: KnowledgeDistillerEngine, tmp_path: Path):
    """Test successfully setting the input directory after initialization."""
    valid_input_dir = tmp_path / "test_input"
    valid_input_dir.mkdir()
    # Give the engine some state to check if it gets reset
    engine_instance.block_decisions = {"dummy": "keep"}
    engine_instance._analysis_completed = True
    # Use wraps= to call the original _reset_state while still tracking calls
    with patch.object(engine_instance, '_reset_state', wraps=engine_instance._reset_state) as mock_reset:
        result = engine_instance.set_input_dir(valid_input_dir)
        assert result is True
        assert engine_instance.input_dir == valid_input_dir.resolve()
        # Check that state was reset
        mock_reset.assert_called_once()
        assert engine_instance.block_decisions == {}
        assert not engine_instance._analysis_completed

def test_set_input_dir_not_a_directory(engine_instance: KnowledgeDistillerEngine, tmp_path: Path):
    """Test setting input directory to a file path fails."""
    invalid_input_file = tmp_path / "test_file.txt"
    invalid_input_file.touch()
    with patch.object(engine_instance, '_reset_state') as mock_reset:
        result = engine_instance.set_input_dir(invalid_input_file)
        assert result is False
        assert engine_instance.input_dir is None # Should not be set
        mock_reset.assert_not_called() # State should not be reset

def test_set_input_dir_does_not_exist(engine_instance: KnowledgeDistillerEngine, tmp_path: Path):
    """Test setting input directory to a non-existent path fails."""
    non_existent_dir = tmp_path / "non_existent"
    with patch.object(engine_instance, '_reset_state') as mock_reset:
        result = engine_instance.set_input_dir(non_existent_dir)
        assert result is False
        assert engine_instance.input_dir is None # Should not be set
        mock_reset.assert_not_called()

# --- load_decisions Tests ---
def test_load_decisions_success_absolute_paths(engine_instance: KnowledgeDistillerEngine, mock_storage: MagicMock, tmp_path: Path):
    """Test loading decisions with absolute paths in the file."""
    abs_file_path = (tmp_path / "docs" / "file1.md").resolve()
    mock_decision_data = [
        {"file": str(abs_file_path), "block_id": "id1", "type": "NarrativeText", "decision": "keep"},
        {"file": str(abs_file_path), "block_id": "id2", "type": "CodeSnippet", "decision": "delete"}
    ]
    mock_storage.load_decisions.return_value = mock_decision_data

    result = engine_instance.load_decisions()

    assert result is True
    assert engine_instance._decisions_loaded is True
    expected_key1 = create_decision_key(str(abs_file_path), "id1", "NarrativeText")
    expected_key2 = create_decision_key(str(abs_file_path), "id2", "CodeSnippet")
    assert expected_key1 in engine_instance.block_decisions
    assert engine_instance.block_decisions[expected_key1] == "keep"
    assert expected_key2 in engine_instance.block_decisions
    assert engine_instance.block_decisions[expected_key2] == "delete"
    mock_storage.load_decisions.assert_called_once_with(engine_instance.decision_file)

def test_load_decisions_success_relative_paths(engine_instance: KnowledgeDistillerEngine, mock_storage: MagicMock, tmp_path: Path):
    """Test loading decisions with relative paths, resolved against input_dir."""
    input_dir = tmp_path / "my_input"
    input_dir.mkdir()
    engine_instance.input_dir = input_dir # Set input_dir for resolving

    relative_file_path1 = "subdir/file1.md"
    relative_file_path2 = "file2.md"
    abs_file_path1 = (input_dir / relative_file_path1).resolve()
    abs_file_path2 = (input_dir / relative_file_path2).resolve()

    mock_decision_data = [
        {"file": relative_file_path1, "block_id": "rel_id1", "type": "Title", "decision": "keep"},
        {"file": relative_file_path2, "block_id": "rel_id2", "type": "ListItem", "decision": "undecided"}
    ]
    mock_storage.load_decisions.return_value = mock_decision_data

    result = engine_instance.load_decisions()

    assert result is True
    assert engine_instance._decisions_loaded is True
    # Keys in memory should use the resolved absolute path
    expected_key1 = create_decision_key(str(abs_file_path1), "rel_id1", "Title")
    expected_key2 = create_decision_key(str(abs_file_path2), "rel_id2", "ListItem")
    assert expected_key1 in engine_instance.block_decisions
    assert engine_instance.block_decisions[expected_key1] == "keep"
    assert expected_key2 in engine_instance.block_decisions
    assert engine_instance.block_decisions[expected_key2] == "undecided"
    mock_storage.load_decisions.assert_called_once_with(engine_instance.decision_file)

def test_load_decisions_relative_paths_no_input_dir(engine_instance: KnowledgeDistillerEngine, mock_storage: MagicMock):
    """Test loading relative paths when input_dir is not set (should use path as is)."""
    engine_instance.input_dir = None # Ensure input_dir is not set
    relative_file_path = "some/relative/path.md"
    mock_decision_data = [
        {"file": relative_file_path, "block_id": "rel_id_no_input", "type": "NarrativeText", "decision": "keep"}
    ]
    mock_storage.load_decisions.return_value = mock_decision_data

    result = engine_instance.load_decisions()

    assert result is True
    assert engine_instance._decisions_loaded is True
    # The key should be created using the relative path directly
    expected_key = create_decision_key(relative_file_path, "rel_id_no_input", "NarrativeText")
    assert expected_key in engine_instance.block_decisions
    assert engine_instance.block_decisions[expected_key] == "keep"
    mock_storage.load_decisions.assert_called_once_with(engine_instance.decision_file)

def test_load_decisions_storage_returns_empty(engine_instance: KnowledgeDistillerEngine, mock_storage: MagicMock):
    """Test loading when the storage layer returns an empty list."""
    mock_storage.load_decisions.return_value = []
    result = engine_instance.load_decisions()
    assert result is False # No decisions loaded
    assert engine_instance._decisions_loaded is False
    assert engine_instance.block_decisions == {}
    mock_storage.load_decisions.assert_called_once_with(engine_instance.decision_file)

def test_load_decisions_invalid_records(engine_instance: KnowledgeDistillerEngine, mock_storage: MagicMock, tmp_path: Path):
    """Test loading a file with a mix of valid and invalid decision records."""
    abs_file_path = (tmp_path / "file_good.md").resolve()
    mock_decision_data = [
        {"file": str(abs_file_path), "block_id": "good1", "type": "NarrativeText", "decision": "keep"},
        {"file": str(abs_file_path), "block_id": "incomplete"}, # Missing type/decision
        {"file": str(abs_file_path), "block_id": "bad_decision", "type": "CodeSnippet", "decision": "maybe"}, # Invalid decision value
        "not a dictionary", # Malformed record
        {"file": str(abs_file_path), "block_id": "good2", "type": "Title", "decision": "delete"}
    ]
    mock_storage.load_decisions.return_value = mock_decision_data

    result = engine_instance.load_decisions()

    assert result is True # Loading itself succeeded, even if some records failed
    assert engine_instance._decisions_loaded is True
    assert len(engine_instance.block_decisions) == 2 # Only the two valid records should be loaded
    expected_key1 = create_decision_key(str(abs_file_path), "good1", "NarrativeText")
    expected_key2 = create_decision_key(str(abs_file_path), "good2", "Title")
    assert expected_key1 in engine_instance.block_decisions
    assert engine_instance.block_decisions[expected_key1] == "keep"
    assert expected_key2 in engine_instance.block_decisions
    assert engine_instance.block_decisions[expected_key2] == "delete"
    mock_storage.load_decisions.assert_called_once_with(engine_instance.decision_file)

def test_load_decisions_storage_exception(engine_instance: KnowledgeDistillerEngine, mock_storage: MagicMock):
    """Test loading when the storage layer raises an exception."""
    mock_storage.load_decisions.side_effect = FileOperationError("Permission denied", error_code="PERMISSION_DENIED")
    result = engine_instance.load_decisions()
    assert result is False
    assert engine_instance._decisions_loaded is False
    assert engine_instance.block_decisions == {}
    mock_storage.load_decisions.assert_called_once_with(engine_instance.decision_file)

# --- save_decisions Tests ---
def test_save_decisions_no_decisions(engine_instance: KnowledgeDistillerEngine, mock_storage: MagicMock):
    """Test saving when there are no decisions in memory."""
    engine_instance.block_decisions = {}
    result = engine_instance.save_decisions()
    assert result is False
    mock_storage.save_decisions.assert_not_called()

def test_save_decisions_success_absolute_paths_no_input_dir(engine_instance: KnowledgeDistillerEngine, mock_storage: MagicMock, tmp_path: Path):
    """Test saving absolute paths when input_dir is not set."""
    engine_instance.input_dir = None
    abs_path1 = (tmp_path / "fileA.md").resolve()
    abs_path2 = (tmp_path / "subdir" / "fileB.md").resolve()
    key1 = create_decision_key(str(abs_path1), "idA", "Title")
    key2 = create_decision_key(str(abs_path2), "idB", "NarrativeText")
    engine_instance.block_decisions = {key1: "keep", key2: "delete"}

    # Expected data structure passed to storage.save_decisions
    expected_data_to_save = [
        {'file': str(abs_path1), 'block_id': 'idA', 'type': 'Title', 'decision': 'keep'},
        {'file': str(abs_path2), 'block_id': 'idB', 'type': 'NarrativeText', 'decision': 'delete'}
    ]
    # Sort for consistent comparison
    expected_data_to_save.sort(key=lambda x: (x['file'], x['block_id']))

    result = engine_instance.save_decisions()

    assert result is True
    mock_storage.save_decisions.assert_called_once()
    # Retrieve arguments passed to the mock
    args, kwargs = mock_storage.save_decisions.call_args
    assert args[0] == engine_instance.decision_file # Check target file path
    saved_data = args[1] # The list of decision dicts
    saved_data.sort(key=lambda x: (x['file'], x['block_id'])) # Sort for comparison
    assert saved_data == expected_data_to_save

def test_save_decisions_success_relative_paths_with_input_dir(engine_instance: KnowledgeDistillerEngine, mock_storage: MagicMock, tmp_path: Path):
    """Test saving relative paths when input_dir is set."""
    input_dir = tmp_path / "input_save"
    input_dir.mkdir()
    engine_instance.input_dir = input_dir

    abs_path1 = (input_dir / "file1.md").resolve()
    abs_path2 = (input_dir / "sub" / "file2.md").resolve()
    abs_path_outside = (tmp_path / "outside.md").resolve() # A file outside input_dir

    key1 = create_decision_key(str(abs_path1), "id1", "CodeSnippet")
    key2 = create_decision_key(str(abs_path2), "id2", "NarrativeText")
    key_outside = create_decision_key(str(abs_path_outside), "id_out", "Title")
    engine_instance.block_decisions = {key1: "keep", key2: "delete", key_outside: "keep"}

    # Expected data structure - paths inside input_dir should be relative
    expected_data_to_save = [
        {'file': "file1.md", 'block_id': 'id1', 'type': 'CodeSnippet', 'decision': 'keep'},
        {'file': str(Path("sub") / "file2.md"), 'block_id': 'id2', 'type': 'NarrativeText', 'decision': 'delete'},
        {'file': str(abs_path_outside), 'block_id': 'id_out', 'type': 'Title', 'decision': 'keep'} # Outside path remains absolute
    ]
    expected_data_to_save.sort(key=lambda x: (x['file'], x['block_id']))

    result = engine_instance.save_decisions()

    assert result is True
    mock_storage.save_decisions.assert_called_once()
    args, kwargs = mock_storage.save_decisions.call_args
    assert args[0] == engine_instance.decision_file
    saved_data = args[1]
    saved_data.sort(key=lambda x: (x['file'], x['block_id']))
    assert saved_data == expected_data_to_save

def test_save_decisions_invalid_key_skipped(engine_instance: KnowledgeDistillerEngine, mock_storage: MagicMock):
    """Test that invalid keys in block_decisions are skipped during save."""
    engine_instance.block_decisions = {"valid::key::Type": "keep", "invalidkey": "delete"}
    expected_data_to_save = [{'file': 'valid', 'block_id': 'key', 'type': 'Type', 'decision': 'keep'}]

    result = engine_instance.save_decisions()

    assert result is True # Should still succeed if at least one valid key exists
    mock_storage.save_decisions.assert_called_once_with(engine_instance.decision_file, expected_data_to_save)

def test_save_decisions_storage_fails(engine_instance: KnowledgeDistillerEngine, mock_storage: MagicMock):
    """Test saving when the storage layer indicates failure."""
    engine_instance.block_decisions = {"a::b::C": "keep"}
    mock_storage.save_decisions.return_value = False # Simulate storage failure
    result = engine_instance.save_decisions()
    assert result is False
    mock_storage.save_decisions.assert_called_once() # Storage was called

def test_save_decisions_storage_exception(engine_instance: KnowledgeDistillerEngine, mock_storage: MagicMock):
    """Test saving when the storage layer raises an exception."""
    engine_instance.block_decisions = {"a::b::C": "keep"}
    mock_storage.save_decisions.side_effect = FileOperationError("Disk full", error_code="DISK_FULL")
    result = engine_instance.save_decisions()
    assert result is False
    mock_storage.save_decisions.assert_called_once()

# --- apply_decisions Tests ---
def test_apply_decisions_analysis_not_completed(engine_instance: KnowledgeDistillerEngine):
    """Test applying decisions when analysis hasn't run."""
    engine_instance._analysis_completed = False
    result = engine_instance.apply_decisions()
    assert result is False

def test_apply_decisions_no_blocks(engine_instance: KnowledgeDistillerEngine):
    """Test applying decisions when there are no blocks."""
    engine_instance.blocks_data = []
    engine_instance._analysis_completed = True # Mark analysis as done
    result = engine_instance.apply_decisions()
    assert result is True # No blocks to process is considered success

@patch('builtins.open', new_callable=mock_open)
@patch('pathlib.Path.mkdir')
def test_apply_decisions_success(mock_mkdir: MagicMock, mock_file_open: MagicMock, engine_instance: KnowledgeDistillerEngine, tmp_path: Path, create_content_block):
    """Test successfully applying decisions and writing output files."""
    input_dir = tmp_path / "apply_input"
    output_dir = tmp_path / "apply_output"
    engine_instance.input_dir = input_dir
    engine_instance.output_dir = output_dir

    file1_path = (input_dir / "file1.md").resolve()
    file2_path = (input_dir / "subdir" / "file2.md").resolve()

    block1a = create_content_block("Content 1a", str(file1_path), "id1a")
    block1b = create_content_block("Content 1b", str(file1_path), "id1b")
    block2a = create_content_block("Content 2a", str(file2_path), "id2a", "Title")
    block2b = create_content_block("Content 2b", str(file2_path), "id2b", "CodeSnippet")
    block2c = create_content_block("Content 2c", str(file2_path), "id2c")
    engine_instance.blocks_data = [block1a, block1b, block2a, block2b, block2c]

    key1a = create_decision_key(str(file1_path), "id1a", "NarrativeText")
    key1b = create_decision_key(str(file1_path), "id1b", "NarrativeText")
    key2a = create_decision_key(str(file2_path), "id2a", "Title")
    key2b = create_decision_key(str(file2_path), "id2b", "CodeSnippet")
    key2c = create_decision_key(str(file2_path), "id2c", "NarrativeText")
    engine_instance.block_decisions = {
        key1a: constants.DECISION_KEEP,
        key1b: constants.DECISION_DELETE,
        key2a: constants.DECISION_UNDECIDED, # Keep
        key2b: constants.DECISION_KEEP,
        key2c: constants.DECISION_DELETE,
    }
    engine_instance._analysis_completed = True

    mock_file1_handle = mock_open()
    mock_file2_handle = mock_open()
    def side_effect_open(path, *args, **kwargs):
        path_obj = Path(path)
        if path_obj.name == f"file1{constants.DEFAULT_OUTPUT_SUFFIX}.md":
            return mock_file1_handle.return_value
        elif path_obj.name == f"file2{constants.DEFAULT_OUTPUT_SUFFIX}.md":
            return mock_file2_handle.return_value
        else:
            return mock_open().return_value
    mock_file_open.side_effect = side_effect_open

    result = engine_instance.apply_decisions()

    assert result is True

    # Check that output directories were created (at least the base dir)
    assert mock_mkdir.call_count >= 1

    output_file1_path = output_dir / f"file1{constants.DEFAULT_OUTPUT_SUFFIX}.md"
    output_file2_path = output_dir / "subdir" / f"file2{constants.DEFAULT_OUTPUT_SUFFIX}.md"
    mock_file_open.assert_any_call(output_file1_path, 'w', encoding=constants.DEFAULT_ENCODING)
    mock_file_open.assert_any_call(output_file2_path, 'w', encoding=constants.DEFAULT_ENCODING)
    assert mock_file_open.call_count == 2

    expected_content_file1 = "Content 1a"
    expected_content_file2 = "Content 2a\n\nContent 2b"
    handle1 = mock_file1_handle()
    handle2 = mock_file2_handle()
    written1 = "".join(c[0][0] for c in handle1.write.call_args_list)
    written2 = "".join(c[0][0] for c in handle2.write.call_args_list)
    assert written1 == expected_content_file1
    assert written2 == expected_content_file2

@patch('builtins.open', new_callable=mock_open)
@patch('pathlib.Path.mkdir')
def test_apply_decisions_uses_relative_keys(mock_mkdir: MagicMock, mock_file_open: MagicMock, engine_instance: KnowledgeDistillerEngine, tmp_path: Path, create_content_block):
    """Test apply_decisions correctly uses relative keys from decisions if input_dir is set."""
    input_dir = tmp_path / "apply_rel_input"
    output_dir = tmp_path / "apply_rel_output"
    engine_instance.input_dir = input_dir # Set input dir
    engine_instance.output_dir = output_dir

    file1_abs = (input_dir / "file1.md").resolve()
    block1 = create_content_block("Keep me", str(file1_abs), "id1")
    engine_instance.blocks_data = [block1]

    # Decision uses a RELATIVE path key
    relative_key = create_decision_key("file1.md", "id1", "NarrativeText")
    engine_instance.block_decisions = {relative_key: constants.DECISION_KEEP}
    engine_instance._analysis_completed = True

    result = engine_instance.apply_decisions()

    assert result is True
    # Check output file creation and content
    output_file1_path = output_dir / f"file1{constants.DEFAULT_OUTPUT_SUFFIX}.md"
    mock_file_open.assert_called_once_with(output_file1_path, 'w', encoding=constants.DEFAULT_ENCODING)
    handle = mock_file_open()
    handle.write.assert_called_once_with("Keep me") # Content should be written

@patch('pathlib.Path.mkdir')
def test_apply_decisions_output_dir_creation_fails(mock_mkdir: MagicMock, engine_instance: KnowledgeDistillerEngine, tmp_path: Path, create_content_block):
    """Test apply_decisions handling when subdirectory creation fails."""
    input_dir = tmp_path / "apply_fail_input"; input_dir.mkdir()
    output_dir = tmp_path / "apply_fail_output" # Base output dir might exist or not
    engine_instance.input_dir = input_dir
    engine_instance.output_dir = output_dir

    file1_abs = (input_dir / "subdir" / "file1.md").resolve()
    block1 = create_content_block("Some content", str(file1_abs), "id1")
    engine_instance.blocks_data = [block1]
    key1 = create_decision_key(str(file1_abs), "id1", "NarrativeText")
    engine_instance.block_decisions = {key1: constants.DECISION_KEEP}
    engine_instance._analysis_completed = True

    # Corrected side_effect function signature - ENSURE 'self' IS PRESENT
    def mkdir_side_effect_func(self, parents=False, exist_ok=False): # Takes self
        if self == output_dir / "subdir": # Check if it's the specific subdir
            raise OSError("Permission denied for testing")
        # Allow creation of the base output directory if needed
        elif self == output_dir:
             return DEFAULT # Or simply do nothing to simulate success
        return DEFAULT # Allow other mkdir calls

    mock_mkdir.side_effect = mkdir_side_effect_func

    # Patch open as well, although it might not be reached for the failed file
    with patch('builtins.open', mock_open()) as mock_open_func:
        result = engine_instance.apply_decisions()
        # The overall process should report failure if any file fails
        assert result is False
        # Verify that the attempt to create the specific subdirectory was made (indirectly)
        # We rely on the fact that the OSError was raised by the side_effect
        # and that mock_open was not called for the target file.
        # Verify that open was NOT called for the file in the failed directory
        output_file1_path = output_dir / "subdir" / f"file1{constants.DEFAULT_OUTPUT_SUFFIX}.md"
        # Check if mock_open_func was called with the specific path
        called_paths = {c[0][0] for c in mock_open_func.call_args_list}
        assert output_file1_path not in called_paths


# --- run_analysis Tests ---
# Mock dependencies for run_analysis
@patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine._process_documents')
@patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine._merge_code_blocks_step')
@patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine._initialize_decisions')
@patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine._update_decisions_from_md5')
def test_run_analysis_success_flow_no_semantic(
    mock_update_md5: MagicMock, mock_init_decisions: MagicMock, mock_merge_blocks: MagicMock, mock_process_docs: MagicMock,
    engine_instance: KnowledgeDistillerEngine, tmp_path: Path):
    """Test the successful orchestration of run_analysis when skipping semantic."""
    input_dir = tmp_path / "run_input"
    input_dir.mkdir()
    engine_instance.input_dir = input_dir
    engine_instance.skip_semantic = True # Explicitly skip semantic

    # Setup mock return values for successful steps
    mock_process_docs.return_value = True
    mock_merge_blocks.return_value = True
    mock_init_decisions.return_value = True
    mock_md5_duplicates_found = [[MagicMock(spec=ContentBlock), MagicMock(spec=ContentBlock)]] # Simulate finding one group
    mock_suggested_md5_decisions = {"key1::id1::Type": "keep", "key2::id2::Type": "delete"}
    # Configure the mocked md5_analyzer on the engine_instance
    engine_instance.md5_analyzer.find_md5_duplicates.return_value = (mock_md5_duplicates_found, mock_suggested_md5_decisions)

    result = engine_instance.run_analysis()

    assert result is True
    assert engine_instance._analysis_completed is True
    # Verify steps were called
    mock_process_docs.assert_called_once()
    mock_merge_blocks.assert_called_once()
    mock_init_decisions.assert_called_once()
    engine_instance.md5_analyzer.find_md5_duplicates.assert_called_once_with(engine_instance.blocks_data, engine_instance.block_decisions)
    mock_update_md5.assert_called_once_with(mock_suggested_md5_decisions)
    # Verify semantic steps were NOT called
    engine_instance.semantic_analyzer.load_semantic_model.assert_not_called()
    engine_instance.semantic_analyzer.find_semantic_duplicates.assert_not_called()
    # Verify results stored on engine
    assert engine_instance.md5_duplicates == mock_md5_duplicates_found
    assert engine_instance.semantic_duplicates == []

@patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine._process_documents')
@patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine._merge_code_blocks_step')
@patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine._initialize_decisions')
@patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine._update_decisions_from_md5')
@patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine._filter_blocks_for_semantic')
def test_run_analysis_success_flow_with_semantic(
    mock_filter_semantic: MagicMock, mock_update_md5: MagicMock, mock_init_decisions: MagicMock, mock_merge_blocks: MagicMock, mock_process_docs: MagicMock,
    engine_instance: KnowledgeDistillerEngine, tmp_path: Path, create_content_block):
    """Test the successful orchestration of run_analysis including semantic."""
    input_dir = tmp_path / "run_input_sem"
    input_dir.mkdir()
    engine_instance.input_dir = input_dir
    engine_instance.skip_semantic = False # Ensure semantic is NOT skipped

    # Setup mocks for successful steps
    mock_process_docs.return_value = True
    mock_merge_blocks.return_value = True
    mock_init_decisions.return_value = True
    engine_instance.md5_analyzer.find_md5_duplicates.return_value = ([], {}) # No MD5 duplicates
    engine_instance.semantic_analyzer.load_semantic_model.return_value = True
    engine_instance.semantic_analyzer._model_loaded = True # Ensure model is marked loaded
    engine_instance.semantic_analyzer.model = MagicMock() # Ensure model object exists

    # Simulate filtering and finding semantic duplicates
    filtered_blocks = [create_content_block("text1", "f1.md", "b1"), create_content_block("text2", "f2.md", "b2")]
    mock_filter_semantic.return_value = filtered_blocks
    mock_semantic_pair = (filtered_blocks[0], filtered_blocks[1], 0.9)
    engine_instance.semantic_analyzer.find_semantic_duplicates.return_value = [mock_semantic_pair]

    result = engine_instance.run_analysis()

    assert result is True
    assert engine_instance._analysis_completed is True
    # Verify all steps were called in order
    mock_process_docs.assert_called_once()
    mock_merge_blocks.assert_called_once()
    mock_init_decisions.assert_called_once()
    engine_instance.md5_analyzer.find_md5_duplicates.assert_called_once()
    mock_update_md5.assert_called_once_with({}) # Called with empty MD5 suggestions
    engine_instance.semantic_analyzer.load_semantic_model.assert_called_once()
    mock_filter_semantic.assert_called_once()
    engine_instance.semantic_analyzer.find_semantic_duplicates.assert_called_once_with(filtered_blocks)
    # Verify results stored on engine
    assert engine_instance.md5_duplicates == []
    assert engine_instance.semantic_duplicates == [mock_semantic_pair]

def test_run_analysis_no_input_dir(engine_instance: KnowledgeDistillerEngine):
    """Test run_analysis fails if input_dir is not set."""
    engine_instance.input_dir = None
    result = engine_instance.run_analysis()
    assert result is False

@patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine._process_documents', return_value=False)
@patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine._merge_code_blocks_step') # Also mock merge
def test_run_analysis_process_docs_fails(mock_merge: MagicMock, mock_process_docs: MagicMock, engine_instance: KnowledgeDistillerEngine, tmp_path: Path):
    """Test run_analysis stops if document processing fails."""
    input_dir = tmp_path / "fail_input"
    input_dir.mkdir()
    engine_instance.input_dir = input_dir
    result = engine_instance.run_analysis()
    assert result is False
    assert not engine_instance._analysis_completed
    mock_process_docs.assert_called_once()
    # Ensure subsequent steps like merge were not called
    mock_merge.assert_not_called() # Check that merge step was skipped


@patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine._process_documents', return_value=True)
@patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine._merge_code_blocks_step', return_value=True)
@patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine._initialize_decisions', return_value=True)
@patch('knowledge_distiller_kd.core.engine.KnowledgeDistillerEngine._update_decisions_from_md5', return_value=None)
def test_run_analysis_semantic_load_fails(mock_update_md5, mock_init_dec, mock_merge, mock_proc_docs, engine_instance: KnowledgeDistillerEngine, tmp_path: Path):
    """Test run_analysis continues but skips semantic steps if model load fails."""
    input_dir = tmp_path / "fail_input_sem"
    input_dir.mkdir()
    engine_instance.input_dir = input_dir
    engine_instance.skip_semantic = False # Attempt semantic analysis

    # Simulate MD5 finding nothing
    engine_instance.md5_analyzer.find_md5_duplicates.return_value = ([], {})
    # Simulate semantic model load failure
    engine_instance.semantic_analyzer.load_semantic_model.return_value = False
    engine_instance.semantic_analyzer._model_loaded = False
    engine_instance.semantic_analyzer.model = None

    result = engine_instance.run_analysis()

    # Analysis should still complete, but semantic steps skipped
    assert result is True
    assert engine_instance._analysis_completed is True
    # Verify MD5 was called
    engine_instance.md5_analyzer.find_md5_duplicates.assert_called_once()
    # Verify semantic load was attempted
    engine_instance.semantic_analyzer.load_semantic_model.assert_called_once()
    # Verify semantic find was NOT called
    engine_instance.semantic_analyzer.find_semantic_duplicates.assert_not_called()
    # Verify skip_semantic flag was set on the engine by the failed load logic (indirectly)
    assert engine_instance.skip_semantic is True

# --- Add tests for public interface methods (getters, setters) ---
def test_get_md5_duplicates(engine_instance: KnowledgeDistillerEngine):
    """Test getting MD5 duplicates."""
    engine_instance._analysis_completed = True
    engine_instance.md5_duplicates = ["group1"] # Dummy data
    assert engine_instance.get_md5_duplicates() == ["group1"]

def test_get_md5_duplicates_analysis_not_done(engine_instance: KnowledgeDistillerEngine):
    """Test getting MD5 duplicates before analysis."""
    engine_instance._analysis_completed = False
    assert engine_instance.get_md5_duplicates() == []

def test_get_semantic_duplicates(engine_instance: KnowledgeDistillerEngine):
    """Test getting semantic duplicates."""
    engine_instance._analysis_completed = True
    engine_instance.skip_semantic = False
    engine_instance.semantic_duplicates = ["pair1"] # Dummy data
    assert engine_instance.get_semantic_duplicates() == ["pair1"]

def test_get_semantic_duplicates_skipped(engine_instance: KnowledgeDistillerEngine):
    """Test getting semantic duplicates when skipped."""
    engine_instance._analysis_completed = True
    engine_instance.skip_semantic = True
    engine_instance.semantic_duplicates = ["pair1"] # Should still return empty
    assert engine_instance.get_semantic_duplicates() == []

def test_get_semantic_duplicates_analysis_not_done(engine_instance: KnowledgeDistillerEngine):
    """Test getting semantic duplicates before analysis."""
    engine_instance._analysis_completed = False
    assert engine_instance.get_semantic_duplicates() == []

def test_update_decision(engine_instance: KnowledgeDistillerEngine):
    """Test updating a decision."""
    key = "file.md::id1::Type"
    assert engine_instance.update_decision(key, constants.DECISION_KEEP) is True
    assert engine_instance.block_decisions[key] == constants.DECISION_KEEP
    assert engine_instance.update_decision(key, constants.DECISION_DELETE) is True
    assert engine_instance.block_decisions[key] == constants.DECISION_DELETE
    assert engine_instance.update_decision(key, constants.DECISION_UNDECIDED) is True
    assert engine_instance.block_decisions[key] == constants.DECISION_UNDECIDED

def test_update_decision_invalid_value(engine_instance: KnowledgeDistillerEngine):
    """Test updating with an invalid decision value."""
    key = "file.md::id1::Type"
    assert engine_instance.update_decision(key, "maybe") is False
    assert key not in engine_instance.block_decisions

def test_update_decision_invalid_key(engine_instance: KnowledgeDistillerEngine):
    """Test updating with an invalid key."""
    assert engine_instance.update_decision("", constants.DECISION_KEEP) is False

# ==================== 增强 test_get_status_summary ====================
def test_get_status_summary(engine_instance: KnowledgeDistillerEngine, tmp_decision_file: Path, tmp_output_dir: Path, create_content_block):
    """Test the status summary dictionary generation with more detail."""
    # Setup some state
    input_dir = tmp_decision_file.parent / "status_input" # Use tmp_path from fixture
    input_dir.mkdir()
    engine_instance.set_input_dir(input_dir)
    engine_instance.skip_semantic = True
    engine_instance.similarity_threshold = 0.85
    block1 = create_content_block("text1", str(input_dir / "f1.md"), "b1")
    block2 = create_content_block("text2", str(input_dir / "f2.md"), "b2")
    engine_instance.blocks_data = [block1, block2]
    key1 = create_decision_key(str(input_dir / "f1.md"), "b1", "NarrativeText")
    engine_instance.block_decisions = {key1: constants.DECISION_KEEP}
    engine_instance.md5_duplicates = [[block1, block2]] # Simulate one group
    engine_instance.semantic_duplicates = [] # Skipped
    engine_instance._analysis_completed = True
    engine_instance._decisions_loaded = True

    summary = engine_instance.get_status_summary()

    assert isinstance(summary, dict)
    # Check presence and types/values of keys
    assert summary["input_dir"] == str(input_dir.resolve())
    assert summary["decision_file"] == str(tmp_decision_file.resolve())
    assert summary["output_dir"] == str(tmp_output_dir.resolve())
    assert summary["skip_semantic"] is True
    assert summary["similarity_threshold"] == 0.85
    assert summary["analysis_completed"] is True
    assert summary["decisions_loaded"] is True
    assert summary["total_blocks"] == 2
    assert summary["md5_duplicates_groups"] == 1
    assert summary["semantic_duplicates_pairs"] == 0 # Because skipped
    assert summary["decided_blocks"] == 1
    # ==================== 修改断言：计算未决定块数 ====================
    assert summary["total_blocks"] - summary["decided_blocks"] == 1 # Calculate undecided
    # ==============================================================
# =======================================================================

def test_set_similarity_threshold(engine_instance: KnowledgeDistillerEngine):
    """Test setting a valid similarity threshold."""
    engine_instance._analysis_completed = True # Set initial state
    assert engine_instance.set_similarity_threshold(0.75) is True
    assert engine_instance.similarity_threshold == 0.75
    assert engine_instance.semantic_analyzer.similarity_threshold == 0.75
    assert engine_instance._analysis_completed is False # Should reset analysis status

def test_set_similarity_threshold_invalid(engine_instance: KnowledgeDistillerEngine):
    """Test setting an invalid similarity threshold."""
    original_threshold = engine_instance.similarity_threshold
    engine_instance._analysis_completed = True # Set initial state
    assert engine_instance.set_similarity_threshold(1.5) is False
    assert engine_instance.similarity_threshold == original_threshold # Should not change
    assert engine_instance.semantic_analyzer.similarity_threshold == original_threshold
    assert engine_instance._analysis_completed is True # Should NOT reset
    assert engine_instance.set_similarity_threshold(-0.1) is False
    assert engine_instance.similarity_threshold == original_threshold
    assert engine_instance._analysis_completed is True # Should NOT reset

def test_set_skip_semantic(engine_instance: KnowledgeDistillerEngine):
    """Test setting the skip_semantic flag."""
    engine_instance.skip_semantic = False
    engine_instance._analysis_completed = True
    engine_instance.set_skip_semantic(True)
    assert engine_instance.skip_semantic is True
    assert engine_instance._analysis_completed is False # Should reset analysis status
    engine_instance.set_skip_semantic(False)
    assert engine_instance.skip_semantic is False
    assert engine_instance._analysis_completed is False # Should reset analysis status

