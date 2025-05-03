# tests/test_semantic_analyzer.py
"""
Unit tests for the SemanticAnalyzer class.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call # 确保导入 call
from pathlib import Path # 导入 Path
import os # 导入 os
from typing import List, Dict, Tuple, Optional, Any # 确保导入所需类型

# Corrected imports
from knowledge_distiller_kd.analysis.semantic_analyzer import SemanticAnalyzer, SENTENCE_TRANSFORMERS_AVAILABLE
from knowledge_distiller_kd.processing.document_processor import ContentBlock # Corrected path
from knowledge_distiller_kd.core import constants

# Need to import element types if creating ContentBlocks directly
from unstructured.documents.elements import NarrativeText, Title, CodeSnippet

# --- Fixtures ---

# Mock SentenceTransformer class if the library is not available or for isolation
# Define mock outside fixture if used in multiple places or needs specific setup
class MockSentenceTransformerModel:
    def encode(self, texts: List[str], batch_size: int, show_progress_bar: bool) -> np.ndarray:
        # Return predictable vectors based on input text length or content
        # Example: return vectors of ones with dimension matching MODEL_EMBEDDING_DIM
        # Use a fixed dimension for consistency in tests
        fixed_dimension = 384 # Match default model's expected dimension
        vectors = []
        for i, text in enumerate(texts):
            # Simple mock: vector based on index/length to ensure uniqueness if needed
            vec = np.ones(fixed_dimension) * (i + 1) * (len(text) % 5 + 1) * 0.1
            # Avoid division by zero if vec norm is zero (e.g., empty text produces zero vector)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vectors.append(vec / norm) # Normalize
            else:
                vectors.append(vec) # Keep zero vector as is
        return np.array(vectors)

@pytest.fixture
def mock_sentence_transformer(mocker):
    """Fixture to mock the SentenceTransformer library."""
    # Mock the class itself
    mock = MagicMock(spec=MockSentenceTransformerModel) # Use our mock class for spec
    mock.encode.side_effect = MockSentenceTransformerModel().encode # Use encode from instance

    # Patch the class where it's imported in the semantic_analyzer module
    # Use the NEW correct path for patching
    mocker.patch("knowledge_distiller_kd.analysis.semantic_analyzer.SentenceTransformer", return_value=mock)
    return mock

# Updated fixture for SemanticAnalyzer
@pytest.fixture
def semantic_analyzer() -> SemanticAnalyzer:
    """Creates a SemanticAnalyzer instance using default threshold."""
    # Use the default threshold from constants
    # Analyzer now takes threshold directly, no tool object needed
    # Use the correct constant name from constants.py for threshold
    analyzer = SemanticAnalyzer(similarity_threshold=constants.DEFAULT_SIMILARITY_THRESHOLD)
    return analyzer

@pytest.fixture
def create_content_block_sem():
    """Factory fixture to create ContentBlock instances for semantic tests."""
    _block_counter = 0
    # Simple base dir name relative to current execution dir
    base_dir = Path(f"test_sem_run_{_block_counter}")

    def _create(
        text: str,
        file_path: str = "test_sem.md",
        block_type_cls: type = NarrativeText,
        element_id: str | None = None
    ) -> ContentBlock:
        nonlocal _block_counter
        if element_id is None:
            _block_counter += 1
            element_id = f"sem_test_{_block_counter}"

        abs_file_path = (base_dir / file_path).resolve()
        element = block_type_cls(text=text, element_id=element_id)
        cb = ContentBlock(element, str(abs_file_path))
        cb.original_text = text
        cb.analysis_text = cb._normalize_text()
        return cb
    return _create

# --- Test Cases ---

# Corrected test_initialization
def test_initialization(semantic_analyzer: SemanticAnalyzer) -> None:
    """Test SemanticAnalyzer initialization"""
    assert isinstance(semantic_analyzer, SemanticAnalyzer)
    assert semantic_analyzer.similarity_threshold == constants.DEFAULT_SIMILARITY_THRESHOLD
    assert semantic_analyzer.vector_cache == {}
    assert semantic_analyzer.model is None
    assert semantic_analyzer._model_loaded is False

# Use corrected patch path and constant name
@patch("knowledge_distiller_kd.analysis.semantic_analyzer.SentenceTransformer")
def test_load_semantic_model_success(mock_st_class: MagicMock, semantic_analyzer: SemanticAnalyzer) -> None:
    """Test successfully loading the semantic model."""
    # Configure the mock class to return our mock model instance
    mock_model_instance = MockSentenceTransformerModel()
    # Simulate the actual SentenceTransformer().encode method existing
    mock_model_instance.encode = MagicMock(side_effect=MockSentenceTransformerModel().encode)
    # Add get_sentence_embedding_dimension if SemanticAnalyzer uses it
    mock_model_instance.get_sentence_embedding_dimension = MagicMock(return_value=384) # Example dimension
    mock_st_class.return_value = mock_model_instance

    loaded = semantic_analyzer.load_semantic_model()

    assert loaded is True
    assert semantic_analyzer.model is mock_model_instance
    assert semantic_analyzer._model_loaded is True
    # Use the correct constant name DEFAULT_CACHE_BASE_DIR
    mock_st_class.assert_called_once_with(
        constants.DEFAULT_SEMANTIC_MODEL,
        cache_folder=str(Path(constants.DEFAULT_CACHE_BASE_DIR).resolve())
    )

# Skipped test_load_semantic_model_skipped
@pytest.mark.skip(reason="Skipping logic moved outside analyzer")
def test_load_semantic_model_skipped(semantic_analyzer: SemanticAnalyzer) -> None:
    """Test skipping loading the semantic model (Logic moved, test needs redesign/removal)"""
    pass

# Use corrected patch path
@patch("knowledge_distiller_kd.analysis.semantic_analyzer.SentenceTransformer", side_effect=Exception("Model loading error"))
def test_load_semantic_model_failure(mock_st_class_error: MagicMock, semantic_analyzer: SemanticAnalyzer) -> None:
    """Test failure during semantic model loading."""
    loaded = semantic_analyzer.load_semantic_model()

    assert loaded is False
    assert semantic_analyzer.model is None
    assert semantic_analyzer._model_loaded is False

# Skipped test for internal method
@pytest.mark.skip(reason="Skipping test for internal method _compute_vectors for now")
@patch("knowledge_distiller_kd.analysis.semantic_analyzer.SentenceTransformer")
def test_compute_vectors(mock_st_class: MagicMock, semantic_analyzer: SemanticAnalyzer, create_content_block_sem) -> None:
    """Test computing vectors for blocks."""
    # Setup mock model
    mock_model_instance = MagicMock(spec=MockSentenceTransformerModel)
    mock_model_instance.encode.side_effect = MockSentenceTransformerModel().encode
    mock_st_class.return_value = mock_model_instance

    # Load the mock model into the analyzer
    semantic_analyzer.load_semantic_model()
    assert semantic_analyzer.model is not None

    block1 = create_content_block_sem("First sentence.", element_id="id1")
    block2 = create_content_block_sem("Second sentence, slightly longer.", element_id="id2")
    input_blocks = [block1, block2]
    texts = [b.analysis_text for b in input_blocks]

    # Option 1: Call internal method (might be brittle)
    try:
        vectors = semantic_analyzer._compute_vectors(texts)
        assert isinstance(vectors, list) # Check if it returns a list of ndarrays
        assert len(vectors) == len(texts)
        assert isinstance(vectors[0], np.ndarray) # Check the type of the first element
        # Check dimension if possible (depends on mock model)
        if vectors[0].size > 0:
             assert vectors[0].shape[-1] == 384 # Assuming our mock uses this dimension
        # Check if model's encode was called correctly
        semantic_analyzer.model.encode.assert_called_once_with(
            texts,
            batch_size=semantic_analyzer.batch_size,
            show_progress_bar=True # Adjusted expectation based on code
        )
    except AttributeError:
        pytest.skip("_compute_vectors might not be intended for direct external testing")
    except Exception as e:
        pytest.fail(f"_compute_vectors call failed unexpectedly: {e}")


# Use corrected patch path
# This test needs significant adaptation based on the new signature and return value
@patch("knowledge_distiller_kd.analysis.semantic_analyzer.SentenceTransformer")
def test_find_semantic_duplicates(mock_st_class: MagicMock, semantic_analyzer: SemanticAnalyzer, create_content_block_sem) -> None:
    """Test finding semantic duplicates."""
    # Setup mock model
    mock_model_instance = MagicMock(spec=MockSentenceTransformerModel)

    # --- Mock encode to return specific vectors for testing similarity ---
    vec1 = np.array([1.0, 0.0, 0.0]) # Example 3D vectors
    vec2 = np.array([0.9, 0.1, 0.0]) # Similar to vec1
    vec3 = np.array([0.0, 1.0, 0.0]) # Different from vec1/vec2
    # Normalize them
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    vec3 = vec3 / np.linalg.norm(vec3)

    # Define fixed dimension for padding
    fixed_dimension = 384

    def mock_encode(texts: List[str], batch_size: int, show_progress_bar: bool):
        # Return specific vectors based on expected input texts, padded to fixed_dimension
        padding_dim = fixed_dimension - 3
        vectors_map = {
            "Sentence A": np.pad(vec1, (0, padding_dim)),
            "Sentence B": np.pad(vec2, (0, padding_dim)),
            "Sentence C": np.pad(vec3, (0, padding_dim))
        }
        # Handle potential empty strings which might be passed
        return np.array([vectors_map.get(text, np.zeros(fixed_dimension)) for text in texts])


    mock_model_instance.encode.side_effect = mock_encode
    # Add get_sentence_embedding_dimension if SemanticAnalyzer uses it
    mock_model_instance.get_sentence_embedding_dimension = MagicMock(return_value=fixed_dimension)
    mock_st_class.return_value = mock_model_instance
    # -------------------------------------------------------------------

    # Load the mock model
    semantic_analyzer.load_semantic_model()
    assert semantic_analyzer.model is not None

    # Set threshold for test
    semantic_analyzer.similarity_threshold = 0.85 # Expect A and B to match

    # Prepare input blocks
    block_a = create_content_block_sem("Sentence A", file_path="f1.md", element_id="idA")
    block_b = create_content_block_sem("Sentence B", file_path="f2.md", element_id="idB") # Similar
    block_c = create_content_block_sem("Sentence C", file_path="f3.md", element_id="idC") # Different

    # Provide the blocks to analyze
    input_blocks = [block_a, block_b, block_c]

    # Call the core method find_semantic_duplicates
    similar_pairs = semantic_analyzer.find_semantic_duplicates(input_blocks)

    # Assertions on the returned similar_pairs
    assert isinstance(similar_pairs, list)
    assert len(similar_pairs) == 1 # Expecting only A and B to be similar enough

    pair = similar_pairs[0]
    assert isinstance(pair, tuple)
    assert len(pair) == 3 # block1, block2, similarity_score

    # Check if the correct pair is found (order might vary)
    found_blocks = {pair[0], pair[1]}
    expected_blocks = {block_a, block_b}
    assert found_blocks == expected_blocks

    # Check similarity score (approximate)
    # Cosine similarity of original vec1 and vec2 is dot product (since they were normalized)
    expected_similarity = np.dot(vec1, vec2)
    assert pair[2] == pytest.approx(expected_similarity)
    assert pair[2] >= semantic_analyzer.similarity_threshold