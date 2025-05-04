# tests/test_constants.py
"""
Tests for constants defined in constants.py.
"""

import pytest
import logging
from pathlib import Path
import os
from knowledge_distiller_kd.core import constants
from typing import Dict, Set, Tuple, Final # Import necessary types

# --- Test Basic String/Value Constants ---

def test_basic_string_constants():
    """Test basic string constants."""
    assert isinstance(constants.LOGGER_NAME, str)
    assert constants.LOGGER_NAME == "kd_tool"

    assert isinstance(constants.DEFAULT_LOG_DIR, str)
    assert constants.DEFAULT_LOG_DIR == "logs"

    assert isinstance(constants.DEFAULT_LOG_FILE, str)
    assert constants.DEFAULT_LOG_FILE == f"{constants.LOGGER_NAME}.log"

    assert isinstance(constants.LOG_FORMAT, str)
    assert isinstance(constants.LOG_DATE_FORMAT, str)

    assert isinstance(constants.DEFAULT_ENCODING, str)
    assert constants.DEFAULT_ENCODING == "utf-8"

    assert isinstance(constants.DEFAULT_OUTPUT_DIR, str)
    assert constants.DEFAULT_OUTPUT_DIR == "output"

    assert isinstance(constants.DEFAULT_OUTPUT_SUFFIX, str)
    assert constants.DEFAULT_OUTPUT_SUFFIX == "_deduped"

    assert isinstance(constants.DECISION_KEEP, str)
    assert constants.DECISION_KEEP == "keep"
    assert isinstance(constants.DECISION_DELETE, str)
    assert constants.DECISION_DELETE == "delete"
    assert isinstance(constants.DECISION_UNDECIDED, str)
    assert constants.DECISION_UNDECIDED == "undecided"

    assert isinstance(constants.DEFAULT_DECISION_DIR, str)
    assert constants.DEFAULT_DECISION_DIR == "decisions"

    # Check DEFAULT_DECISION_FILE construction
    assert isinstance(constants.DEFAULT_DECISION_FILE, str)
    # Use os.path.join for platform-independent comparison if needed, or assert exact string
    assert constants.DEFAULT_DECISION_FILE == os.path.join(constants.DEFAULT_DECISION_DIR, "decisions.json")

    assert isinstance(constants.DECISION_KEY_SEPARATOR, str)
    assert constants.DECISION_KEY_SEPARATOR == "::"

    assert isinstance(constants.MD5_HASH_ALGORITHM, str)
    assert constants.MD5_HASH_ALGORITHM == "md5"

    assert isinstance(constants.VERSION, str) # Check version constant

# --- Test Dictionary Constants ---

def test_log_level_map():
    """Test the LOG_LEVEL_MAP dictionary."""
    assert isinstance(constants.LOG_LEVEL_MAP, dict)
    assert "INFO" in constants.LOG_LEVEL_MAP
    assert constants.LOG_LEVEL_MAP["INFO"] == logging.INFO
    assert constants.LOG_LEVEL_MAP["DEBUG"] == logging.DEBUG
    # Add more checks if needed

def test_error_codes_dict():
    """Test the ERROR_CODES dictionary."""
    assert isinstance(constants.ERROR_CODES, dict)
    # Check for existence of some keys and type of values
    assert "FILE_NOT_FOUND" in constants.ERROR_CODES
    assert isinstance(constants.ERROR_CODES["FILE_NOT_FOUND"], int)
    assert "UNEXPECTED_ERROR" in constants.ERROR_CODES
    assert isinstance(constants.ERROR_CODES["UNEXPECTED_ERROR"], int)
    # Check if values are unique (optional)
    # assert len(constants.ERROR_CODES.values()) == len(set(constants.ERROR_CODES.values()))

# --- Test Set/Tuple Constants ---

def test_supported_extensions():
    """Test the SUPPORTED_EXTENSIONS set."""
    assert isinstance(constants.SUPPORTED_EXTENSIONS, set)
    assert ".md" in constants.SUPPORTED_EXTENSIONS
    assert ".markdown" in constants.SUPPORTED_EXTENSIONS
    assert len(constants.SUPPORTED_EXTENSIONS) == 2

# --- Test Analysis Constants ---

def test_semantic_constants():
    """Test semantic analysis related constants."""
    assert isinstance(constants.DEFAULT_SEMANTIC_MODEL, str)
    assert len(constants.DEFAULT_SEMANTIC_MODEL) > 0

    assert isinstance(constants.DEFAULT_SIMILARITY_THRESHOLD, float)
    assert 0.0 <= constants.DEFAULT_SIMILARITY_THRESHOLD <= 1.0

    assert isinstance(constants.DEFAULT_BATCH_SIZE, int)
    assert constants.DEFAULT_BATCH_SIZE > 0

def test_cache_constants():
    """Test cache related constants."""
    # Check CACHE_DIR
    assert isinstance(constants.CACHE_DIR, str)
    assert constants.CACHE_DIR == "cache" # Check the actual value

    # Check DEFAULT_CACHE_BASE_DIR
    assert hasattr(constants, 'DEFAULT_CACHE_BASE_DIR') # Check existence
    assert isinstance(constants.DEFAULT_CACHE_BASE_DIR, str)
    assert constants.DEFAULT_CACHE_BASE_DIR == ".kd_cache" # Check the value

    # Check VECTOR_CACHE_FILE
    assert isinstance(constants.VECTOR_CACHE_FILE, str)
    assert constants.VECTOR_CACHE_FILE.endswith(".pkl")
    # Check relationship using os.path.join for platform independence
    assert constants.VECTOR_CACHE_FILE == os.path.join(constants.CACHE_DIR, "vector_cache.pkl")

# --- Test Block Type Constants ---

def test_block_type_constants():
    """Test the defined block type constants."""
    # List the constants ACTUALLY defined in constants.py
    defined_block_types = [
        ('BLOCK_TYPE_TITLE', 'Title'),
        ('BLOCK_TYPE_TEXT', 'NarrativeText'), # Value is "NarrativeText"
        ('BLOCK_TYPE_LIST_ITEM', 'ListItem'),
        ('BLOCK_TYPE_CODE', 'CodeSnippet'),
        ('BLOCK_TYPE_TABLE', 'Table'),
    ]
    # Check that each defined constant exists and has the correct string value
    for name, expected_value in defined_block_types:
        assert hasattr(constants, name)
        value = getattr(constants, name)
        assert isinstance(value, str)
        assert value == expected_value

# --- Test UI Constants ---
def test_ui_constants():
    """Test UI related constants."""
    assert isinstance(constants.PREVIEW_LENGTH, int)
    assert constants.PREVIEW_LENGTH > 0

# --- Removed tests for constants that no longer exist ---
# 可以添加对其他 utils 函数的测试，例如：
# def test_find_markdown_files(...)
# def test_calculate_md5(...)