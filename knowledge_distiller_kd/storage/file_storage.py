# knowledge_distiller_kd/storage/file_storage.py
"""
Handles reading and writing decision data to/from JSON files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# 导入项目内部模块 (使用相对路径)
from ..core.error_handler import FileOperationError, handle_error # 导入可能需要的错误类型
from ..core import constants # 导入常量

logger = logging.getLogger(constants.LOGGER_NAME)

class FileStorage:
    """
    Manages the persistence of block decisions using JSON files.
    """

    def __init__(self):
        """Initialize the FileStorage."""
        # Initialization logic if needed (e.g., setting default encoding)
        self.encoding = constants.DEFAULT_ENCODING

    def save_decisions(self, filepath: Path, decisions_data: List[Dict[str, Any]]) -> bool:
        """
        Saves the provided decision data to the specified JSON file.

        Args:
            filepath: The Path object representing the target JSON file.
            decisions_data: A list of dictionaries, where each dictionary
                            represents a block decision record.

        Returns:
            True if saving was successful, False otherwise.

        Raises:
            FileOperationError: If there's an issue writing the file (e.g., permissions).
        """
        logger.info(f"Attempting to save {len(decisions_data)} decisions to {filepath}")
        try:
            # --- Logic to be moved from KDToolCLI.save_decisions ---
            # 1. Ensure parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # 2. Write data to JSON file
            with open(filepath, 'w', encoding=self.encoding) as f:
                json.dump(decisions_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully saved decisions to {filepath}")
            return True
        except IOError as e:
            # Catch specific IO errors (like permission denied)
            err_msg = f"Error writing decisions to file {filepath}: {e}"
            logger.error(err_msg, exc_info=True)
            # Re-raise as a specific project error for consistent handling
            raise FileOperationError(err_msg, error_code="WRITE_FILE_FAILED") from e
        except Exception as e:
            # Catch any other unexpected errors during saving
            err_msg = f"Unexpected error saving decisions to {filepath}: {e}"
            logger.error(err_msg, exc_info=True)
            # Handle or re-raise as appropriate
            handle_error(e, f"Saving decisions to {filepath}") # Use generic handler
            return False # Indicate failure

    def load_decisions(self, filepath: Path) -> List[Dict[str, Any]]:
        """
        Loads decision data from the specified JSON file.

        Args:
            filepath: The Path object representing the source JSON file.

        Returns:
            A list of decision dictionaries loaded from the file.
            Returns an empty list if the file doesn't exist or is invalid.

        Raises:
            FileOperationError: If there's an issue reading the file (e.g., permissions).
        """
        logger.info(f"Attempting to load decisions from {filepath}")
        if not filepath.exists():
            logger.warning(f"Decision file not found: {filepath}. Returning empty list.")
            return []

        try:
            with open(filepath, 'r', encoding=self.encoding) as f:
                # --- Logic to be moved from KDToolCLI.load_decisions ---
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        logger.error(f"Invalid format in decision file {filepath}: Top level must be a list.")
                        return [] # Return empty list for invalid format
                    # Basic validation (optional, can add more checks if needed)
                    # for record in data:
                    #     if not isinstance(record, dict):
                    #         logger.warning(f"Skipping non-dict record in {filepath}: {record}")
                    #         # Decide whether to skip or fail entirely
                    logger.info(f"Successfully loaded {len(data)} decision records from {filepath}")
                    return data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON from {filepath}: {e}")
                    return [] # Return empty list for decode error
        except IOError as e:
            err_msg = f"Error reading decisions file {filepath}: {e}"
            logger.error(err_msg, exc_info=True)
            raise FileOperationError(err_msg, error_code="READ_FILE_FAILED") from e
        except Exception as e:
            err_msg = f"Unexpected error loading decisions from {filepath}: {e}"
            logger.error(err_msg, exc_info=True)
            handle_error(e, f"Loading decisions from {filepath}")
            return [] # Indicate failure by returning empty list

