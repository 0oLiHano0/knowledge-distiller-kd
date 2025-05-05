# knowledge_distiller_kd/analysis/md5_analyzer.py
"""
MD5 Analyzer module for detecting exact duplicate content blocks.
Refactored to use ContentBlock DTO from core.models and be independent.
Version 3: Ensures only ContentBlockDTO from core.models is used.
"""
# --- Standard Library Imports ---
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, DefaultDict
from collections import defaultdict

# --- Project Internal Imports ---
# Use relative imports (..) to access core modules
from ..core.error_handler import handle_error, AnalysisError
from ..core.utils import logger, create_decision_key # Import necessary utils
# Import the correct DTO and Enums from core.models
from ..core.models import ContentBlock as ContentBlockDTO, BlockType
# REMOVED incorrect import: from ..processing.document_processor import ContentBlock

# --- Constants for Decisions (Align with Engine's usage) ---
DECISION_KEEP = 'keep'
DECISION_DELETE = 'delete'
DECISION_UNDECIDED = 'undecided'

# --- Class Definition ---

class MD5Analyzer:
    """
    MD5 Analyzer class for detecting identical content blocks based on MD5 hash.
    Operates exclusively on ContentBlockDTO objects from core.models.
    """

    def __init__(self):
        """Initializes the MD5Analyzer."""
        pass # No state needed currently

    def find_md5_duplicates(
        self,
        blocks_data: List[ContentBlockDTO], # Expect DTO type
        current_decisions: Dict[str, str]
    ) -> Tuple[List[List[ContentBlockDTO]], Dict[str, str]]: # Return DTOs
        """
        Finds content blocks with identical MD5 hashes.
        Skips specific block types (e.g., HEADING) and blocks already marked for deletion.

        Args:
            blocks_data: A list of ContentBlockDTO objects to analyze.
            current_decisions: A dictionary of current decisions {decision_key: decision_string}.

        Returns:
            A tuple containing:
                - A list of duplicate groups, where each group is a list of ContentBlockDTOs.
                - A dictionary of suggested decision updates {decision_key: decision_string}.
        """
        logger.info("Starting MD5 duplicate detection...")
        duplicate_groups: List[List[ContentBlockDTO]] = []
        suggested_decisions: Dict[str, str] = {}

        if not blocks_data:
            logger.warning("MD5 analysis skipped: No blocks provided.")
            return duplicate_groups, suggested_decisions

        try:
            # Group blocks by MD5 hash
            hash_groups: DefaultDict[str, List[ContentBlockDTO]] = defaultdict(list)
            total_blocks = len(blocks_data)
            processed_count = 0
            skipped_headings = 0
            skipped_code_fences = 0
            skipped_deleted = 0
            skipped_no_path = 0
            skipped_type_error = 0

            logger.info(f"Starting MD5 duplicate detection for {total_blocks} blocks.")
            logger.debug(f"Number of current decisions provided: {len(current_decisions)}")

            # Calculate hash for each block
            for block in blocks_data:
                # *** STRONGLY CHECK for ContentBlockDTO from core.models ***
                if not isinstance(block, ContentBlockDTO):
                    logger.warning(f"Skipping item as it's not a ContentBlockDTO from core.models: {type(block)}")
                    skipped_type_error += 1
                    continue

                # Skip HEADING blocks (using the correct Enum from models)
                if block.block_type == BlockType.HEADING:
                    original_path = block.metadata.get('original_path', 'UnknownPath')
                    logger.debug(f"Skipping Heading block for MD5 analysis: {original_path}#{block.block_id}")
                    skipped_headings += 1
                    continue

                # --- Check current decision ---
                original_path = block.metadata.get('original_path')
                if not original_path:
                    logger.warning(f"Block {block.block_id} missing 'original_path' in metadata. Cannot check decision or create key.")
                    skipped_no_path += 1
                    continue

                try:
                    key = create_decision_key(
                        str(Path(original_path).resolve()), # Resolve for consistency
                        block.block_id,
                        block.block_type.value # Use enum value
                    )
                except Exception as e:
                    logger.error(f"Error creating decision key for block {block.block_id} in {original_path}: {e}")
                    continue

                decision = current_decisions.get(key, DECISION_UNDECIDED)
                if decision == DECISION_DELETE:
                    logger.debug(f"Skipping block already marked for deletion: {key}")
                    skipped_deleted += 1
                    continue
                # --------------------

                # Use the 'text' attribute from the DTO
                text_to_hash = block.text

                # Skip blocks containing only code fence end
                if block.block_type == BlockType.CODE and text_to_hash.strip() == '```':
                    logger.debug(f"Skipping block containing only code fence end: {key}")
                    skipped_code_fences += 1
                    continue

                # Calculate MD5 hash
                if not isinstance(text_to_hash, str):
                    logger.warning(f"Text for block {key} is not a string: {type(text_to_hash)}. Skipping.")
                    continue

                block_type_str = block.block_type.value # Use enum value
                hash_input = f"{block_type_str}:{text_to_hash.strip()}"
                try:
                    md5_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
                    hash_groups[md5_hash].append(block) # Add the DTO object
                except Exception as e:
                     logger.error(f"Error calculating hash for block {key}: {e}")

                processed_count += 1

            logger.info(f"MD5 hash calculation complete. Processed: {processed_count}. Skipped: "
                        f"Headings={skipped_headings}, CodeFences={skipped_code_fences}, "
                        f"Deleted={skipped_deleted}, NoPath={skipped_no_path}, TypeErrors={skipped_type_error}")
            logger.info(f"Hash calculation resulted in {len(hash_groups)} unique hash groups.")

            # Identify duplicate groups and generate suggested decisions
            duplicate_group_count = 0
            for md5_hash, duplicate_block_list in hash_groups.items():
                if len(duplicate_block_list) > 1:
                    duplicate_group_count += 1
                    logger.info(f"Found MD5 duplicate group {duplicate_group_count}, hash: {md5_hash}, blocks: {len(duplicate_block_list)}")

                    try:
                        duplicate_block_list.sort(key=lambda b: (
                            str(Path(b.metadata.get('original_path', '')).resolve()),
                            str(b.block_id)
                        ))
                    except Exception as e:
                         logger.error(f"Error sorting duplicate blocks for hash {md5_hash}: {e}. Skipping suggestions for this group.")
                         continue

                    duplicate_groups.append(duplicate_block_list)

                    # --- Generate Decision Suggestions ---
                    if not duplicate_block_list: continue

                    first_block = duplicate_block_list[0]
                    first_block_path = first_block.metadata.get('original_path')
                    if not first_block_path: continue

                    first_key = create_decision_key(str(Path(first_block_path).resolve()), first_block.block_id, first_block.block_type.value)
                    if current_decisions.get(first_key, DECISION_UNDECIDED) != DECISION_DELETE:
                        suggested_decisions[first_key] = DECISION_KEEP
                        logger.debug(f"Suggesting KEEP for block: {first_key}")
                    else:
                         logger.debug(f"First block {first_key} in group {md5_hash} was already marked delete, not suggesting KEEP.")

                    for block_to_delete in duplicate_block_list[1:]:
                        delete_block_path = block_to_delete.metadata.get('original_path')
                        if not delete_block_path: continue

                        delete_key = create_decision_key(str(Path(delete_block_path).resolve()), block_to_delete.block_id, block_to_delete.block_type.value)
                        if current_decisions.get(delete_key, DECISION_UNDECIDED) == DECISION_UNDECIDED:
                            suggested_decisions[delete_key] = DECISION_DELETE
                            logger.debug(f"Suggesting DELETE for block: {delete_key}")
                        else:
                             logger.debug(f"Block {delete_key} in group {md5_hash} already has a decision ({current_decisions.get(delete_key)}), not suggesting DELETE.")
                    # --------------------

            if duplicate_groups:
                logger.info(f"MD5 analysis finished. Found {len(duplicate_groups)} groups of exact duplicates (excluding headings).")
            else:
                logger.info("MD5 analysis finished. No exact duplicates found (excluding headings).")

            return duplicate_groups, suggested_decisions

        except Exception as e:
            logger.error(f"Unexpected error during MD5 duplicate finding: {e}", exc_info=True)
            handle_error(e, "finding MD5 duplicates")
            return [], {}
