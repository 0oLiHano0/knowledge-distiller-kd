# knowledge_distiller_kd/core/engine.py
"""
Core engine for the Knowledge Distiller tool.
Encapsulates business logic, state management, and process orchestration.
Refactored to use StorageInterface via dependency injection.
Phase 5: Refactored run_analysis orchestration logic.
Version 7: Corrected DTO conversion logic robustly, fixed status summary attribute,
           adjusted error returns, corrected _initialize_decisions logic.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set, DefaultDict
from collections import defaultdict
import uuid # Import uuid for generating block IDs if needed
import re

# Import internal project modules (using relative paths)
from . import constants
from .error_handler import KDError, ConfigurationError, handle_error, validate_file_path, FileOperationError, AnalysisError
from .utils import logger, create_decision_key, parse_decision_key
from ..storage.storage_interface import StorageInterface # Use the interface
# Import DTOs/Enums from core.models (use final confirmed version)
from ..core.models import (
    ContentBlock as ContentBlockDTO,
    UserDecision as UserDecisionDTO,
    AnalysisResult as AnalysisResultDTO,
    FileRecord as FileRecordDTO,
    AnalysisType, DecisionType, BlockType # Ensure BlockType is imported
)
from ..analysis.md5_analyzer import MD5Analyzer
from ..analysis.semantic_analyzer import SemanticAnalyzer
# Keep old ContentBlock import ONLY if process_directory still returns it.
from ..processing.document_processor import ContentBlock as OldContentBlock, process_directory, DocumentProcessingError
# Assume merge_code_blocks works with OldContentBlock for now, needs review.
from ..processing.block_merger import merge_code_blocks

# --- Constants for Decisions ---
DECISION_KEEP = 'keep'
DECISION_DELETE = 'delete'
DECISION_UNDECIDED = 'undecided'
METADATA_DECISION_KEY = 'kd_processing_status'


class KnowledgeDistillerEngine:
    """
    The core engine responsible for analysis, decision management, and state.
    Uses StorageInterface for persistence.
    """

    def __init__(
        self,
        storage: StorageInterface,
        input_dir: Optional[Union[str, Path]] = None,
        decision_file: Optional[Union[str, Path]] = None, # Config path name remains for consistency
        output_dir: Optional[Union[str, Path]] = None,  # Config path name remains for consistency
        skip_semantic: bool = False,
        similarity_threshold: float = constants.DEFAULT_SIMILARITY_THRESHOLD
    ):
        logger.info("Initializing KnowledgeDistillerEngine...")
        self.storage: StorageInterface = storage

        self.input_dir: Optional[Path] = None
        if input_dir:
            try:
                validated_input_dir = validate_file_path(Path(input_dir), must_exist=True)
                if not validated_input_dir.is_dir():
                    raise ConfigurationError(f"Input path is not a directory: {validated_input_dir}")
                self.input_dir = validated_input_dir
            except (ConfigurationError, FileOperationError) as e:
                 logger.error(f"Invalid input directory provided during initialization: {e}")
                 raise ConfigurationError(f"Engine initialization failed due to invalid input directory: {e}") from e

        # Store config paths with corrected internal names
        self.decision_file_config_path: Path = Path(decision_file or constants.DEFAULT_DECISION_FILE).resolve()
        self.output_dir_config_path: Path = Path(output_dir or constants.DEFAULT_OUTPUT_DIR).resolve()

        self.skip_semantic: bool = skip_semantic
        self.similarity_threshold: float = max(0.0, min(1.0, similarity_threshold))

        # Internal state for analysis run
        self.blocks_data: List[ContentBlockDTO] = []
        self.block_decisions: Dict[str, str] = {}
        self.md5_duplicates: List[List[ContentBlockDTO]] = []
        self.semantic_duplicates: List[Tuple[ContentBlockDTO, ContentBlockDTO, float]] = []

        # Status flags
        self._decisions_loaded: bool = False
        self._analysis_completed: bool = False

        # Analyzers
        try:
            self.md5_analyzer = MD5Analyzer()
            self.semantic_analyzer = SemanticAnalyzer(similarity_threshold=self.similarity_threshold)
        except Exception as e:
            logger.critical(f"Failed to initialize analyzers: {e}", exc_info=True)
            raise ConfigurationError(f"Engine initialization failed due to analyzer error: {e}") from e
        logger.info("KnowledgeDistillerEngine initialized successfully.")

    def _reset_state(self) -> None:
        """Resets the internal state related to a specific analysis run."""
        logger.debug("Resetting engine analysis state...")
        self.blocks_data.clear()
        self.block_decisions.clear()
        self.md5_duplicates.clear()
        self.semantic_duplicates.clear()
        self._decisions_loaded = False
        self._analysis_completed = False
        logger.debug("Engine analysis state reset.")

    def set_input_dir(self, input_dir: Union[str, Path]) -> bool:
        """Sets the input directory and resets the engine analysis state."""
        logger.info(f"Attempting to set input directory to: {input_dir}")
        try:
            input_path = Path(input_dir)
            resolved_path = validate_file_path(input_path, must_exist=True)
            if not resolved_path.is_dir():
                logger.error(f"Setting input directory failed: '{resolved_path}' is not a directory.")
                print(f"[Error] Path '{resolved_path}' is not a valid directory.")
                return False
            logger.info(f"Input directory set to: {resolved_path}")
            self.input_dir = resolved_path
            self._reset_state()
            print(f"[*] Input directory set to: {self.input_dir}")
            return True
        except (FileOperationError, ConfigurationError) as e:
            handle_error(e, "setting input directory"); print(f"[Error] Error setting input directory: {e}"); return False
        except Exception as e:
            handle_error(e, "setting input directory"); print(f"[Error] Unexpected error setting input directory: {e}"); return False

    def run_analysis(self) -> bool:
        """
        Orchestrates the full analysis workflow. Returns True if successful, False otherwise.
        """
        if not self.input_dir:
            logger.error("Analysis aborted: Input directory not set.")
            print("[Error] Input directory not set.")
            return False
        logger.info(f"--- Starting analysis for folder: {self.input_dir} ---")
        print(f"\n[*] Starting analysis for folder: {self.input_dir}")

        self._reset_state()
        analysis_successful = True

        try:
            print("\n[*] Step 1: Processing documents & saving initial blocks...")
            # _process_documents now returns False if critical errors occur
            if not self._process_documents():
                 # Error already logged in _process_documents
                 raise AnalysisError("Document processing failed critically.")
            if not self.blocks_data:
                 logger.warning("No blocks were processed. Analysis may not yield results.")
                 print("[Warning] No content blocks were found or processed.")
                 # Continue analysis, but MD5/Semantic might do nothing.
            print(f"[*] Step 1 complete. ({len(self.blocks_data)} blocks processed)")

            print("\n[*] Step 2: Merging code blocks (in-memory)...")
            if not self._merge_code_blocks_step():
                logger.warning("Code block merging step failed or skipped, continuing analysis...")
                print("[Warning] Code block merging skipped or failed, analysis will continue.")
            else:
                print("[*] Step 2 complete.")

            print("\n[*] Step 3: Loading/Initializing decisions...")
            if not self.load_decisions():
                 logger.info("No prior decisions loaded from storage. Initializing defaults.")
                 # Initialize ensures the map exists even if loading fails or returns false
            self._initialize_decisions() # Initialize defaults for any blocks missing in map
            print(f"[*] Step 3 complete. ({len(self.block_decisions)} block decisions mapped in memory)")

            print("\n[*] Step 4: MD5 Deduplication...")
            md5_duplicates_found, suggested_md5_decisions = self.md5_analyzer.find_md5_duplicates(
                self.blocks_data, self.block_decisions
            )
            self.md5_duplicates = md5_duplicates_found
            self._update_decisions_from_md5(suggested_md5_decisions)
            print(f"[*] Step 4 complete: MD5 Deduplication ({len(self.md5_duplicates)} duplicate groups found)")

            if not self.skip_semantic:
                print("\n[*] Step 5a: Loading semantic model...")
                model_loaded = self.semantic_analyzer.load_semantic_model()
                if not model_loaded or not self._model_loaded_successfully():
                    logger.warning("Semantic model failed to load or unavailable. Skipping semantic analysis.")
                    print("[Warning] Semantic model failed to load or unavailable. Skipping semantic analysis.")
                    self.skip_semantic = True
                else:
                    print("[*] Step 5a complete: Loading semantic model")
                    if self._model_loaded_successfully():
                        print("\n[*] Step 5b: Semantic Deduplication...")
                        blocks_for_semantic = self._filter_blocks_for_semantic()
                        if blocks_for_semantic:
                            self.semantic_duplicates = self.semantic_analyzer.find_semantic_duplicates(blocks_for_semantic)
                            print(f"[*] Step 5b complete: Semantic Deduplication ({len(self.semantic_duplicates)} similar pairs found)")
                        else:
                            print("[*] Step 5b complete: No suitable blocks for Semantic Deduplication.")
                            self.semantic_duplicates = []
                    else:
                        logger.info("Skipping semantic deduplication step as model is not loaded.")
                        print("[*] Skipping step: Semantic Deduplication (model not loaded).")
            else:
                logger.info("Skipping semantic analysis steps as configured.")
                print("[*] Skipping steps: Loading semantic model, Semantic Deduplication.")

            self._analysis_completed = True
            logger.info("Analysis process completed successfully.")
            print("\n[*] Analysis workflow completed.")

        except AnalysisError as ae:
            logger.error(f"Analysis process failed: {ae}", exc_info=False)
            print(f"\n[Error] Analysis failed: {ae}")
            analysis_successful = False; self._analysis_completed = False
        except Exception as e:
            handle_error(e, "running analysis workflow")
            print(f"\n[Error] An unexpected error occurred during analysis: {e}")
            analysis_successful = False; self._analysis_completed = False

        return analysis_successful


    def _model_loaded_successfully(self) -> bool:
        """Checks if the semantic model is loaded and ready."""
        is_loaded = getattr(self.semantic_analyzer, '_model_loaded', False)
        model_exists = getattr(self.semantic_analyzer, 'model', None) is not None
        return not self.skip_semantic and is_loaded and model_exists

    def _filter_blocks_for_semantic(self) -> List[ContentBlockDTO]:
        """
        Filters blocks in memory (`self.blocks_data`) for semantic analysis.
        Excludes headings and blocks marked delete in `self.block_decisions`.
        """
        logger.debug("Filtering in-memory blocks for semantic analysis...")
        blocks_to_analyze: List[ContentBlockDTO] = []
        skipped_headings = 0; skipped_deleted = 0; skipped_no_path = 0

        if not self.blocks_data: logger.warning("No blocks to filter."); return []
        if not self._decisions_loaded: logger.warning("Decisions not loaded before filtering."); self._initialize_decisions()

        for block_dto in self.blocks_data:
            if block_dto.block_type == BlockType.HEADING: skipped_headings += 1; continue
            original_path = block_dto.metadata.get('original_path')
            if not original_path: logger.warning(f"Block {block_dto.block_id} missing path."); skipped_no_path += 1; continue
            try:
                 key = create_decision_key(str(Path(original_path).resolve()), block_dto.block_id, block_dto.block_type.value)
                 decision = self.block_decisions.get(key, DECISION_UNDECIDED)
                 if decision == DECISION_DELETE: skipped_deleted += 1; continue
            except Exception as e: logger.warning(f"Error checking decision for {block_dto.block_id}: {e}. Skipping."); continue
            blocks_to_analyze.append(block_dto)

        logger.info(f"Semantic filtering: Kept {len(blocks_to_analyze)}. Skipped: Headings={skipped_headings}, Deleted={skipped_deleted}, NoPath={skipped_no_path}.")
        return blocks_to_analyze

    def _process_documents(self) -> bool:
        """
        Processes documents, converts to DTOs, saves to storage, updates internal state.
        Version 7: Robust DTO conversion, corrected Enum usage, adjusted return logic.
        Returns False only if processing critically fails for all files.
        """
        if not self.input_dir: logger.error("Input dir not set."); return False
        logger.info(f"Processing documents in directory: {self.input_dir}")

        try:
            results: Dict[str, List[OldContentBlock]] = process_directory(self.input_dir, recursive=True)
            if not results: logger.warning(f"No processable files found in {self.input_dir}."); self.blocks_data.clear(); return True

            self.blocks_data.clear()
            all_processed_dtos: List[ContentBlockDTO] = []
            processed_files_count = 0; total_blocks_extracted = 0; file_processing_errors = 0
            processed_at_least_one_file_successfully = False

            for file_path_str, old_blocks in results.items():
                abs_file_path = Path(file_path_str).resolve()
                logger.debug(f"Processing file: {abs_file_path}")
                file_id: Optional[str] = None
                dtos_for_file: List[ContentBlockDTO] = []
                conversion_errors_in_file = False

                try:
                    file_id = self.storage.register_file(str(abs_file_path))
                    if not file_id: logger.error(f"Failed to register file {abs_file_path}. Skipping."); file_processing_errors += 1; continue

                    for i, old_block in enumerate(old_blocks):
                        # Create NEW dictionary for each block's metadata
                        current_block_metadata = {}
                        try:
                            # Get attributes safely
                            current_block_text = getattr(old_block, 'analysis_text', '')
                            if not isinstance(current_block_text, str): current_block_text = str(current_block_text)

                            # Handle metadata copy safely
                            source_metadata = getattr(old_block, 'metadata', {})
                            if isinstance(source_metadata, dict):
                                current_block_metadata = source_metadata.copy()
                            current_block_metadata['original_path'] = str(abs_file_path)

                            current_block_id = getattr(old_block, 'block_id', None)
                            if not current_block_id: current_block_id = str(uuid.uuid4())

                            # Map BlockType
                            element_type_name = "Unknown"
                            if hasattr(old_block, 'element') and hasattr(old_block.element, '__class__'):
                                element_type_name = old_block.element.__class__.__name__
                            # *** CORRECTED: Use BlockType.UNKNOWN ***
                            current_block_type_enum = BlockType.UNKNOWN
                            try:
                                found_type = False
                                for bt in BlockType: # Use corrected Enum from models
                                    if bt.value.lower() == element_type_name.lower():
                                        current_block_type_enum = bt; found_type = True; break
                                if not found_type:
                                     # Fallback mapping
                                     if "Title" in element_type_name or "Heading" in element_type_name: current_block_type_enum = BlockType.HEADING
                                     elif "ListItem" in element_type_name: current_block_type_enum = BlockType.LIST_ITEM
                                     elif "Code" in element_type_name: current_block_type_enum = BlockType.CODE
                                     elif "Table" in element_type_name: current_block_type_enum = BlockType.TABLE
                                     elif "Narrative" in element_type_name or "Text" in element_type_name: current_block_type_enum = BlockType.TEXT
                                     if current_block_type_enum == BlockType.UNKNOWN:
                                        logger.warning(f"Could not map element type '{element_type_name}'. Using {current_block_type_enum.name}.")
                                     else:
                                         logger.debug(f"Mapped element type '{element_type_name}' to {current_block_type_enum.name}")
                            except Exception as enum_map_err:
                                 logger.error(f"Error mapping BlockType for '{element_type_name}': {enum_map_err}. Using {current_block_type_enum.name}.")

                            # Create DTO with distinct values from this iteration
                            dto = ContentBlockDTO(
                                file_id=str(file_id), # Ensure file_id is str
                                text=str(current_block_text), # Ensure text is str
                                block_type=current_block_type_enum,
                                block_id=str(current_block_id), # Ensure id is str
                                metadata=dict(current_block_metadata) # Ensure metadata is dict
                            )
                            dtos_for_file.append(dto)
                            total_blocks_extracted += 1

                        except Exception as conversion_err:
                            logger.error(f"Error converting block {i} in {abs_file_path}: {conversion_err}", exc_info=True); conversion_errors_in_file = True; continue

                    # Save blocks for this file if any were converted successfully
                    if dtos_for_file:
                        logger.debug(f"Saving {len(dtos_for_file)} DTOs for file_id {file_id}")
                        self.storage.save_blocks(file_id=file_id, blocks=dtos_for_file)
                        all_processed_dtos.extend(dtos_for_file)
                        processed_files_count += 1
                        processed_at_least_one_file_successfully = True # Mark success
                    elif conversion_errors_in_file:
                        # File processed, but all block conversions failed
                        logger.error(f"All blocks failed conversion for file: {abs_file_path}")
                        file_processing_errors += 1
                    else:
                        logger.debug(f"No blocks extracted for file: {abs_file_path}")
                        # Consider this a successfully processed file (just empty)
                        processed_files_count += 1
                        processed_at_least_one_file_successfully = True


                except FileOperationError as storage_e: logger.error(f"Storage error for {abs_file_path}: {storage_e}. Skipping file."); file_processing_errors += 1; continue
                except Exception as file_proc_err: logger.error(f"Unexpected error processing file {abs_file_path}: {file_proc_err}", exc_info=True); file_processing_errors += 1; continue

            self.blocks_data = all_processed_dtos # Update engine state
            logger.info(f"Document processing complete. Files processed: {processed_files_count}, DTOs created: {total_blocks_extracted}, File errors: {file_processing_errors}.")
            # Return False only if NO files were successfully processed at all
            return processed_at_least_one_file_successfully

        except DocumentProcessingError as e: handle_error(e, "processing documents"); print(f"[Error] Processing documents: {e}"); return False
        except Exception as e: handle_error(e, "unexpected error during processing/storage"); print(f"[Error] Unexpected processing/storage error: {e}"); return False

    # ... (rest of the methods remain the same as engine_py_v4_final_fixes) ...
    # _merge_code_blocks_step, _initialize_decisions, _update_decisions_from_md5,
    # load_decisions, save_decisions, apply_decisions, get_md5_duplicates,
    # get_semantic_duplicates, update_decision, get_status_summary,
    # set_similarity_threshold, set_skip_semantic

    def _merge_code_blocks_step(self) -> bool:
        """Placeholder for merging code blocks (needs DTO refactor)."""
        if not self.blocks_data: logger.info("No blocks for code merging."); return True
        logger.info(f"Starting code block merging step for {len(self.blocks_data)} blocks...")
        try:
            logger.warning("Code block merging needs refactoring for DTOs. Skipping effective merge.")
            return True
        except Exception as e: handle_error(e, "merging code blocks"); print(f"[Error] Merging code blocks: {e}"); return False

    def _initialize_decisions(self) -> bool:
        """Initializes the in-memory decision map with defaults, avoiding overwrites."""
        if not self.blocks_data: logger.info("No blocks, skipping decision init."); self.block_decisions.clear(); return True
        logger.info(f"Initializing default decisions in memory map for {len(self.blocks_data)} blocks...")
        initialized_count = 0; error_count = 0; processed_count = 0
        # Do NOT clear existing map - allow load_decisions to populate first
        # self.block_decisions.clear()

        for block_dto in self.blocks_data:
             processed_count += 1
             original_path = block_dto.metadata.get('original_path')
             if not original_path: error_count += 1; logger.warning(f"Block {block_dto.block_id} missing 'original_path'."); continue
             try:
                 key = create_decision_key(str(Path(original_path).resolve()), block_dto.block_id, block_dto.block_type.value)
                 # *** FIXED: Set default ONLY if key does NOT exist ***
                 if key not in self.block_decisions:
                     self.block_decisions[key] = DECISION_UNDECIDED
                     initialized_count += 1
             except Exception as e: error_count += 1; logger.error(f"Failed key gen for block {block_dto.block_id}: {e}", exc_info=False)
        logger.info(f"Decision initialization complete. Processed: {processed_count}, Initialized defaults: {initialized_count}, Errors: {error_count}.")
        print(f"[*] Default decision init (memory): {initialized_count} initialized defaults, {error_count} errors.")
        # Mark decisions as loaded/initialized once done (might already be set by load_decisions)
        self._decisions_loaded = True
        return True

    def _update_decisions_from_md5(self, suggested_decisions: Dict[str, str]):
        """Applies MD5 suggestions to the in-memory decision map (only if undecided)."""
        if not suggested_decisions: logger.info("No MD5 suggestions."); return
        logger.info(f"Applying {len(suggested_decisions)} MD5 suggestions to in-memory map...")
        updated_count = 0; skipped_invalid = 0
        for key, suggested_decision in suggested_decisions.items():
            if suggested_decision in [DECISION_KEEP, DECISION_DELETE]:
                 if key in self.block_decisions:
                     if self.block_decisions[key] == DECISION_UNDECIDED:
                         self.block_decisions[key] = suggested_decision; updated_count += 1
                     else: logger.debug(f"Skipping MD5 suggestion for '{key}': already decided as '{self.block_decisions[key]}'.")
                 else: logger.warning(f"Skipping MD5 suggestion for unknown key '{key}'."); skipped_invalid += 1
            elif suggested_decision != DECISION_UNDECIDED: logger.warning(f"Skipping invalid MD5 suggestion value '{suggested_decision}' for key '{key}'"); skipped_invalid += 1
        logger.info(f"Applied {updated_count} MD5 suggestions to undecided blocks. Skipped invalid/existing: {skipped_invalid}.")

    def load_decisions(self) -> bool:
        """Loads decisions from storage block metadata into the in-memory map."""
        logger.info(f"Attempting to load decisions from storage block metadata ('{METADATA_DECISION_KEY}')...")
        self.block_decisions.clear(); self._decisions_loaded = False
        loaded_count = 0; error_count = 0; fetched_blocks: List[ContentBlockDTO] = []
        try:
            fetched_blocks = self.storage.get_blocks_for_analysis()
            if not fetched_blocks: logger.warning("Storage returned no blocks."); self._decisions_loaded = True; return False
            logger.info(f"Processing {len(fetched_blocks)} blocks for decisions...")
            for block_dto in fetched_blocks:
                # --- 添加日志 ---
                if block_dto.block_id == "a6fe03c62136119cbc37b307d4a6f509": # 只打印我们关心的块
                    logger.debug(f"Metadata loaded for block {block_dto.block_id} in load_decisions: {block_dto.metadata}")
                # --- 结束日志 ---
                original_path = block_dto.metadata.get('original_path')
                original_path = block_dto.metadata.get('original_path')
                if not original_path: logger.warning(f"Block {block_dto.block_id} missing 'original_path'."); error_count += 1; continue
                try:
                    key = create_decision_key(str(Path(original_path).resolve()), block_dto.block_id, block_dto.block_type.value)
                    decision = block_dto.metadata.get(METADATA_DECISION_KEY, DECISION_UNDECIDED)
                    if decision not in [DECISION_KEEP, DECISION_DELETE, DECISION_UNDECIDED]:
                        logger.warning(f"Invalid status '{decision}' for block {block_dto.block_id}. Using UNDECIDED."); decision = DECISION_UNDECIDED
                    self.block_decisions[key] = decision
                    if decision != DECISION_UNDECIDED: loaded_count += 1
                except Exception as e: logger.error(f"Error processing block {block_dto.block_id}: {e}", exc_info=False); error_count += 1; continue
            self._decisions_loaded = True
            logger.info(f"Decision loading complete: {loaded_count} explicit decisions loaded, {error_count} errors."); print(f"[*] Decisions loaded: Processed {len(fetched_blocks)} blocks.")
            return True
        except FileOperationError as e: handle_error(e, "loading blocks"); print(f"[Error] Storage error loading blocks: {e}"); self._decisions_loaded = False; return False
        except Exception as e: handle_error(e, "loading decisions"); print(f"[Error] Unexpected error loading decisions: {e}"); self._decisions_loaded = False; return False

    def save_decisions(self) -> bool:
        """Saves in-memory decisions back to storage via block metadata."""
        if not self.block_decisions: logger.warning("No decisions in memory map."); print("[!] No decisions to save."); return False
        logger.info(f"Saving {len(self.block_decisions)} decisions to storage via metadata ('{METADATA_DECISION_KEY}')...")
        updated_blocks_by_file_id: DefaultDict[str, List[ContentBlockDTO]] = defaultdict(list)
        processed_count = 0; error_count = 0; blocks_to_fetch_ids: Set[str] = set()
        for key in self.block_decisions:
            try: _, block_id, _ = parse_decision_key(key); blocks_to_fetch_ids.add(block_id)
            except Exception as e: logger.error(f"Error parsing key '{key}': {e}"); error_count += 1; continue
        if not blocks_to_fetch_ids: logger.error("No valid block IDs."); return False
        logger.debug(f"Fetching {len(blocks_to_fetch_ids)} blocks..."); fetched_blocks_map: Dict[str, ContentBlockDTO] = {}; fetch_errors = 0
        for block_id in blocks_to_fetch_ids:
             try: block = self.storage.get_block(block_id); fetched_blocks_map[block_id] = block if block else None
             except Exception as e: logger.error(f"Error fetching block {block_id}: {e}"); error_count += 1; fetch_errors += 1
        if fetch_errors > 0: logger.warning(f"{fetch_errors} errors fetching blocks.")
        blocks_requiring_save_count = 0
        for key, decision_to_save in self.block_decisions.items():
             processed_count += 1
             try:
                 _, block_id, _ = parse_decision_key(key)
                 block = fetched_blocks_map.get(block_id)
                 if block:
                     current_decision = block.metadata.get(METADATA_DECISION_KEY)
                     if current_decision != decision_to_save:
                          if not isinstance(block.metadata, dict): block.metadata = {}
                          block.metadata[METADATA_DECISION_KEY] = decision_to_save
                          if block.file_id: updated_blocks_by_file_id[block.file_id].append(block); blocks_requiring_save_count += 1
                          else: logger.warning(f"Block {block_id} missing file_id."); error_count += 1
                 else: logger.warning(f"Block {block_id} for key '{key}' not found during save step.")
             except Exception as e: logger.error(f"Error processing key '{key}' for saving: {e}"); error_count += 1
        if not updated_blocks_by_file_id: logger.info(f"No metadata updates needed. Processed: {processed_count}, Errors: {error_count}"); print("[*] No decision changes to save."); return True
        logger.info(f"Attempting to save {blocks_requiring_save_count} blocks with updated metadata..."); save_successful = True; files_saved_count = 0; save_errors = 0
        for file_id, blocks_to_save in updated_blocks_by_file_id.items():
             logger.debug(f"Saving {len(blocks_to_save)} blocks for file_id: {file_id}")
             try: self.storage.save_blocks(file_id=file_id, blocks=blocks_to_save); files_saved_count += 1
             except Exception as e: logger.error(f"Failed save for file_id {file_id}: {e}"); save_successful = False; save_errors += 1; error_count += len(blocks_to_save)
        total_errors = error_count
        if save_successful: logger.info(f"Successfully saved decisions for {blocks_requiring_save_count} blocks across {files_saved_count} files. Total errors: {total_errors}."); print(f"[*] Decisions saved for {blocks_requiring_save_count} blocks. Errors: {total_errors}.")
        else: logger.error(f"Errors saving decisions. Blocks needing save: {blocks_requiring_save_count}, File save errors: {save_errors}, Total errors: {total_errors}."); print(f"[Error] Failed to save decisions. Errors: {total_errors}")
        return save_successful

# --- 在 engine.py 文件中 ---
    # 确保文件顶部有这些 import:
    # import re # 可能需要引入 re 来解析标题级别
    # from pathlib import Path
    # from .utils import logger, create_decision_key
    # from .models import BlockType, FileRecordDTO, DECISION_KEEP, DECISION_DELETE, DECISION_UNDECIDED
    # from . import constants
    import re # <--- 添加这个 import

 # --- 在 engine.py 文件中 ---
    # 确保文件顶部有这些 import:
    # from pathlib import Path
    # from .utils import logger, create_decision_key
    # from .models import BlockType, FileRecordDTO, DECISION_KEEP, DECISION_DELETE, DECISION_UNDECIDED
    # from . import constants

    def apply_decisions(self) -> Dict[Path, str]:
        """
        Applies decisions stored in the engine's memory map (`self.block_decisions`),
        generating output content for blocks that are not marked DELETE.
        Attempts to restore basic Markdown formatting.
        """
        logger.info(f"Applying decisions to generate output content...")
        print(f"[*] Applying decisions...")
        output_content_map: Dict[Path, str] = {}
        processed_files_count = 0
        generated_files_count = 0
        error_files: List[str] = []
        all_files: List[FileRecordDTO] = []

        # 检查内存中的决策映射是否存在
        if not self.block_decisions:
             logger.warning("In-memory decision map (self.block_decisions) is empty. Output might include all blocks or be empty.")
             # 可以考虑如果决策映射为空是否应该返回空字典，取决于期望行为
             # return {}

        try:
            # 1. 获取所有已注册的文件记录
            all_files = self.storage.list_files()
            if not all_files:
                logger.warning("No files registered in storage.")
                return {}
            total_files = len(all_files)
            logger.info(f"Found {total_files} registered files.")

            # 2. 遍历每个文件
            for i, file_record in enumerate(all_files):
                original_path_str = file_record.original_path
                file_id = file_record.file_id
                original_path: Optional[Path] = None
                resolved_original_path: Optional[str] = None # 用于生成 key

                # 安全地处理和解析路径
                try:
                    if not original_path_str or not isinstance(original_path_str, str):
                        raise ValueError("Missing/invalid original_path in file record")
                    original_path = Path(original_path_str)
                    resolved_original_path = str(original_path.resolve()) # 解析一次路径
                except Exception as path_err:
                    logger.error(f"Invalid path '{original_path_str}' for file {file_id}: {path_err}. Skipping file.")
                    error_files.append(f"FileID:{file_id}(InvalidPath)")
                    continue

                logger.debug(f"Processing file {i+1}/{total_files}: {original_path.name} (ID: {file_id})")
                output_lines_for_file: List[str] = [] # 存储此文件最终输出的行

                try:
                    # 3. 获取该文件的所有块
                    blocks_in_file = self.storage.get_blocks_by_file(file_id)
                    # 注意：这里的顺序可能依赖于存储的实现。如果需要严格按原文顺序，
                    # 可能需要在处理文档时（_process_documents）记录并存储块的顺序信息。
                    # 目前假设 get_blocks_by_file 返回的顺序大致正确。

                    if not blocks_in_file:
                        logger.info(f"No blocks found for {original_path.name} in storage.")
                        processed_files_count += 1
                        continue

                    # 4. 遍历文件中的每个块
                    for block_dto in blocks_in_file:
                        decision = DECISION_UNDECIDED # 默认值

                        # 5. 获取该块的决策 (核心修改!)
                        try:
                            # 使用之前解析好的 resolved_original_path
                            key = create_decision_key(resolved_original_path, block_dto.block_id, block_dto.block_type.value)
                            # 从内存映射 self.block_decisions 获取决策
                            decision = self.block_decisions.get(key, DECISION_UNDECIDED)
                        except Exception as key_err:
                            logger.error(f"Error creating/getting decision key for block {block_dto.block_id}: {key_err}. Treating as UNDECIDED.")
                            decision = DECISION_UNDECIDED

                        # 6. 如果决策不是 DELETE，则处理并格式化
                        if decision != DECISION_DELETE:
                            text = block_dto.text
                            # 7. 尝试还原基本 Markdown 格式 (核心修改!)
                            if block_dto.block_type == BlockType.HEADING:
                                # 简单处理：假设所有标题都是一级标题
                                output_lines_for_file.append(f"# {text}")
                            elif block_dto.block_type == BlockType.CODE:
                                # 简单处理：使用通用代码块标记。
                                # 未来可优化：从元数据读取语言信息 (如果存储了的话)
                                code_lang = block_dto.metadata.get("code_language", "") # 尝试获取语言
                                output_lines_for_file.append(f"```{code_lang}\n{text}\n```")
                            elif block_dto.block_type == BlockType.LIST_ITEM:
                                # 简单处理：假设是无序列表项
                                output_lines_for_file.append(f"- {text}")
                            else: # 其他类型 (TEXT, TABLE, UNKNOWN 等) 直接添加文本
                                output_lines_for_file.append(text)

                    # 8. 如果此文件有内容需要输出
                    if output_lines_for_file:
                        # 确定输出路径 (保持原有逻辑)
                        output_sub_dir = self.output_dir_config_path
                        if self.input_dir and original_path.is_absolute() and self.input_dir.is_absolute():
                             try: relative_parent = original_path.parent.relative_to(self.input_dir); output_sub_dir = self.output_dir_config_path / relative_parent
                             except ValueError: logger.warning(f"Path {original_path} not relative to {self.input_dir}. Using default output dir.")
                             except Exception as rel_path_err: logger.error(f"Error calculating relative path for {original_path}: {rel_path_err}. Using default output dir.")
                        elif self.input_dir: logger.warning(f"Input dir or original path not absolute. Using default output dir.")

                        output_suffix = ".md"
                        if hasattr(constants, 'DEFAULT_OUTPUT_SUFFIX'): output_suffix = constants.DEFAULT_OUTPUT_SUFFIX + output_suffix
                        output_filename = original_path.stem + output_suffix
                        output_filepath = output_sub_dir / output_filename

                        # 使用双换行连接各块内容
                        output_content_map[output_filepath] = '\n\n'.join(output_lines_for_file)
                        logger.info(f"Generated content for {output_filepath} ({len(output_lines_for_file)} blocks kept)")
                        generated_files_count += 1
                    else:
                        logger.info(f"No content kept for {original_path.name}.")

                    processed_files_count += 1

                except FileOperationError as storage_e:
                    logger.error(f"Storage error processing blocks for {original_path.name}: {storage_e}")
                    error_files.append(original_path.name)
                    continue
                except Exception as file_proc_e:
                    logger.error(f"Failed processing blocks/generating output for {original_path.name}: {file_proc_e}", exc_info=True)
                    error_files.append(original_path.name)
                    continue

        # ... (方法末尾的异常处理和日志保持不变) ...
        except FileOperationError as storage_list_e:
            logger.error(f"Storage error listing files: {storage_list_e}")
            print(f"[Error] Storage list error: {storage_list_e}")
            return {}
        except Exception as outer_e:
            logger.error(f"Unexpected error applying decisions: {outer_e}", exc_info=True)
            print(f"[Error] Applying decisions: {outer_e}")
            return {}

        logger.info(f"Decision application complete. Processed {processed_files_count}/{len(all_files)} files. Generated content for {generated_files_count}.")
        print(f"\n[*] Decision application complete: Content generated for {generated_files_count} files.")
        if error_files:
            print(f"[Warning] Errors processing files: {', '.join(error_files)}")
        return output_content_map
        logger.info(f"Applying decisions to generate output content...")
        print(f"[*] Applying decisions...")
        output_content_map: Dict[Path, str] = {}
        processed_files_count = 0
        generated_files_count = 0
        error_files: List[str] = []
        all_files: List[FileRecordDTO] = []

        if not self.block_decisions:
             logger.warning("In-memory decision map (self.block_decisions) is empty. Output might include all blocks or be empty.")

        try:
            all_files = self.storage.list_files()
            if not all_files:
                logger.warning("No files registered in storage.")
                return {}
            total_files = len(all_files)
            logger.info(f"Found {total_files} registered files.")

            for i, file_record in enumerate(all_files):
                original_path_str = file_record.original_path
                file_id = file_record.file_id
                original_path: Optional[Path] = None
                resolved_original_path: Optional[str] = None

                try:
                    if not original_path_str or not isinstance(original_path_str, str):
                        raise ValueError("Missing/invalid original_path in file record")
                    original_path = Path(original_path_str)
                    resolved_original_path = str(original_path.resolve())
                except Exception as path_err:
                    logger.error(f"Invalid path '{original_path_str}' for file {file_id}: {path_err}. Skipping file.")
                    error_files.append(f"FileID:{file_id}(InvalidPath)")
                    continue

                logger.debug(f"Processing file {i+1}/{total_files}: {original_path.name} (ID: {file_id})")
                output_lines_for_file: List[str] = []

                try:
                    blocks_in_file = self.storage.get_blocks_by_file(file_id)
                    if not blocks_in_file:
                        logger.info(f"No blocks found for {original_path.name} in storage.")
                        processed_files_count += 1
                        continue

                    for block_dto in blocks_in_file:
                        decision = DECISION_UNDECIDED
                        try:
                            key = create_decision_key(resolved_original_path, block_dto.block_id, block_dto.block_type.value)
                            decision = self.block_decisions.get(key, DECISION_UNDECIDED)
                        except Exception as key_err:
                            logger.error(f"Error creating/getting decision key for block {block_dto.block_id}: {key_err}. Treating as UNDECIDED.")
                            decision = DECISION_UNDECIDED

                        if decision != DECISION_DELETE:
                            text = block_dto.text
                            # --- 增强格式化 ---
                    # ... (在 apply_decisions 方法内部) ...
                        elif block_dto.block_type == BlockType.HEADING:
                            logger.debug(f"--- Processing HEADING block {block_dto.block_id} ---") # 日志块开始
                            level = 1 # 默认级别
                            original_elem_type_str = block_dto.metadata.get('block_type', 'Heading') # 获取原始类型字符串
                            category_depth = block_dto.metadata.get('element_metadata', {}).get('category_depth')
                            logger.debug(f"  Initial level: {level}")
                            logger.debug(f"  Original element type string: '{original_elem_type_str}' (type: {type(original_elem_type_str)})")
                            logger.debug(f"  Category depth: {category_depth} (type: {type(category_depth)})")

                            if isinstance(original_elem_type_str, str):
                                logger.debug("  original_elem_type_str is a string. Checking content...")
                                if "Title" in original_elem_type_str:
                                    level = 1
                                    logger.debug(f"  'Title' found in original type string. Set level to: {level}")
                                else:
                                    logger.debug("  'Title' NOT found. Checking for Header-N pattern...")
                                    # 尝试从 "Header-2", "Heading-3" 等提取数字
                                    match = re.search(r'[-_](\d+)$', original_elem_type_str)
                                    if match:
                                        logger.debug(f"  Found pattern like Header-N. Match group 1: {match.group(1)}")
                                        try:
                                            level = int(match.group(1))
                                            logger.debug(f"  Set level from regex: {level}")
                                        except ValueError:
                                            level = 2 # 提取失败，默认二级
                                            logger.debug(f"  Regex number conversion failed. Set level to default: {level}")
                                    # 如果没有数字，但有 category_depth，可以尝试使用它
                                    elif category_depth is not None:
                                        logger.debug(f"  Header-N pattern NOT found, but category_depth exists: {category_depth}")
                                        try:
                                            # category_depth 通常从0开始，对应 H1 可能是 1 或 0？需测试
                                            # 假设 depth 0/1 是 H1, 2 是 H2 ...
                                            level = max(1, int(category_depth)) # 至少一级
                                            logger.debug(f"  Set level from category_depth ({category_depth}): {level}")
                                        except ValueError:
                                            level = 2 # 转换失败，默认二级
                                            logger.debug(f"  Category depth conversion failed. Set level to default: {level}")
                                    else:
                                        level = 2 # 其他情况默认二级标题
                                        logger.debug(f"  No Title, no Header-N, no category_depth. Set level to default: {level}")
                            else:
                                logger.debug("  original_elem_type_str is NOT a string.")

                            level = max(1, min(6, level)) # 限制在 1-6 级
                            prefix = "#" * level
                            logger.debug(f"  Final level: {level}, Prefix: '{prefix}'") # 最终级别
                            output_lines_for_file.append(f"{prefix} {text}")
                            logger.debug(f"--- Finished processing HEADING block {block_dto.block_id} ---") # 日志块结束
                        elif block_dto.block_type == BlockType.CODE:
                            code_lang = block_dto.metadata.get("code_language", "") # 尝试获取语言（可能仍然没有）
                            output_lines_for_file.append(f"```{code_lang}\n{text}\n```")
                        elif block_dto.block_type == BlockType.LIST_ITEM:
                                # 仍然是简单的无序列表处理
                            output_lines_for_file.append(f"- {text}")
                        else: # Default for TEXT, TABLE, UNKNOWN etc.
                            output_lines_for_file.append(text)
                            # --- 结束增强格式化 ---

                    # ... (后续确定输出路径和保存文件的逻辑不变) ...
                    if output_lines_for_file:
                        output_sub_dir = self.output_dir_config_path
                        if self.input_dir and original_path.is_absolute() and self.input_dir.is_absolute():
                             try: relative_parent = original_path.parent.relative_to(self.input_dir); output_sub_dir = self.output_dir_config_path / relative_parent
                             except ValueError: logger.warning(f"Path {original_path} not relative to {self.input_dir}. Using default output dir.")
                             except Exception as rel_path_err: logger.error(f"Error calculating relative path for {original_path}: {rel_path_err}. Using default output dir.")
                        elif self.input_dir: logger.warning(f"Input dir or original path not absolute. Using default output dir.")

                        output_suffix = ".md"
                        if hasattr(constants, 'DEFAULT_OUTPUT_SUFFIX'): output_suffix = constants.DEFAULT_OUTPUT_SUFFIX + output_suffix
                        output_filename = original_path.stem + output_suffix
                        output_filepath = output_sub_dir / output_filename

                        output_content_map[output_filepath] = '\n\n'.join(output_lines_for_file)
                        logger.info(f"Generated content for {output_filepath} ({len(output_lines_for_file)} blocks kept)")
                        generated_files_count += 1
                    else:
                        logger.info(f"No content kept for {original_path.name}.")

                    processed_files_count += 1

                except FileOperationError as storage_e:
                    # ... (处理存储错误) ...
                    continue
                except Exception as file_proc_e:
                    # ... (处理文件处理错误) ...
                    continue

        # ... (方法末尾的异常处理和日志保持不变) ...
        except FileOperationError as storage_list_e:
            logger.error(f"Storage error listing files: {storage_list_e}")
            print(f"[Error] Storage list error: {storage_list_e}")
            return {}
        except Exception as outer_e:
            logger.error(f"Unexpected error applying decisions: {outer_e}", exc_info=True)
            print(f"[Error] Applying decisions: {outer_e}")
            return {}

        logger.info(f"Decision application complete. Processed {processed_files_count}/{len(all_files)} files. Generated content for {generated_files_count}.")
        print(f"\n[*] Decision application complete: Content generated for {generated_files_count} files.")
        if error_files:
            print(f"[Warning] Errors processing files: {', '.join(error_files)}")
        return output_content_map
        
        logger.info(f"Applying decisions to generate output content...")
        print(f"[*] Applying decisions...")
        output_content_map: Dict[Path, str] = {}
        processed_files_count = 0
        generated_files_count = 0
        error_files: List[str] = []
        all_files: List[FileRecordDTO] = []

        # 检查内存中的决策映射是否存在
        if not self.block_decisions:
             logger.warning("In-memory decision map (self.block_decisions) is empty. Output might include all blocks or be empty.")
             # 可以考虑如果决策映射为空是否应该返回空字典，取决于期望行为
             # return {}

        try:
            # 1. 获取所有已注册的文件记录
            all_files = self.storage.list_files()
            if not all_files:
                logger.warning("No files registered in storage.")
                return {}
            total_files = len(all_files)
            logger.info(f"Found {total_files} registered files.")

            # 2. 遍历每个文件
            for i, file_record in enumerate(all_files):
                original_path_str = file_record.original_path
                file_id = file_record.file_id
                original_path: Optional[Path] = None
                resolved_original_path: Optional[str] = None # 用于生成 key

                # 安全地处理和解析路径
                try:
                    if not original_path_str or not isinstance(original_path_str, str):
                        raise ValueError("Missing/invalid original_path in file record")
                    original_path = Path(original_path_str)
                    resolved_original_path = str(original_path.resolve()) # 解析一次路径
                except Exception as path_err:
                    logger.error(f"Invalid path '{original_path_str}' for file {file_id}: {path_err}. Skipping file.")
                    error_files.append(f"FileID:{file_id}(InvalidPath)")
                    continue

                logger.debug(f"Processing file {i+1}/{total_files}: {original_path.name} (ID: {file_id})")
                output_lines_for_file: List[str] = [] # 存储此文件最终输出的行

                try:
                    # 3. 获取该文件的所有块
                    blocks_in_file = self.storage.get_blocks_by_file(file_id)
                    # 注意：这里的顺序可能依赖于存储的实现。如果需要严格按原文顺序，
                    # 可能需要在处理文档时（_process_documents）记录并存储块的顺序信息。
                    # 目前假设 get_blocks_by_file 返回的顺序大致正确。

                    if not blocks_in_file:
                        logger.info(f"No blocks found for {original_path.name} in storage.")
                        processed_files_count += 1
                        continue

                    # 4. 遍历文件中的每个块
                    for block_dto in blocks_in_file:
                        decision = DECISION_UNDECIDED # 默认值

                        # 5. 获取该块的决策 (核心修改!)
                        try:
                            # 使用之前解析好的 resolved_original_path
                            key = create_decision_key(resolved_original_path, block_dto.block_id, block_dto.block_type.value)
                            # 从内存映射 self.block_decisions 获取决策
                            decision = self.block_decisions.get(key, DECISION_UNDECIDED)
                        except Exception as key_err:
                            logger.error(f"Error creating/getting decision key for block {block_dto.block_id}: {key_err}. Treating as UNDECIDED.")
                            decision = DECISION_UNDECIDED

                        # 6. 如果决策不是 DELETE，则处理并格式化
                        if decision != DECISION_DELETE:
                            text = block_dto.text
                            # 7. 尝试还原基本 Markdown 格式 (核心修改!)
                            if block_dto.block_type == BlockType.HEADING:
                                # 简单处理：假设所有标题都是一级标题
                                output_lines_for_file.append(f"# {text}")
                            elif block_dto.block_type == BlockType.CODE:
                                # 简单处理：使用通用代码块标记。
                                # 未来可优化：从元数据读取语言信息 (如果存储了的话)
                                code_lang = block_dto.metadata.get("code_language", "") # 尝试获取语言
                                output_lines_for_file.append(f"```{code_lang}\n{text}\n```")
                            elif block_dto.block_type == BlockType.LIST_ITEM:
                                # 简单处理：假设是无序列表项
                                output_lines_for_file.append(f"- {text}")
                            else: # 其他类型 (TEXT, TABLE, UNKNOWN 等) 直接添加文本
                                output_lines_for_file.append(text)

                    # 8. 如果此文件有内容需要输出
                    if output_lines_for_file:
                        # 确定输出路径 (保持原有逻辑)
                        output_sub_dir = self.output_dir_config_path
                        if self.input_dir and original_path.is_absolute() and self.input_dir.is_absolute():
                             try: relative_parent = original_path.parent.relative_to(self.input_dir); output_sub_dir = self.output_dir_config_path / relative_parent
                             except ValueError: logger.warning(f"Path {original_path} not relative to {self.input_dir}. Using default output dir.")
                             except Exception as rel_path_err: logger.error(f"Error calculating relative path for {original_path}: {rel_path_err}. Using default output dir.")
                        elif self.input_dir: logger.warning(f"Input dir or original path not absolute. Using default output dir.")

                        output_suffix = ".md"
                        if hasattr(constants, 'DEFAULT_OUTPUT_SUFFIX'): output_suffix = constants.DEFAULT_OUTPUT_SUFFIX + output_suffix
                        output_filename = original_path.stem + output_suffix
                        output_filepath = output_sub_dir / output_filename

                        # 使用双换行连接各块内容
                        output_content_map[output_filepath] = '\n\n'.join(output_lines_for_file)
                        logger.info(f"Generated content for {output_filepath} ({len(output_lines_for_file)} blocks kept)")
                        generated_files_count += 1
                    else:
                        logger.info(f"No content kept for {original_path.name}.")

                    processed_files_count += 1

                except FileOperationError as storage_e:
                    logger.error(f"Storage error processing blocks for {original_path.name}: {storage_e}")
                    error_files.append(original_path.name)
                    continue
                except Exception as file_proc_e:
                    logger.error(f"Failed processing blocks/generating output for {original_path.name}: {file_proc_e}", exc_info=True)
                    error_files.append(original_path.name)
                    continue

        # ... (方法末尾的异常处理和日志保持不变) ...
        except FileOperationError as storage_list_e:
            logger.error(f"Storage error listing files: {storage_list_e}")
            print(f"[Error] Storage list error: {storage_list_e}")
            return {}
        except Exception as outer_e:
            logger.error(f"Unexpected error applying decisions: {outer_e}", exc_info=True)
            print(f"[Error] Applying decisions: {outer_e}")
            return {}

        logger.info(f"Decision application complete. Processed {processed_files_count}/{len(all_files)} files. Generated content for {generated_files_count}.")
        print(f"\n[*] Decision application complete: Content generated for {generated_files_count} files.")
        if error_files:
            print(f"[Warning] Errors processing files: {', '.join(error_files)}")
        return output_content_map

    # --- Public Interface Methods for UI ---

    def get_md5_duplicates(self) -> List[List[ContentBlockDTO]]:
        if not self._analysis_completed: logger.warning("Requesting MD5 duplicates, but analysis not completed."); return []
        return self.md5_duplicates

    def get_semantic_duplicates(self) -> List[Tuple[ContentBlockDTO, ContentBlockDTO, float]]:
        if not self._analysis_completed: logger.warning("Requesting semantic duplicates, but analysis not completed."); return []
        if self.skip_semantic: logger.info("Semantic analysis was skipped."); return []
        return self.semantic_duplicates

    def update_decision(self, block_key: str, decision: str) -> bool:
        """Updates decision for a block, persists via storage metadata."""
        if decision not in [DECISION_KEEP, DECISION_DELETE, DECISION_UNDECIDED]: logger.error(f"Invalid decision: '{decision}'"); return False
        if not block_key: logger.error("Invalid block key."); return False
        logger.debug(f"Updating decision for key '{block_key}' to '{decision}'...")
        try:
            _, block_id, _ = parse_decision_key(block_key)
            if not block_id: logger.error(f"Could not parse block_id from key '{block_key}'."); return False
            block = self.storage.get_block(block_id)
            if not block: logger.error(f"Block ID '{block_id}' not found."); self.block_decisions.pop(block_key, None); return False
            current_decision = block.metadata.get(METADATA_DECISION_KEY, DECISION_UNDECIDED)
            if current_decision == decision: logger.debug(f"Decision already '{decision}'.");  return True
            block.metadata = block.metadata.copy() # <<< 添加在这里
            if not isinstance(block.metadata, dict): block.metadata = {}
            block.metadata[METADATA_DECISION_KEY] = decision
            if not block.file_id: logger.error(f"Block {block_id} missing file_id."); return False
            logger.debug(f"Metadata before saving block {block.block_id} in update_decision: {block.metadata}") # 添加这行日志
            self.storage.save_blocks(file_id=block.file_id, blocks=[block]) # 这是原来的行            self.block_decisions[block_key] = decision
            self.block_decisions[block_key] = decision
            logger.info(f"Updated decision for block '{block_id}' to '{decision}'."); print(f"[*] Decision updated for {block_id}.")
            return True
        except FileOperationError as e: handle_error(e, f"updating decision {block_key}"); print(f"[Error] Storage error: {e}"); return False
        except ValueError as e: logger.error(f"Error parsing key '{block_key}': {e}"); return False
        except Exception as e: handle_error(e, f"updating decision {block_key}"); print(f"[Error] Unexpected error: {e}"); return False

    def get_status_summary(self) -> Dict[str, Any]:
        """Provides a summary of the current engine state."""
        logger.debug("Generating status summary...")
        md5_count = len(self.md5_duplicates); semantic_count = len(self.semantic_duplicates) if not self.skip_semantic else 0
        total_blocks_in_mem = len(self.blocks_data)
        decided_count_in_mem = sum(1 for d in self.block_decisions.values() if d in [DECISION_KEEP, DECISION_DELETE])
        undecided_count_in_mem = len(self.block_decisions) - decided_count_in_mem
        summary = {
            "input_dir": str(self.input_dir.resolve()) if self.input_dir else "Not set",
            # *** FIXED: Use correct attribute name ***
            "decision_file_config": str(self.decision_file_config_path),
            "output_dir_config": str(self.output_dir_config_path),
            "storage_implementation": self.storage.__class__.__name__,
            "skip_semantic": self.skip_semantic, "similarity_threshold": self.similarity_threshold,
            "analysis_completed": self._analysis_completed, "decisions_loaded_in_memory": self._decisions_loaded,
            "total_blocks_processed_last_run": total_blocks_in_mem,
            "md5_duplicates_groups_last_run": md5_count, "semantic_duplicates_pairs_last_run": semantic_count,
            "decided_blocks_in_memory_map": decided_count_in_mem,
            "undecided_blocks_in_memory_map": undecided_count_in_mem
        }
        logger.debug(f"Status summary: {summary}")
        return summary

    def set_similarity_threshold(self, threshold: float) -> bool:
        """Sets the semantic similarity threshold and resets analysis status."""
        if 0.0 <= threshold <= 1.0:
            logger.info(f"Setting similarity threshold to {threshold}")
            self.similarity_threshold = threshold
            if hasattr(self.semantic_analyzer, 'similarity_threshold'): self.semantic_analyzer.similarity_threshold = threshold
            self._analysis_completed = False; print(f"[*] Similarity threshold set to {threshold}. Re-analysis required.")
            return True
        else: logger.error(f"Invalid threshold: {threshold}. Must be 0.0-1.0."); print(f"[Error] Invalid threshold: {threshold}."); return False

    def set_skip_semantic(self, skip: bool) -> None:
        """Sets the flag to skip semantic analysis and resets analysis status."""
        logger.info(f"Setting skip_semantic to {skip}")
        self.skip_semantic = skip; self._analysis_completed = False
        status = "enabled" if skip else "disabled"; print(f"[*] Skipping semantic analysis {status}. Re-analysis required.")

# --- End of File ---
