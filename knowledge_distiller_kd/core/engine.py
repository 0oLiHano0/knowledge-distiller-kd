# knowledge_distiller_kd/core/engine.py
"""
Core engine for the Knowledge Distiller tool.
Encapsulates business logic, state management, and process orchestration.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set, DefaultDict
from collections import defaultdict

# 导入项目内部模块 (使用相对路径)
from . import constants
from .error_handler import KDError, ConfigurationError, handle_error, validate_file_path, FileOperationError, AnalysisError
from .utils import logger, create_decision_key, parse_decision_key
from ..storage.file_storage import FileStorage
from ..analysis.md5_analyzer import MD5Analyzer
from ..analysis.semantic_analyzer import SemanticAnalyzer
from ..processing.document_processor import ContentBlock, process_directory, DocumentProcessingError
from ..processing.block_merger import merge_code_blocks


class KnowledgeDistillerEngine:
    """
    The core engine responsible for analysis, decision management, and state.
    """

    def __init__(
        self,
        storage: FileStorage,
        input_dir: Optional[Union[str, Path]] = None,
        decision_file: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        skip_semantic: bool = False,
        similarity_threshold: float = constants.DEFAULT_SIMILARITY_THRESHOLD
    ):
        logger.info("Initializing KnowledgeDistillerEngine...")
        self.storage = storage
        self.input_dir: Optional[Path] = None
        if input_dir:
            try:
                validated_input_dir = validate_file_path(Path(input_dir), must_exist=True)
                if not validated_input_dir.is_dir(): raise ConfigurationError(f"Input path is not a directory: {validated_input_dir}")
                self.input_dir = validated_input_dir
            except (ConfigurationError, FileOperationError) as e:
                 logger.error(f"Invalid input directory provided during initialization: {e}")
                 raise ConfigurationError(f"Engine initialization failed due to invalid input directory: {e}") from e
        self.decision_file: Path = Path(decision_file or constants.DEFAULT_DECISION_FILE).resolve()
        self.output_dir: Path = Path(output_dir or constants.DEFAULT_OUTPUT_DIR).resolve()
        self.skip_semantic: bool = skip_semantic
        self.similarity_threshold: float = max(0.0, min(1.0, similarity_threshold))
        self.blocks_data: List[ContentBlock] = []
        self.block_decisions: Dict[str, str] = {}
        self.md5_duplicates: List[List[ContentBlock]] = []
        self.semantic_duplicates: List[Tuple[ContentBlock, ContentBlock, float]] = []
        self._decisions_loaded: bool = False
        self._analysis_completed: bool = False
        try:
            self.md5_analyzer = MD5Analyzer()
            self.semantic_analyzer = SemanticAnalyzer(similarity_threshold=self.similarity_threshold)
        except Exception as e:
            logger.critical(f"Failed to initialize analyzers: {e}", exc_info=True)
            raise ConfigurationError(f"Engine initialization failed due to analyzer error: {e}") from e
        logger.info("KnowledgeDistillerEngine initialized successfully.")

    def _reset_state(self) -> None:
        logger.debug("Resetting engine state...")
        self.blocks_data.clear(); self.block_decisions.clear(); self.md5_duplicates.clear(); self.semantic_duplicates.clear()
        self._decisions_loaded = False; self._analysis_completed = False
        logger.debug("Engine state reset.")

    def set_input_dir(self, input_dir: Union[str, Path]) -> bool:
        logger.info(f"Attempting to set input directory to: {input_dir}")
        try:
            input_path = Path(input_dir); resolved_path = validate_file_path(input_path, must_exist=True)
            if not resolved_path.is_dir(): logger.error(f"Setting input directory failed: '{resolved_path}' is not a directory."); print(f"[Error] Path '{resolved_path}' is not a valid directory."); return False
            logger.info(f"Input directory set to: {resolved_path}"); self.input_dir = resolved_path; self._reset_state(); print(f"[*] Input directory set to: {self.input_dir}"); return True
        except (FileOperationError, ConfigurationError) as e: handle_error(e, "setting input directory"); print(f"[Error] Error setting input directory: {e}"); return False
        except Exception as e: handle_error(e, "setting input directory"); print(f"[Error] Unexpected error setting input directory: {e}"); return False

    def run_analysis(self) -> bool:
        if not self.input_dir: logger.error("Analysis aborted: Input directory not set."); print("[Error] Input directory not set."); return False
        logger.info(f"--- Starting analysis for folder: {self.input_dir} ---"); print(f"\n[*] Starting analysis for folder: {self.input_dir}")
        self._reset_state(); self._analysis_completed = False; analysis_successful = True
        try:
            print("\n[*] Step: Processing documents...");
            if not self._process_documents(): raise AnalysisError("Document processing failed.")
            print("[*] Step complete: Processing documents")
            print("\n[*] Step: Merging code blocks...")
            if not self._merge_code_blocks_step(): logger.warning("Code block merging step failed, continuing analysis..."); print("[Warning] Code block merging failed, analysis will continue.")
            else: print("[*] Step complete: Merging code blocks")
            print("\n[*] Step: Initializing decisions...")
            if not self._initialize_decisions(): raise AnalysisError("Decision initialization failed.")
            print("[*] Step complete: Initializing decisions")
            print("\n[*] Step: MD5 Deduplication...")
            md5_duplicates_found, suggested_md5_decisions = self.md5_analyzer.find_md5_duplicates(self.blocks_data, self.block_decisions)
            self.md5_duplicates = md5_duplicates_found; self._update_decisions_from_md5(suggested_md5_decisions)
            print(f"[*] Step complete: MD5 Deduplication ({len(self.md5_duplicates)} groups found)")
            if not self.skip_semantic:
                print("\n[*] Step: Loading semantic model...")
                model_loaded = self.semantic_analyzer.load_semantic_model()
                if not model_loaded or not self._model_loaded_successfully(): logger.warning("Semantic model failed to load or unavailable. Skipping semantic analysis."); print("[Warning] Semantic model failed to load or unavailable. Skipping semantic analysis."); self.skip_semantic = True
                else:
                    print("[*] Step complete: Loading semantic model")
                    if self._model_loaded_successfully():
                        print("\n[*] Step: Semantic Deduplication...")
                        blocks_for_semantic = self._filter_blocks_for_semantic()
                        self.semantic_duplicates = self.semantic_analyzer.find_semantic_duplicates(blocks_for_semantic)
                        print(f"[*] Step complete: Semantic Deduplication ({len(self.semantic_duplicates)} pairs found)")
                    else: logger.info("Skipping semantic deduplication step as model is not loaded."); print("[*] Skipping step: Semantic Deduplication (model not loaded).")
            else: logger.info("Skipping semantic analysis steps as configured."); print("[*] Skipping steps: Loading semantic model, Semantic Deduplication.")
            self._analysis_completed = True; logger.info("Analysis process completed successfully."); print("\n[*] Analysis workflow completed.")
        except AnalysisError as ae: logger.error(f"Analysis process failed: {ae}", exc_info=False); print(f"\n[Error] Analysis failed: {ae}"); analysis_successful = False; self._analysis_completed = False
        except Exception as e: handle_error(e, "running analysis workflow"); print(f"\n[Error] An unexpected error occurred during analysis: {e}"); analysis_successful = False; self._analysis_completed = False
        return analysis_successful

    def _model_loaded_successfully(self) -> bool:
        is_loaded = getattr(self.semantic_analyzer, '_model_loaded', False)
        model_exists = getattr(self.semantic_analyzer, 'model', None) is not None
        return not self.skip_semantic and is_loaded and model_exists

    def _filter_blocks_for_semantic(self) -> List[ContentBlock]:
        logger.debug("Filtering blocks for semantic analysis...")
        blocks_to_analyze: List[ContentBlock] = []; skipped_titles = 0; skipped_deleted = 0
        for block in self.blocks_data:
            if block.block_type == constants.BLOCK_TYPE_TITLE: skipped_titles += 1; continue
            try:
                 key = create_decision_key(str(Path(block.file_path).resolve()), block.block_id, block.block_type)
                 if self.block_decisions.get(key) == constants.DECISION_DELETE: skipped_deleted += 1; continue
            except Exception as e: logger.warning(f"Error creating key for block {block.block_id} during semantic filtering: {e}. Skipping block."); continue
            blocks_to_analyze.append(block)
        logger.info(f"Semantic filtering complete. Kept {len(blocks_to_analyze)} blocks. Skipped titles: {skipped_titles}, skipped MD5 deleted: {skipped_deleted}.")
        return blocks_to_analyze

    def _process_documents(self) -> bool:
        if not self.input_dir: return False
        logger.info(f"Processing documents in directory: {self.input_dir}")
        try:
            results = process_directory(self.input_dir, recursive=True)
            self.blocks_data.clear(); total_blocks = 0
            for file_path, blocks in results.items(): self.blocks_data.extend(blocks); total_blocks += len(blocks)
            if not results: logger.warning(f"No processable markdown files found or no content extracted in {self.input_dir}.")
            logger.info(f"Successfully processed {len(results)} files, extracted {total_blocks} blocks.")
            print(f"[*] Document processing: {len(results)} files processed, {total_blocks} blocks extracted.")
            return True
        except DocumentProcessingError as e: handle_error(e, "processing documents"); print(f"[Error] Error processing documents: {e}"); return False
        except Exception as e: handle_error(e, "unexpected error during document processing"); print(f"[Error] Unexpected error during document processing: {e}"); return False

    def _merge_code_blocks_step(self) -> bool:
        if not self.blocks_data: logger.info("No blocks to process for code merging."); return True
        logger.info(f"Starting code block merging for {len(self.blocks_data)} blocks...")
        try:
            original_count = len(self.blocks_data)
            self.blocks_data = merge_code_blocks(self.blocks_data)
            new_count = len(self.blocks_data)
            logger.info(f"Code block merging finished. Block count changed from {original_count} to {new_count}.")
            print(f"[*] Code block merging: Count changed from {original_count} to {new_count}.")
            return True
        except Exception as e: handle_error(e, "merging code blocks"); print(f"[Error] Unexpected error during code block merging: {e}"); return False

    def _initialize_decisions(self) -> bool:
        if not self.blocks_data: logger.info("No blocks available, skipping decision initialization."); self.block_decisions.clear(); return True
        logger.info(f"Initializing decisions for {len(self.blocks_data)} blocks...")
        self.block_decisions.clear(); initialized_count = 0; error_count = 0
        for block in self.blocks_data:
             try:
                 key = create_decision_key(str(Path(block.file_path).resolve()), block.block_id, block.block_type)
                 self.block_decisions[key] = constants.DECISION_UNDECIDED
                 initialized_count += 1
             except Exception as e: error_count += 1; logger.error(f"Failed to create decision key for block {block.block_id} in {block.file_path}: {e}", exc_info=False)
        logger.info(f"Initialized {initialized_count} decisions. Failed for {error_count} blocks.")
        print(f"[*] Decision initialization: {initialized_count} initialized, {error_count} failed.")
        return True

    def _update_decisions_from_md5(self, suggested_decisions: Dict[str, str]):
        if not suggested_decisions: logger.info("No MD5-based decision suggestions to apply."); return
        logger.info(f"Applying {len(suggested_decisions)} MD5-based decision suggestions...")
        updated_count = 0
        for key, suggested_decision in suggested_decisions.items(): self.block_decisions[key] = suggested_decision; updated_count += 1
        logger.info(f"Applied {updated_count} MD5 suggestions.")

    def load_decisions(self) -> bool:
        logger.info(f"Attempting to load decisions from: {self.decision_file}")
        try:
            decisions_from_file = self.storage.load_decisions(self.decision_file)
            if not decisions_from_file: logger.warning(f"No decision records loaded from {self.decision_file}."); self.block_decisions.clear(); self._decisions_loaded = False; return False
            self.block_decisions.clear(); logger.info(f"Successfully read {len(decisions_from_file)} decision records, applying..."); loaded_count = 0; error_count = 0
            for record in decisions_from_file:
                if not isinstance(record, dict): logger.warning(f"Skipping malformed decision record: {record}"); error_count += 1; continue
                file_str = record.get('file'); block_id = record.get('block_id'); block_type = record.get('type'); decision = record.get('decision')
                if not all([file_str, block_id, block_type, decision]): logger.warning(f"Skipping incomplete decision record: {record}"); error_count += 1; continue
                if decision not in [constants.DECISION_KEEP, constants.DECISION_DELETE, constants.DECISION_UNDECIDED]: logger.warning(f"Skipping record with invalid decision value ('{decision}'): {record}"); error_count += 1; continue
                try:
                    path_in_record = Path(file_str); final_path_str = file_str
                    if not path_in_record.is_absolute() and self.input_dir:
                        abs_path = (self.input_dir / path_in_record).resolve(); final_path_str = str(abs_path); logger.debug(f"Resolved relative path '{file_str}' to '{final_path_str}' using input_dir.")
                    elif path_in_record.is_absolute(): final_path_str = str(path_in_record.resolve())
                    elif not path_in_record.is_absolute() and not self.input_dir: logger.warning(f"Decision record uses relative path '{file_str}' but input_dir is not set. Using relative path as key.")
                    key = create_decision_key(final_path_str, str(block_id), str(block_type))
                    self.block_decisions[key] = decision; loaded_count += 1
                except Exception as e: logger.error(f"Error processing decision record {record}: {e}"); error_count += 1; continue
            logger.info(f"Decision loading complete: Successfully applied {loaded_count} decisions, {error_count} records failed.")
            print(f"[*] Decision loading complete: Applied {loaded_count}, Failed {error_count}.")
            self._decisions_loaded = loaded_count > 0; return self._decisions_loaded
        except FileOperationError as e: handle_error(e, f"loading decision file {self.decision_file}"); print(f"[Error] File error while loading decisions: {e}"); self._decisions_loaded = False; return False
        except Exception as e: handle_error(e, f"loading decision file {self.decision_file}"); print(f"[Error] Unexpected error while loading decisions: {e}"); self._decisions_loaded = False; return False

    def save_decisions(self) -> bool:
        if not self.block_decisions: logger.warning("No decisions available to save."); print("[!] No decisions to save."); return False
        logger.info(f"Preparing to save {len(self.block_decisions)} decisions to: {self.decision_file}")
        decisions_to_save = []; saved_count = 0; error_count = 0
        for key, decision in self.block_decisions.items():
            try:
                file_path_str, block_id, block_type = parse_decision_key(key)
                if file_path_str is None: logger.warning(f"Could not parse decision key '{key}', skipping save for this entry."); error_count += 1; continue
                path_to_save_str = file_path_str
                if self.input_dir:
                    try:
                        abs_path = Path(file_path_str)
                        if abs_path.is_absolute() and abs_path.is_relative_to(self.input_dir):
                             relative_path = abs_path.relative_to(self.input_dir); path_to_save_str = str(relative_path); logger.debug(f"Calculated relative path for saving: '{path_to_save_str}' from '{abs_path}'")
                        elif not abs_path.is_absolute(): logger.debug(f"Path '{file_path_str}' is already relative, saving as is."); path_to_save_str = file_path_str
                        else: logger.warning(f"Path '{file_path_str}' is absolute but not under input_dir '{self.input_dir}'. Saving absolute path.")
                    except ValueError: logger.warning(f"Cannot determine if '{file_path_str}' is relative to '{self.input_dir}'. Saving original/absolute path.")
                    except Exception as path_e: logger.error(f"Error processing path '{file_path_str}' for relative saving: {path_e}. Saving original/absolute path.")
                record = {'file': path_to_save_str, 'block_id': str(block_id), 'type': block_type, 'decision': decision}
                decisions_to_save.append(record); saved_count += 1
            except Exception as e: logger.error(f"Error processing decision key '{key}' for saving: {e}"); error_count += 1; continue
        if not decisions_to_save: logger.error("No valid decision records generated for saving."); print("[Error] No valid decisions could be prepared for saving."); return False
        try:
            success = self.storage.save_decisions(self.decision_file, decisions_to_save)
            if success: logger.info(f"Successfully requested save of {saved_count} decisions to {self.decision_file}. {error_count} entries failed processing."); print(f"[*] Successfully saved {saved_count} decisions. {error_count} entries had processing errors."); return True
            else: logger.error(f"Storage layer indicated failure saving decisions to {self.decision_file}."); print(f"[Error] Failed to save decisions to {self.decision_file.name}. Check logs for details."); return False
        except FileOperationError as e: handle_error(e, f"saving decision file {self.decision_file}"); print(f"[Error] File error while saving decisions: {e}"); return False
        except Exception as e: handle_error(e, f"saving decision file {self.decision_file}"); print(f"[Error] Unexpected error while saving decisions: {e}"); return False

    # --- Output Generation (Moved from KDToolCLI) ---
    def apply_decisions(self) -> bool:
        """
        Applies the decisions to generate deduplicated output files.
        """
        if not self._analysis_completed: logger.warning("Cannot apply decisions: Analysis has not been completed successfully."); print("[Warning] Please run analysis successfully before applying decisions."); return False
        if not self.blocks_data: logger.warning("No content blocks available to process for applying decisions."); print("[Warning] No content blocks found."); return True
        logger.info(f"Applying decisions to generate output files in: {self.output_dir}")
        print(f"[*] Applying decisions to generate output files...")
        files_to_process: DefaultDict[Path, List[ContentBlock]] = defaultdict(list)
        for block in self.blocks_data:
            try: abs_path = Path(block.file_path).resolve(); files_to_process[abs_path].append(block)
            except Exception as e: logger.error(f"Error resolving path for block {block.block_id} from {block.file_path}: {e}. Skipping block."); continue
        if not files_to_process: logger.warning("No files found to process after grouping blocks."); return True

        processed_files_count = 0; generated_files_count = 0; error_files = []; total_files = len(files_to_process)
        overall_success = True # Track if any file processing fails

        for i, (abs_file_path, blocks_in_file) in enumerate(files_to_process.items()):
            original_path = abs_file_path; logger.debug(f"Processing file {i+1}/{total_files}: {original_path.name}")
            output_sub_dir = self.output_dir; file_proc_success = True # Assume success for this file initially
            try:
                # Determine and create output subdirectory if needed
                if self.input_dir:
                    try:
                        relative_part = original_path.parent.relative_to(self.input_dir); output_sub_dir = self.output_dir / relative_part
                        output_sub_dir.mkdir(parents=True, exist_ok=True)
                    except ValueError: logger.warning(f"Cannot make path {original_path.parent} relative to {self.input_dir}. Outputting to base output dir.")
                    except OSError as e: logger.error(f"Cannot create output subdirectory {output_sub_dir}: {e}. Skipping file."); error_files.append(original_path.name); overall_success = False; continue # Skip this file
                    except Exception as e_mkdir: logger.error(f"Unexpected error creating output subdirectory {output_sub_dir}: {e_mkdir}"); error_files.append(original_path.name); overall_success = False; continue # Skip this file
                else:
                    # Ensure base output directory exists if input_dir is not set
                     self.output_dir.mkdir(parents=True, exist_ok=True)

                output_filename = original_path.stem + constants.DEFAULT_OUTPUT_SUFFIX + original_path.suffix
                output_filepath = output_sub_dir / output_filename
                kept_blocks_content: List[str] = []

                blocks_in_file.sort(key=lambda b: b.block_id) # Sort for consistent output
                for block in blocks_in_file:
                    decision = constants.DECISION_UNDECIDED
                    try:
                        key_abs = create_decision_key(str(abs_file_path), block.block_id, block.block_type)
                        decision = self.block_decisions.get(key_abs, constants.DECISION_UNDECIDED)
                        if decision == constants.DECISION_UNDECIDED and self.input_dir:
                             try: relative_path_str = str(abs_file_path.relative_to(self.input_dir)); key_rel = create_decision_key(relative_path_str, block.block_id, block.block_type); decision = self.block_decisions.get(key_rel, constants.DECISION_UNDECIDED)
                             except ValueError: pass
                    except Exception as key_err: logger.error(f"Error generating decision key for block {block.block_id} in {original_path.name}: {key_err}"); decision = constants.DECISION_UNDECIDED
                    if decision != constants.DECISION_DELETE: kept_blocks_content.append(block.original_text)

                if kept_blocks_content:
                    with open(output_filepath, 'w', encoding=constants.DEFAULT_ENCODING) as f_out: f_out.write('\n\n'.join(kept_blocks_content))
                    log_output_path = output_filepath.relative_to(Path.cwd()) if output_filepath.is_relative_to(Path.cwd()) else output_filepath
                    logger.info(f"Successfully wrote {len(kept_blocks_content)} blocks to output file: {log_output_path.name}"); generated_files_count += 1
                else: logger.info(f"No content kept for file {original_path.name}. Output file not generated.")
                processed_files_count += 1
            except Exception as e: # Catch errors during block processing or file writing for *this* file
                logger.error(f"Failed to process or write output for {original_path.name}: {e}", exc_info=True)
                error_files.append(original_path.name); file_proc_success = False; overall_success = False

        logger.info(f"Decision application complete: Processed {processed_files_count}/{total_files} files. Generated {generated_files_count} output files.")
        print(f"\n[*] Decision application complete: {generated_files_count} files generated in '{self.output_dir}'.")
        if error_files: print(f"[Warning] Failed to process output for the following original files: {', '.join(error_files)}")
        return overall_success

    # --- Public Interface Methods for UI ---
    def get_md5_duplicates(self) -> List[List[ContentBlock]]:
        if not self._analysis_completed: logger.warning("Cannot get MD5 duplicates: Analysis not completed."); return []
        return self.md5_duplicates

    def get_semantic_duplicates(self) -> List[Tuple[ContentBlock, ContentBlock, float]]:
        if not self._analysis_completed: logger.warning("Cannot get semantic duplicates: Analysis not completed."); return []
        if self.skip_semantic: logger.info("Semantic analysis was skipped."); return []
        return self.semantic_duplicates

    def update_decision(self, block_key: str, decision: str) -> bool:
        if decision not in [constants.DECISION_KEEP, constants.DECISION_DELETE, constants.DECISION_UNDECIDED]: logger.error(f"Invalid decision value provided: '{decision}'"); return False
        if not block_key: logger.error("Cannot update decision: Invalid block key provided."); return False
        logger.debug(f"Updating decision for key '{block_key}' to '{decision}'")
        self.block_decisions[block_key] = decision; return True

    def get_status_summary(self) -> Dict[str, Any]:
        md5_count = len(self.md5_duplicates); semantic_count = len(self.semantic_duplicates) if not self.skip_semantic else 0
        total_blocks = len(self.blocks_data); decided_count = sum(1 for d in self.block_decisions.values() if d != constants.DECISION_UNDECIDED)
        return {"input_dir": str(self.input_dir) if self.input_dir else "Not set", "decision_file": str(self.decision_file), "output_dir": str(self.output_dir), "skip_semantic": self.skip_semantic, "similarity_threshold": self.similarity_threshold, "analysis_completed": self._analysis_completed, "decisions_loaded": self._decisions_loaded, "total_blocks": total_blocks, "md5_duplicates_groups": md5_count, "semantic_duplicates_pairs": semantic_count, "decided_blocks": decided_count,}

    def set_similarity_threshold(self, threshold: float) -> bool:
        # ==================== Correction: Fixed SyntaxError ====================
        if 0.0 <= threshold <= 1.0:
            logger.info(f"Setting similarity threshold to {threshold}")
            self.similarity_threshold = threshold
            # Update analyzer's threshold as well
            if hasattr(self.semantic_analyzer, 'similarity_threshold'):
                 self.semantic_analyzer.similarity_threshold = threshold
            self._analysis_completed = False # Require re-analysis after changing threshold
            return True
        else:
            logger.error(f"Invalid similarity threshold: {threshold}. Must be between 0.0 and 1.0.")
            return False
        # =======================================================================

    def set_skip_semantic(self, skip: bool) -> None:
        logger.info(f"Setting skip_semantic to {skip}"); self.skip_semantic = skip; self._analysis_completed = False

