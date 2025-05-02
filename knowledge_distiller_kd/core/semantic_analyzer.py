# KD_Tool_CLI/knowledge_distiller_kd/core/semantic_analyzer.py
"""
语义分析器模块，用于检测语义重复内容。

此模块提供以下功能：
1. 加载语义模型
2. 计算文本的语义向量
3. 查找语义相似的内容
4. 处理语义重复内容
"""

import logging
import collections
import time
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, DefaultDict, Set, TYPE_CHECKING
import numpy as np

# 尝试导入sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as st_util # 别名避免与内置 util 冲突
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st_util = None # 定义以避免 NameError
    # 如果库不可用，定义一个假的 SentenceTransformer 以避免 NameError
    class SentenceTransformer: pass # type: ignore

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer # type: ignore

from knowledge_distiller_kd.core.error_handler import (
    KDError, FileOperationError, ModelError, AnalysisError, UserInputError,
    handle_error, safe_file_operation, validate_file_path
)
from knowledge_distiller_kd.core.utils import (
    setup_logger,
    create_decision_key,
    parse_decision_key,
    display_block_preview,
    get_markdown_parser,
    sort_blocks_key,
    logger # 使用 utils 中配置好的 logger
)
from knowledge_distiller_kd.core import constants
# 导入 ContentBlock
from knowledge_distiller_kd.core.document_processor import ContentBlock

logger = logger # 使用 utils 中配置好的 logger

class SemanticAnalyzer:
    """
    语义分析器类，用于查找和处理语义重复内容。
    """

    def __init__(self, tool, similarity_threshold: float = constants.DEFAULT_SIMILARITY_THRESHOLD) -> None:
        """
        初始化语义分析器。

        Args:
            tool: KDToolCLI 实例的引用
            similarity_threshold: 相似度阈值（0.0 到 1.0 之间）
        """
        self.tool = tool
        self.model: Optional[SentenceTransformer] = None
        self.model_name: str = constants.DEFAULT_SEMANTIC_MODEL
        self.model_version: Optional[int] = None # 用于存储模型维度
        self.similarity_threshold: float = max(0.0, min(1.0, similarity_threshold)) # 确保范围
        self.semantic_duplicates: List[Tuple[ContentBlock, ContentBlock, float]] = [] # 存储相似对
        self.semantic_id_to_key: Dict[str, str] = {} # 显示ID -> 决策Key
        self.vector_cache: Dict[str, np.ndarray] = {} # 文本Hash -> 向量
        self._model_loaded: bool = False # 模型加载状态标志

        if not SENTENCE_TRANSFORMERS_AVAILABLE and not self.tool.skip_semantic:
            logger.error("`sentence-transformers` library not found.")
            logger.error("Please install it: pip install sentence-transformers")
            logger.warning("Semantic analysis feature will be disabled.")
            self.tool.skip_semantic = True # 强制跳过

    def load_semantic_model(self) -> None:
        """
        加载 Sentence Transformer 模型。
        """
        if self.tool.skip_semantic:
            logger.info("Semantic analysis is skipped by configuration. Model loading skipped.")
            return
        if self._model_loaded and self.model:
            logger.info("Semantic model '%s' already loaded.", self.model_name)
            return
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Cannot load semantic model: `sentence-transformers` is not installed.")
            self.tool.skip_semantic = True
            return

        try:
            model_name_to_load = self.model_name
            logger.info(f"Loading semantic model: {model_name_to_load} ... (This may take some time)")
            start_time = time.time()

            self.model = SentenceTransformer(model_name_to_load) # type: ignore

            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                 self.model_version = self.model.get_sentence_embedding_dimension()
            elif hasattr(self.model, 'encode'):
                 try:
                      dummy_embedding = self.model.encode("test")
                      self.model_version = len(dummy_embedding)
                 except Exception:
                      logger.warning("Could not determine model embedding dimension.")
                      self.model_version = None
            else:
                 self.model_version = None

            if self.model_version is None:
                logger.warning("Failed to get model embedding dimension information.")

            end_time = time.time()
            self._model_loaded = True
            logger.info(f"Semantic model loaded successfully. Time taken: {end_time - start_time:.2f} seconds")
            if self.model_version:
                logger.info(f"Model embedding dimension: {self.model_version}")

        except Exception as e:
            logger.error(f"Failed to load semantic model '{self.model_name}': {e}", exc_info=True)
            handle_error(e, f"加载语义模型 {self.model_name}")
            logger.warning("Semantic deduplication feature will be unavailable.")
            self.model = None
            self._model_loaded = False
            self.tool.skip_semantic = True

    def _compute_vectors(self, texts: List[str], batch_size: int = constants.DEFAULT_BATCH_SIZE) -> List[np.ndarray]:
        """
        批量计算文本的向量嵌入，使用缓存。

        Args:
            texts: 要计算向量的文本列表
            batch_size: 批处理大小

        Returns:
            List[np.ndarray]: 文本向量列表 (对应输入 texts)
        """
        if not texts or self.model is None:
            return []

        all_vectors: List[Optional[np.ndarray]] = [None] * len(texts) # 初始化结果列表
        texts_to_compute_indices: List[int] = [] # 需要计算的文本的索引
        texts_to_compute: List[str] = []         # 需要计算的文本内容
        hashes_to_compute: List[str] = []        # 需要计算的文本的哈希

        # 检查缓存
        for i, text in enumerate(texts):
            if not text: # 跳过空文本
                 all_vectors[i] = np.array([])
                 continue
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            if text_hash in self.vector_cache:
                all_vectors[i] = self.vector_cache[text_hash]
            else:
                texts_to_compute_indices.append(i)
                texts_to_compute.append(text)
                hashes_to_compute.append(text_hash)

        # 批量计算未缓存的向量
        if texts_to_compute:
            logger.info(f"Computing vectors for {len(texts_to_compute)} new texts (batch size: {batch_size})...")
            computed_vectors_list: List[np.ndarray] = []
            try:
                # Ensure model.encode returns a list or ndarray
                encode_result = self.model.encode(
                    texts_to_compute,
                    batch_size=batch_size,
                    show_progress_bar=True
                )
                if isinstance(encode_result, np.ndarray):
                    computed_vectors_list = [vec for vec in encode_result]
                elif isinstance(encode_result, list):
                    computed_vectors_list = encode_result
                else:
                     raise TypeError(f"Unexpected return type from model.encode: {type(encode_result)}")

            except Exception as e:
                 logger.error(f"Error during batch vector encoding: {e}", exc_info=True)
                 computed_vectors_list = [np.array([])] * len(texts_to_compute)

            # 填充结果并更新缓存
            if len(computed_vectors_list) == len(texts_to_compute):
                 for original_index, computed_vector, text_hash in zip(texts_to_compute_indices, computed_vectors_list, hashes_to_compute):
                     all_vectors[original_index] = computed_vector
                     self.vector_cache[text_hash] = computed_vector
            else:
                 logger.error("Mismatch between computed vectors and texts to compute. Filling with empty arrays.")
                 for original_index in texts_to_compute_indices:
                      all_vectors[original_index] = np.array([])

        # 确保所有条目都被处理（即使是空或错误）
        final_vectors: List[np.ndarray] = []
        for vec_opt in all_vectors:
            if vec_opt is None:
                final_vectors.append(np.array([]))
            else:
                final_vectors.append(vec_opt)

        return final_vectors


    def find_semantic_duplicates(self) -> None:
        """
        使用 Sentence Transformer 模型查找语义相似的块。
        优化：跳过标题块和已被 MD5 标记为删除的块。
        """
        logger.info("Starting semantic similarity analysis (Optimized, skipping titles)...")
        if self.tool.skip_semantic:
            logger.warning("Semantic analysis skipped as requested or due to previous errors.")
            self.semantic_duplicates = []
            return
        if self.model is None:
            logger.error("Semantic model not loaded, cannot perform semantic analysis.")
            self.semantic_duplicates = []
            return
        if not self.tool.blocks_data or len(self.tool.blocks_data) < 2:
            logger.info("Not enough blocks (<2) for semantic comparison.")
            self.semantic_duplicates = []
            return

        start_time = time.time()
        self.semantic_duplicates.clear() # 清空旧结果

        # --- 优化步骤 1: 识别需要分析的块 (非标题, 非MD5删除块) ---
        blocks_to_analyze: List[ContentBlock] = [] # 存储需要计算向量的 ContentBlock 对象
        block_keys_analyzed: Set[str] = set() # 跟踪已处理的块键，避免重复添加
        skipped_titles = 0
        skipped_deleted = 0

        logger.info("Filtering blocks for semantic analysis (excluding titles and MD5 deletes)...")
        for block in self.tool.blocks_data:
            # ==================== 修改：跳过标题块 ====================
            if block.block_type == constants.BLOCK_TYPE_TITLE:
                skipped_titles += 1
                continue
            # ========================================================

            try:
                key = create_decision_key(block.file_path, block.block_id, block.block_type)
                decision = self.tool.block_decisions.get(key, constants.DECISION_UNDECIDED)

                if decision == constants.DECISION_DELETE:
                    skipped_deleted += 1
                    continue # 跳过已被 MD5 删除的块

                # 检查是否已处理过这个key（以防万一）
                if key in block_keys_analyzed:
                    continue

                # 添加到待分析列表
                blocks_to_analyze.append(block)
                block_keys_analyzed.add(key)

            except Exception as e:
                 logger.warning(f"Error processing block {getattr(block, 'block_id', 'N/A')} for semantic pre-filtering: {e}")
                 continue # 跳过有问题的块

        logger.info(f"Semantic pre-filtering complete. Skipped titles: {skipped_titles}, Skipped MD5 deletes: {skipped_deleted}.")

        unique_block_count = len(blocks_to_analyze)
        if unique_block_count < 2:
            logger.info(f"Not enough unique blocks (<2) after filtering for semantic comparison.")
            return

        # --- 优化步骤 2: 计算选中块的向量 ---
        logger.info(f"Computing vectors for {unique_block_count} unique blocks (excluding titles)...")
        texts_to_encode = [block.analysis_text for block in blocks_to_analyze]
        block_vectors = self._compute_vectors(texts_to_encode)

        # 过滤掉计算失败的向量
        valid_indices = [i for i, vec in enumerate(block_vectors) if vec is not None and vec.size > 0]
        if len(valid_indices) < unique_block_count:
             logger.warning(f"Vector computation failed for {unique_block_count - len(valid_indices)} blocks. Proceeding with valid vectors.")
             if len(valid_indices) < 2:
                  logger.error("Not enough valid vectors (<2) for comparison.")
                  return
             blocks_to_analyze = [blocks_to_analyze[i] for i in valid_indices]
             block_vectors = [block_vectors[i] for i in valid_indices]
             unique_block_count = len(blocks_to_analyze)


        # --- 优化步骤 3: 查找语义相似对 ---
        logger.info(f"Finding semantically similar pairs among {unique_block_count} blocks using threshold {self.similarity_threshold}...")
        try:
            if not block_vectors: # 如果过滤后没有向量了
                 logger.info("No valid vectors remaining for similarity calculation.")
                 return

            embeddings = np.array(block_vectors)
            if embeddings.ndim == 1: # 处理只有一个有效向量的情况
                 logger.info("Only one valid vector remaining, cannot compute similarity matrix.")
                 return
            if embeddings.size == 0: # 处理空数组的情况
                logger.info("Embeddings array is empty, cannot compute similarity matrix.")
                return

            # 添加检查，确保 st_util 存在
            if st_util is None:
                logger.error("Sentence Transformers util (st_util) is not available. Cannot calculate similarity.")
                return

            similarity_matrix = st_util.cos_sim(embeddings, embeddings)

            found_pairs = 0
            for i in range(unique_block_count):
                for j in range(i + 1, unique_block_count):
                    similarity_score = similarity_matrix[i][j].item()

                    if similarity_score >= self.similarity_threshold:
                        block1 = blocks_to_analyze[i]
                        block2 = blocks_to_analyze[j]
                        self.semantic_duplicates.append((block1, block2, similarity_score))
                        found_pairs += 1

            logger.info(f"Similarity calculation complete. Found {found_pairs} pairs above threshold.")

        except Exception as e:
             logger.error(f"Error during similarity calculation: {e}", exc_info=True)
             handle_error(e, "计算语义相似度")
             return

        self.semantic_duplicates.sort(key=lambda x: x[2], reverse=True)

        end_time = time.time()
        logger.info(f"Semantic analysis complete. Found {len(self.semantic_duplicates)} similar pairs (titles excluded). Time taken: {end_time - start_time:.2f} seconds")


    def _display_semantic_duplicates_list(self) -> bool:
        """
        在控制台显示所有语义相似对，并为每个块分配唯一的显示 ID。
        """
        print("\n--- 语义相似内容块列表 (已排除标题) ---")
        if self.tool.skip_semantic:
            print("[*] 语义分析已被跳过。")
            return False
        if not self.semantic_duplicates:
            print(f"[*] 未找到相似度 > {self.similarity_threshold:.2f} 的语义相似项 (已排除标题)。")
            return False

        self.semantic_id_to_key.clear()
        print("\n对号 | 相似度 | 显示ID | 文件名 (类型 #块ID)         | 当前决策  | 内容预览 (原始文本)")
        print("-----|---------|--------|-----------------------------|-----------|----------------------")
        pair_num = 0
        total_pairs = len(self.semantic_duplicates)

        for pair_idx, (block1, block2, score) in enumerate(self.semantic_duplicates):
            pair_num += 1
            display_id1 = f"s{pair_num}-1"
            display_id2 = f"s{pair_num}-2"

            try:
                key1 = create_decision_key(block1.file_path, block1.block_id, block1.block_type)
                key2 = create_decision_key(block2.file_path, block2.block_id, block2.block_type)
            except Exception as e:
                 logger.warning(f"无法为相似对 {pair_num} 中的块创建决策键: {e}. 跳过此对的显示。")
                 continue

            self.semantic_id_to_key[display_id1] = key1
            self.semantic_id_to_key[display_id2] = key2

            current_decision1 = self.tool.block_decisions.get(key1, constants.DECISION_UNDECIDED)
            current_decision2 = self.tool.block_decisions.get(key2, constants.DECISION_UNDECIDED)

            preview1 = display_block_preview(block1.original_text)
            preview2 = display_block_preview(block2.original_text)

            file_name1 = Path(block1.file_path).name
            file_name2 = Path(block2.file_path).name

            info_str1 = f"{file_name1} ({block1.block_type} #{block1.block_id})"
            info_str2 = f"{file_name2} ({block2.block_type} #{block2.block_id})"

            print(f" {pair_num:<4}| {score:.4f}  | {display_id1:<6} | {info_str1:<27} | {current_decision1:<9} | {preview1}")
            print(f"     |         | {display_id2:<6} | {info_str2:<27} | {current_decision2:<9} | {preview2}")
            print("-----|---------|--------|-----------------------------|-----------|----------------------")

        print(f"\n[*] 共显示 {pair_num}/{total_pairs} 对语义相似块 (已排除标题)。")
        return True


    def review_semantic_duplicates_interactive(self) -> None:
        """
        提供交互式界面，让用户处理语义相似对。
        """
        logger.info("Starting interactive review of semantic duplicates...")

        if not self._display_semantic_duplicates_list():
            logger.info("No semantic duplicates to review (titles excluded).")
            print("\n--- 语义相似项处理完成 ---")
            return

        while True:
            print("\n语义相似项处理选项:")
            print("  k <ID> [<ID>...]      - 保留指定ID的块 (Keep)")
            print("  d <ID> [<ID>...]      - 删除指定ID的块 (Delete)")
            print("  r <ID> [<ID>...]      - 重置指定ID块的决策为 undecided (Reset)")
            print("  k1d2 <对号> [<对号>...] - 对于指定对号，保留第1个块，删除第2个块")
            print("  k2d1 <对号> [<对号>...] - 对于指定对号，保留第2个块，删除第1个块")
            print("  ka <对号> [<对号>...]   - 对于指定对号，保留两个块 (Keep All in pair)")
            print("  da <对号> [<对号>...]   - 对于指定对号，删除两个块 (Delete All in pair)")
            print("  ra <对号> [<对号>...]   - 对于指定对号，重置两个块的决策 (Reset All in pair)")
            print("  save                - 保存当前所有决策到文件")
            print("  q                   - 完成语义相似项处理并退出")

            action = input("请输入操作: ").lower().strip()
            logger.debug(f"User input for semantic review: '{action}'")

            redisplay_list = False

            if action == 'q':
                logger.info("Quitting interactive semantic review.")
                break
            elif action == 'save':
                logger.info("User chose 'save' during semantic review.")
                if self.tool.save_decisions():
                    print(" [*] 决策已保存。")
                else:
                    print(" [!] 保存决策时遇到问题。")
                continue

            parts = action.split()
            if not parts:
                continue

            command = parts[0]
            ids_input = parts[1:]

            if command not in ['k', 'd', 'r', 'k1d2', 'k2d1', 'ka', 'da', 'ra']:
                logger.warning(f"Invalid command: '{command}'")
                print("[错误] 无效的操作命令。")
                continue

            if not ids_input:
                logger.warning("Command requires at least one ID or pair number.")
                print("[错误] 命令需要至少一个 ID 或对号。")
                continue

            processed_count = 0
            error_occurred = False
            try:
                if command in ['k', 'd', 'r']:
                    target_decision = {
                        'k': constants.DECISION_KEEP,
                        'd': constants.DECISION_DELETE,
                        'r': constants.DECISION_UNDECIDED
                    }[command]
                    for display_id in ids_input:
                        if display_id in self.semantic_id_to_key:
                            key = self.semantic_id_to_key[display_id]
                            self.tool.block_decisions[key] = target_decision
                            print(f"  [*] 块 {display_id} 的决策更新为: {target_decision}")
                            processed_count += 1
                        else:
                            print(f"  [!] 错误: 无效的显示 ID '{display_id}'。")
                            error_occurred = True
                elif command in ['k1d2', 'k2d1', 'ka', 'da', 'ra']:
                    for pair_num_str in ids_input:
                        try:
                            pair_num = int(pair_num_str)
                            id1 = f"s{pair_num}-1"
                            id2 = f"s{pair_num}-2"

                            if id1 in self.semantic_id_to_key and id2 in self.semantic_id_to_key:
                                key1 = self.semantic_id_to_key[id1]
                                key2 = self.semantic_id_to_key[id2]
                                decision1, decision2 = constants.DECISION_UNDECIDED, constants.DECISION_UNDECIDED

                                if command == 'k1d2': decision1, decision2 = constants.DECISION_KEEP, constants.DECISION_DELETE
                                elif command == 'k2d1': decision1, decision2 = constants.DECISION_DELETE, constants.DECISION_KEEP
                                elif command == 'ka':   decision1, decision2 = constants.DECISION_KEEP, constants.DECISION_KEEP
                                elif command == 'da':   decision1, decision2 = constants.DECISION_DELETE, constants.DECISION_DELETE
                                elif command == 'ra':   decision1, decision2 = constants.DECISION_UNDECIDED, constants.DECISION_UNDECIDED

                                self.tool.block_decisions[key1] = decision1
                                self.tool.block_decisions[key2] = decision2
                                print(f"  [*] 第 {pair_num} 对块 ({id1}, {id2}) 的决策已更新。")
                                processed_count += 2
                            else:
                                print(f"  [!] 错误: 无效的对号 '{pair_num_str}' 或找不到对应的块 ID。")
                                error_occurred = True
                        except ValueError:
                            print(f"  [!] 错误: 对号 '{pair_num_str}' 必须是数字。")
                            error_occurred = True

            except Exception as e:
                 logger.error(f"Error processing command '{action}': {e}", exc_info=True)
                 print(f"[错误] 处理命令时发生意外错误: {e}")
                 error_occurred = True

            if processed_count > 0:
                 print(f"[*] 共处理了 {processed_count} 个块的决策。")
                 redisplay_list = True

            if error_occurred:
                 print("[!] 处理过程中发生错误，部分操作可能未完成。")

            if redisplay_list and not error_occurred:
                # 重新显示列表以反映更改
                if not self._display_semantic_duplicates_list():
                     # 如果重新显示时发现没有更多项了，就退出循环
                     logger.info("No more semantic duplicates to review after update.")
                     break

        logger.info("Finished interactive review of semantic duplicates.")
        print("\n--- 语义相似项处理完成 ---")

