# KD_Tool_CLI/knowledge_distiller_kd/core/semantic_analyzer.py
"""
语义分析器模块，用于检测语义重复内容。

此模块提供以下功能：
1. 加载语义模型
2. 计算文本的语义向量
3. 查找语义相似的内容
4. 处理语义重复内容
"""

# [DEPENDENCIES]
# 1. Python Standard Library: collections, numpy, time, re, hashlib
# 2. 需要安装：sentence-transformers, numpy
# 3. 同项目模块: constants, utils, error_handler, document_processor

import logging
import collections
import time
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, DefaultDict, Set
import numpy as np

# 尝试导入sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as st_util # 别名避免与内置 util 冲突
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st_util = None # 定义以避免 NameError
    SentenceTransformer = None # 定义以避免 NameError

from knowledge_distiller_kd.core.error_handler import (
    KDError, FileOperationError, ModelError, AnalysisError, UserInputError, # 添加 UserInputError
    handle_error, safe_file_operation, validate_file_path
)
from knowledge_distiller_kd.core.utils import (
    setup_logger,
    create_decision_key,
    parse_decision_key,
    # extract_text_from_children, # 已移到 ContentBlock
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

    该类负责：
    1. 加载和管理语义模型
    2. 计算文本的语义向量
    3. 查找语义相似的内容块
    4. 生成语义重复的报告
    5. 管理语义重复的决策

    Attributes:
        tool: KDToolCLI 实例的引用
        model: SentenceTransformer 模型实例
        model_name: 使用的模型名称
        model_version: 模型维度信息
        similarity_threshold: 相似度阈值
        semantic_duplicates: 存储语义相似对 [(block1, block2, score), ...]
        semantic_id_to_key: 存储交互时显示的语义 ID 到决策键的映射 {display_id: decision_key}
        vector_cache: 存储文本哈希到向量的缓存 {text_hash: np.ndarray}
        _model_loaded: 标志模型是否已成功加载
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

        # 初始化时检查库可用性，但不在此处加载模型
        if not SENTENCE_TRANSFORMERS_AVAILABLE and not self.tool.skip_semantic:
            logger.error("`sentence-transformers` library not found.")
            logger.error("Please install it: pip install sentence-transformers")
            logger.warning("Semantic analysis feature will be disabled.")
            self.tool.skip_semantic = True # 强制跳过

    def load_semantic_model(self) -> None:
        """
        加载 Sentence Transformer 模型。
        由 KDToolCLI 在需要时调用。
        """
        # 如果设置了跳过，或者模型已加载，或者库不可用，则直接返回
        if self.tool.skip_semantic:
            logger.info("Semantic analysis is skipped by configuration. Model loading skipped.")
            return
        if self._model_loaded and self.model:
            logger.info("Semantic model '%s' already loaded.", self.model_name)
            return
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Cannot load semantic model: `sentence-transformers` is not installed.")
            self.tool.skip_semantic = True # 确保跳过状态一致
            return

        try:
            model_name_to_load = self.model_name
            logger.info(f"Loading semantic model: {model_name_to_load} ... (This may take some time)")
            start_time = time.time()

            # 使用 SentenceTransformer 加载模型
            self.model = SentenceTransformer(model_name_to_load)

            # 获取模型维度信息
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                 self.model_version = self.model.get_sentence_embedding_dimension()
            elif hasattr(self.model, 'encode'):
                 # 尝试编码一个虚拟句子来获取维度
                 try:
                      dummy_embedding = self.model.encode("test")
                      self.model_version = len(dummy_embedding)
                 except Exception:
                      logger.warning("Could not determine model embedding dimension.")
                      self.model_version = None # 未知维度
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
            self.tool.skip_semantic = True # 加载失败，强制跳过

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

        all_vectors = [None] * len(texts) # 初始化结果列表
        texts_to_compute_indices = [] # 需要计算的文本的索引
        texts_to_compute = []         # 需要计算的文本内容
        hashes_to_compute = []        # 需要计算的文本的哈希

        # 检查缓存
        for i, text in enumerate(texts):
            if not text: # 跳过空文本
                 all_vectors[i] = np.array([]) # 或者使用适合下游处理的空表示
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
            computed_vectors = []
            try:
                computed_vectors = self.model.encode(
                    texts_to_compute,
                    batch_size=batch_size,
                    show_progress_bar=True # 显示进度条
                )
            except Exception as e:
                 logger.error(f"Error during batch vector encoding: {e}", exc_info=True)
                 # 可以选择如何处理错误，例如为失败的向量填充 None 或空数组
                 computed_vectors = [np.array([])] * len(texts_to_compute) # 示例：填充空数组

            # 填充结果并更新缓存
            if len(computed_vectors) == len(texts_to_compute):
                 for original_index, computed_vector, text_hash in zip(texts_to_compute_indices, computed_vectors, hashes_to_compute):
                     all_vectors[original_index] = computed_vector
                     self.vector_cache[text_hash] = computed_vector
            else:
                 logger.error("Mismatch between computed vectors and texts to compute. Filling with empty arrays.")
                 for original_index in texts_to_compute_indices:
                      all_vectors[original_index] = np.array([])


        # 确保所有条目都被处理（即使是空或错误）
        for i in range(len(all_vectors)):
             if all_vectors[i] is None:
                  all_vectors[i] = np.array([]) # 确保返回列表不含 None

        # 过滤掉空数组（如果下游无法处理）？或者由下游处理
        # return [v for v in all_vectors if v.size > 0]
        return all_vectors


    def find_semantic_duplicates(self) -> None:
        """
        使用 Sentence Transformer 模型查找语义相似的块。
        优化：跳过对 MD5 精确重复块（非保留块）的向量计算和比较。
        """
        logger.info("Starting semantic similarity analysis (Optimized)...")
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

        # --- 优化步骤 1: 识别需要分析的唯一块 (非MD5删除块) ---
        blocks_to_analyze = [] # 存储需要计算向量的 ContentBlock 对象
        block_keys_analyzed = set() # 跟踪已处理的块键，避免因排序问题重复添加

        # 确保 MD5 分析已运行并自动决策
        if not self.tool.md5_analyzer or not self.tool.md5_analyzer.md5_duplicates:
             logger.info("MD5 analysis did not find duplicates or was not run. Analyzing all blocks for semantics.")
             # 如果没有 MD5 重复，理论上所有块都是唯一的，都需要分析
             blocks_to_analyze = self.tool.blocks_data # 复制列表或直接使用
             # 需要确保这些块的key也被记录，以防万一
             for block in blocks_to_analyze:
                  try:
                      key = create_decision_key(block.file_path, block.block_id, block.block_type)
                      block_keys_analyzed.add(key)
                  except Exception: pass # 忽略无法创建键的块
        else:
             logger.info("MD5 analysis found duplicates. Selecting non-deleted blocks for semantic analysis.")
             # 选择未被 MD5 标记为 'delete' 的块进行语义分析
             for block in self.tool.blocks_data:
                  try:
                      key = create_decision_key(block.file_path, block.block_id, block.block_type)
                      decision = self.tool.block_decisions.get(key, constants.DECISION_UNDECIDED)

                      if decision != constants.DECISION_DELETE and key not in block_keys_analyzed:
                           # 只添加未被删除且未被处理的块
                           blocks_to_analyze.append(block)
                           block_keys_analyzed.add(key)
                      # else:
                           # logger.debug(f"Skipping block {key} (decision: {decision}) for semantic analysis.")
                  except Exception as e:
                       logger.warning(f"Error processing block {getattr(block, 'block_id', 'N/A')} for semantic pre-filtering: {e}")
                       continue # 跳过有问题的块

        unique_block_count = len(blocks_to_analyze)
        if unique_block_count < 2:
            logger.info("Not enough unique blocks (<2) after MD5 filtering for semantic comparison.")
            return

        # --- 优化步骤 2: 计算选中块的向量 ---
        logger.info(f"Computing vectors for {unique_block_count} unique blocks...")
        # 提取 analysis_text 用于计算向量
        texts_to_encode = [block.analysis_text for block in blocks_to_analyze]
        block_vectors = self._compute_vectors(texts_to_encode)

        # 过滤掉计算失败的向量（例如空数组）及其对应的块
        valid_indices = [i for i, vec in enumerate(block_vectors) if vec is not None and vec.size > 0]
        if len(valid_indices) < len(blocks_to_analyze):
             logger.warning(f"Vector computation failed for {len(blocks_to_analyze) - len(valid_indices)} blocks. Proceeding with valid vectors.")
             if len(valid_indices) < 2:
                  logger.error("Not enough valid vectors (<2) for comparison.")
                  return
             # 更新 blocks_to_analyze 和 block_vectors 以只包含有效的条目
             blocks_to_analyze = [blocks_to_analyze[i] for i in valid_indices]
             block_vectors = [block_vectors[i] for i in valid_indices]
             unique_block_count = len(blocks_to_analyze) # 更新计数


        # --- 优化步骤 3: 查找语义相似对 ---
        logger.info(f"Finding semantically similar pairs among {unique_block_count} blocks using threshold {self.similarity_threshold}...")
        try:
            # 使用 sentence_transformers.util 进行高效计算
            # 需要确保 block_vectors 是 numpy 数组或 tensor 列表
            embeddings = np.array(block_vectors) # 转换为 numpy 数组
            # 计算所有对之间的余弦相似度
            similarity_matrix = st_util.cos_sim(embeddings, embeddings)

            # 遍历上三角矩阵（不包括对角线）查找相似对
            found_pairs = 0
            for i in range(unique_block_count):
                for j in range(i + 1, unique_block_count):
                    similarity_score = similarity_matrix[i][j].item() # 获取标量值

                    if similarity_score >= self.similarity_threshold:
                        block1 = blocks_to_analyze[i]
                        block2 = blocks_to_analyze[j]
                        self.semantic_duplicates.append((block1, block2, similarity_score))
                        found_pairs += 1
                        # logger.debug(f"Found similar pair: {block1.block_id} & {block2.block_id}, score: {similarity_score:.4f}")

            logger.info(f"Similarity calculation complete. Found {found_pairs} pairs above threshold.")

        except Exception as e:
             logger.error(f"Error during similarity calculation: {e}", exc_info=True)
             handle_error(e, "计算语义相似度")
             # 出错则不继续，保留已找到的（可能为空）
             return

        # 按相似度降序排序结果
        self.semantic_duplicates.sort(key=lambda x: x[2], reverse=True)

        end_time = time.time()
        logger.info(f"Semantic analysis complete. Found {len(self.semantic_duplicates)} similar pairs. Time taken: {end_time - start_time:.2f} seconds")


    def _display_semantic_duplicates_list(self) -> bool:
        """
        在控制台显示所有语义相似对，并为每个块分配唯一的显示 ID。
        """
        print("\n--- 语义相似内容块列表 ---")
        if self.tool.skip_semantic:
            print("[*] 语义分析已被跳过。")
            return False
        if not self.semantic_duplicates:
            print(f"[*] 未找到相似度 > {self.similarity_threshold:.2f} 的语义相似项。")
            return False

        self.semantic_id_to_key.clear() # 清空旧的映射
        print("\n对号 | 相似度 | 显示ID | 文件名 (类型 #块ID)         | 当前决策  | 内容预览 (原始文本)")
        print("-----|---------|--------|-----------------------------|-----------|----------------------")
        pair_num = 0
        total_pairs = len(self.semantic_duplicates)

        # 遍历找到的语义相似对 (已按分数降序排列)
        for pair_idx, (block1, block2, score) in enumerate(self.semantic_duplicates):
            pair_num += 1
            # 显示进度
            # if pair_num % 10 == 0:
            #     print(f"\n[*] 正在显示第 {pair_num}/{total_pairs} 对...")

            # 为每对中的两个块创建显示 ID，例如 s1-1, s1-2
            display_id1 = f"s{pair_num}-1"
            display_id2 = f"s{pair_num}-2"

            # 创建决策键
            try:
                key1 = create_decision_key(block1.file_path, block1.block_id, block1.block_type)
                key2 = create_decision_key(block2.file_path, block2.block_id, block2.block_type)
            except Exception as e:
                 logger.warning(f"无法为相似对 {pair_num} 中的块创建决策键: {e}. 跳过此对的显示。")
                 continue

            # 存储显示 ID 到决策键的映射
            self.semantic_id_to_key[display_id1] = key1
            self.semantic_id_to_key[display_id2] = key2

            # 获取当前决策状态
            current_decision1 = self.tool.block_decisions.get(key1, constants.DECISION_UNDECIDED)
            current_decision2 = self.tool.block_decisions.get(key2, constants.DECISION_UNDECIDED)

            # 生成内容预览 (使用原始文本)
            preview1 = display_block_preview(block1.original_text)
            preview2 = display_block_preview(block2.original_text)

            # 获取文件名字符串
            file_name1 = Path(block1.file_path).name
            file_name2 = Path(block2.file_path).name

            # 格式化块信息
            info_str1 = f"{file_name1} ({block1.block_type} #{block1.block_id})"
            info_str2 = f"{file_name2} ({block2.block_type} #{block2.block_id})"

            # 打印第一块的信息
            print(f" {pair_num:<4}| {score:.4f}  | {display_id1:<6} | {info_str1:<27} | {current_decision1:<9} | {preview1}")
            # 打印第二块的信息
            print(f"     |         | {display_id2:<6} | {info_str2:<27} | {current_decision2:<9} | {preview2}")
            print("-----|---------|--------|-----------------------------|-----------|----------------------")

        print(f"\n[*] 共显示 {pair_num}/{total_pairs} 对语义相似块。")
        return True


    def review_semantic_duplicates_interactive(self) -> None:
        """
        提供交互式界面，让用户处理语义相似对。
        """
        logger.info("Starting interactive review of semantic duplicates...")

        # 初始显示列表
        if not self._display_semantic_duplicates_list():
            logger.info("No semantic duplicates to review.")
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

            redisplay_list = False # 默认操作后不自动刷新列表，除非是查询类或无效输入

            if action == 'q':
                logger.info("Quitting interactive semantic review.")
                break
            elif action == 'save':
                logger.info("User chose 'save' during semantic review.")
                if self.tool.save_decisions():
                    print(" [*] 决策已保存。")
                else:
                    print(" [!] 保存决策时遇到问题。")
                continue # 保存后继续显示选项

            parts = action.split()
            if not parts:
                continue # 忽略空输入

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
                                decision1, decision2 = constants.DECISION_UNDECIDED, constants.DECISION_UNDECIDED # 默认

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
                 redisplay_list = True # 成功处理后刷新列表

            if error_occurred:
                 print("[!] 处理过程中发生错误，部分操作可能未完成。")
                 # 不刷新列表可能更好，让用户看到错误信息

            # 如果需要且没有错误，重新显示列表
            if redisplay_list and not error_occurred:
                self._display_semantic_duplicates_list()

        logger.info("Finished interactive review of semantic duplicates.")
        print("\n--- 语义相似项处理完成 ---")
        