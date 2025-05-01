"""
语义分析器模块，用于检测语义重复内容。

此模块提供以下功能：
1. 加载语义模型
2. 计算文本的语义向量
3. 查找语义相似的内容
4. 处理语义重复内容
"""

# [DEPENDENCIES]
# 1. Python Standard Library: collections, numpy
# 2. 需要安装：sentence-transformers (可选)
# 3. 同项目模块: constants, utils, error_handler (使用绝对导入)

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
    from sentence_transformers import util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from knowledge_distiller_kd.core.error_handler import (
    KDError, FileOperationError, ModelError, AnalysisError,
    handle_error, safe_file_operation, validate_file_path
)
from knowledge_distiller_kd.core.utils import (
    setup_logger,
    create_decision_key,
    parse_decision_key,
    extract_text_from_children,
    display_block_preview,
    get_markdown_parser,
    sort_blocks_key,
    logger
)
from knowledge_distiller_kd.core import constants

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
        similarity_threshold: 相似度阈值
        duplicate_blocks: 存储重复内容块的字典 {hash: [(file, index, type, text), ...]}
        semantic_id_to_key: 存储语义 ID 到决策键的映射 {id: key}
    """

    def __init__(self, tool, similarity_threshold: float = 0.85) -> None:
        """
        初始化语义分析器。

        Args:
            tool: KDToolCLI 实例的引用
            similarity_threshold: 相似度阈值（0.0 到 1.0 之间）
        """
        self.tool = tool
        self.model = None
        self.model_name = constants.DEFAULT_SEMANTIC_MODEL
        self.similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        self.duplicate_blocks = {}
        self.semantic_id_to_key = {}
        self.semantic_duplicates = [] # 存储语义相似对
        self.vector_cache = {} # 向量缓存
        self._model_loaded = False # 模型加载状态标志

        # 检查 sentence_transformers 是否可用
        if not SENTENCE_TRANSFORMERS_AVAILABLE and not self.tool.skip_semantic:
            logger.error("`sentence-transformers` library not found.")
            logger.error("Please install it: pip install sentence-transformers")
            logger.warning("Semantic analysis feature will be disabled.")
            self.tool.skip_semantic = True # 强制跳过语义分析

    def load_semantic_model(self) -> None:
        """
        加载 Sentence Transformer 模型。

        Process:
            1. 检查是否应该跳过语义分析
            2. 检查模型是否已加载
            3. 检查库是否可用
            4. 加载指定的模型
            5. 记录加载时间
            6. 验证模型版本

        Note:
            - 如果加载失败，会禁用语义分析功能
            - 使用 constants.DEFAULT_SEMANTIC_MODEL 作为默认模型
            - 加载过程可能需要较长时间
            - 支持模型缓存和预加载
        """
        # 如果设置了跳过，或者模型已加载，或者库不可用，则直接返回
        if self.tool.skip_semantic:
            logger.info("Semantic analysis is skipped by configuration.")
            return
        if self._model_loaded and self.model:
            logger.info("Semantic model already loaded.")
            return
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Cannot load semantic model because `sentence-transformers` is not installed.")
            self.tool.skip_semantic = True
            return

        try:
            model_name = constants.DEFAULT_SEMANTIC_MODEL
            logger.info(f"Loading semantic model: {model_name} ... (This may take some time)")
            start_time = time.time()
            
            # 使用 SentenceTransformer 加载模型
            self.model = SentenceTransformer(model_name)
            
            # 获取并验证模型版本
            self.model_version = self.model.get_sentence_embedding_dimension()
            if not self.model_version:
                raise ValueError("Failed to get model version information")
                
            end_time = time.time()
            self._model_loaded = True
            logger.info(f"Semantic model loaded successfully. Time taken: {end_time - start_time:.2f} seconds")
            logger.info(f"Model version: {self.model_version}")
            
        except Exception as e:
            logger.error(f"Failed to load semantic model: {constants.DEFAULT_SEMANTIC_MODEL}", exc_info=True)
            logger.warning("Semantic deduplication feature will be unavailable.")
            self.model = None
            self._model_loaded = False
            self.tool.skip_semantic = True # 加载失败，强制跳过

    def _compute_vectors(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        批量计算文本的向量嵌入。

        Args:
            texts: 要计算向量的文本列表
            batch_size: 批处理大小

        Returns:
            List[np.ndarray]: 文本向量列表

        Note:
            - 使用批处理提高效率
            - 支持向量缓存
            - 自动处理空文本
        """
        if not texts:
            return []

        vectors = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_vectors = self.model.encode(batch_texts, show_progress_bar=False)
            # 缓存向量
            for text, vector in zip(batch_texts, batch_vectors):
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                self.vector_cache[text_hash] = vector
            vectors.extend(batch_vectors)
        return vectors

    def _get_cached_vector(self, text: str) -> Optional[np.ndarray]:
        """
        从缓存中获取文本的向量，如果不存在则计算并缓存。

        Args:
            text: 要获取向量的文本

        Returns:
            Optional[np.ndarray]: 文本的向量，如果计算失败则返回 None
        """
        # 计算文本的哈希值作为缓存键
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        if text_hash in self.vector_cache:
            return self.vector_cache[text_hash]
            
        try:
            vector = self.model.encode(text, show_progress_bar=False)
            self.vector_cache[text_hash] = vector
            return vector
        except Exception as e:
            logger.warning(f"Failed to compute vector for text: {e}")
            return None

    def find_semantic_duplicates(self) -> None:
        """
        使用 Sentence Transformer 模型查找语义相似的块。
        优化：跳过对 MD5 精确重复块的向量计算和比较。

        Process:
            1. 检查是否应该执行语义分析
            2. 识别唯一块（基于 MD5）
            3. 计算唯一块的向量嵌入
            4. 查找语义相似对
            5. 映射回原始块

        Note:
            - 使用余弦相似度计算相似度
            - 只比较唯一块，避免重复计算
            - 使用阈值过滤低相似度对
            - 结果按相似度降序排序
            - 支持批处理和向量缓存
        """
        logger.info("Starting semantic similarity analysis (Optimized)...")
        # 检查是否应跳过或是否可以执行
        if self.tool.skip_semantic:
            logger.warning("Semantic analysis skipped as requested or due to previous errors.")
            self.semantic_duplicates = []
            return
        if self.model is None:
            logger.error("Semantic model not loaded, cannot perform semantic analysis.")
            self.semantic_duplicates = []
            return
        if len(self.tool.blocks_data) < 2:
            logger.info("Not enough blocks (<2) for semantic comparison.")
            self.semantic_duplicates = []
            return
        # 检查 MD5 分析是否已运行并有结果
        if not hasattr(self.tool, 'md5_analyzer'):
             logger.error("MD5 analyzer instance not found. Cannot perform optimized semantic analysis.")
             return

        start_time = time.time()
        self.semantic_duplicates = [] # 清空旧结果
        self.vector_cache.clear() # 清空向量缓存

        # --- 优化步骤 1: 识别唯一块 ---
        unique_blocks_info = [] # 存储唯一块的信息 (原始 block_info)
        original_indices_map = [] # 存储 unique_blocks_info 中每个块在原始 blocks_data 中的索引
        md5_hash_to_representative_index = {} # {md5_hash: index_in_unique_blocks_info}
        unique_block_count = 0

        # 遍历所有块，识别唯一块
        for block_info in self.tool.blocks_data:
            file_path, b_index, b_type, text_to_hash = block_info
            try:
                # 计算文本的 MD5 哈希值
                content_with_type = f"{b_type}::{text_to_hash}"
                hash_object = hashlib.md5(content_with_type.encode('utf-8'))
                hex_dig = hash_object.hexdigest()

                # 如果这个 MD5 哈希值已经存在，跳过这个块
                if hex_dig in md5_hash_to_representative_index:
                    continue

                # 这是一个新的唯一块
                md5_hash_to_representative_index[hex_dig] = unique_block_count
                unique_blocks_info.append(block_info)
                original_indices_map.append(len(unique_blocks_info) - 1)
                unique_block_count += 1

            except Exception as e:
                logger.warning(f"Error processing block {file_path}#{b_index}: {e}")
                continue

        if unique_block_count < 2:
            logger.info("Not enough unique blocks (<2) for semantic comparison.")
            return

        # --- 优化步骤 2: 计算唯一块的向量 ---
        logger.info(f"Computing vectors for {unique_block_count} unique blocks...")
        unique_texts = [info[3] for info in unique_blocks_info] # 提取文本内容
        unique_vectors = self._compute_vectors(unique_texts) # 批量计算向量

        if len(unique_vectors) != unique_block_count:
            logger.error("Vector computation failed for some blocks.")
            return

        # --- 优化步骤 3: 查找语义相似对 ---
        logger.info("Finding semantic similar pairs...")
        similarity_matrix = util.pytorch_cos_sim(unique_vectors, unique_vectors)
        
        # 遍历上三角矩阵（不包括对角线）
        for i in range(len(unique_vectors)):
            for j in range(i + 1, len(unique_vectors)):
                similarity = similarity_matrix[i][j].item()
                
                # 如果相似度超过阈值，添加到结果中
                if similarity >= self.tool.similarity_threshold:
                    self.semantic_duplicates.append((
                        unique_blocks_info[i],
                        unique_blocks_info[j],
                        similarity
                    ))

        # 按相似度降序排序结果
        self.semantic_duplicates.sort(key=lambda x: x[2], reverse=True)
        
        end_time = time.time()
        logger.info(f"Semantic analysis complete. Found {len(self.semantic_duplicates)} similar pairs. Time taken: {end_time - start_time:.2f} seconds")

    def _display_semantic_duplicates_list(self) -> bool:
        """
        在控制台显示所有语义相似对，并为每个块分配唯一的显示 ID。

        Process:
            1. 清空并重新构建显示 ID 到决策键的映射
            2. 按相似度降序排序相似对
            3. 为每对中的块分配显示 ID
            4. 显示块信息和预览
            5. 显示进度信息

        Returns:
            bool: 如果有相似项显示返回 True，否则返回 False
        """
        print("\n--- 语义相似内容块列表 ---")
        if self.tool.skip_semantic:
            print("[*] 语义分析已被跳过。")
            return False
        if not self.semantic_duplicates:
            print(f"[*] 未找到相似度 > {self.tool.similarity_threshold} 的语义相似项。")
            return False

        self.semantic_id_to_key.clear() # 清空旧的映射
        print("对号 | 相似度 | 块ID | 文件名 (类型 #索引) | 当前决策 | 内容预览")
        print("-----|---------|------|-----------------------|------------|----------")
        pair_num = 0
        total_pairs = len(self.semantic_duplicates)

        # 遍历找到的语义相似对 (按分数降序排列显示)
        sorted_duplicates = sorted(self.semantic_duplicates, key=lambda x: x[2], reverse=True)

        for pair_idx, (info1, info2, score) in enumerate(sorted_duplicates):
            pair_num += 1
            # 显示进度
            if pair_num % 10 == 0:
                print(f"\n[*] 正在显示第 {pair_num}/{total_pairs} 对...")
            
            # 为每对中的两个块创建显示 ID，例如 s1-1, s1-2
            id1 = f"s{pair_num}-1"
            id2 = f"s{pair_num}-2"

            file_path1, b_index1, b_type1, b_text1 = info1
            file_path2, b_index2, b_type2, b_text2 = info2

            # 创建决策键
            try:
                abs_path1_str = str(Path(file_path1).resolve())
                key1 = create_decision_key(abs_path1_str, b_index1, b_type1)
            except Exception as e:
                 logger.warning(f"无法为块创建决策键 {file_path1}#{b_index1}: {e}. 跳过此对的显示。")
                 continue
            try:
                abs_path2_str = str(Path(file_path2).resolve())
                key2 = create_decision_key(abs_path2_str, b_index2, b_type2)
            except Exception as e:
                 logger.warning(f"无法为块创建决策键 {file_path2}#{b_index2}: {e}. 跳过此对的显示。")
                 continue

            # 存储显示 ID 到决策键的映射
            self.semantic_id_to_key[id1] = key1
            self.semantic_id_to_key[id2] = key2

            # 获取当前决策状态
            current_decision1 = self.tool.block_decisions.get(key1, 'undecided')
            current_decision2 = self.tool.block_decisions.get(key2, 'undecided')

            # 生成内容预览
            preview1 = display_block_preview(b_text1)
            preview2 = display_block_preview(b_text2)

            # 获取文件名字符串
            file_name1 = file_path1.name if isinstance(file_path1, Path) else str(file_path1)
            file_name2 = file_path2.name if isinstance(file_path2, Path) else str(file_path2)

            # 打印第一块的信息
            print(f" {pair_num} | {score:.4f} | {id1:<4} | {file_name1} ({b_type1} #{b_index1}) | {current_decision1:<10} | {preview1}")
            # 打印第二块的信息
            print(f"   |         | {id2:<4} | {file_name2} ({b_type2} #{b_index2}) | {current_decision2:<10} | {preview2}")
            print("-----|---------|------|-----------------------|------------|----------")

        print(f"\n[*] 共显示 {pair_num}/{total_pairs} 对语义相似块。")
        return True

    def review_semantic_duplicates_interactive(self) -> None:
        """
        提供交互式界面，让用户处理语义相似对。

        Process:
            1. 显示所有相似对
            2. 提供操作选项：
               - k1d2: 保留第一个，删除第二个
               - k2d1: 保留第二个，删除第一个
               - k: 保留两个块
               - d: 删除两个块
               - r: 重置决策
               - save: 保存当前决策
               - q: 退出
            3. 根据用户输入更新决策
            4. 支持批量操作
            5. 显示操作进度

        Note:
            - 每个操作后重新显示更新后的列表
            - 提供清晰的错误提示
            - 支持保存中间结果
            - 显示操作进度
        """
        logger.info("Starting interactive review of semantic duplicates...")

        # 初始显示列表
        if not self._display_semantic_duplicates_list():
            logger.info("No semantic duplicates to review.")
            print("\n--- 语义相似项处理完成 ---")
            return

        while True:
            # 显示操作选项
            print("\n操作选项:")
            print(" k1d2 <块ID> [块ID...] - 保留第一个，删除第二个")
            print(" k2d1 <块ID> [块ID...] - 保留第二个，删除第一个")
            print(" k <块ID> [块ID...]    - 保留两个块")
            print(" d <块ID> [块ID...]    - 删除两个块")
            print(" r <块ID> [块ID...]    - 重置决策")
            print(" save                  - 保存当前所有决策到文件")
            print(" q                     - 完成语义相似项处理并退出")

            action = input("请输入操作 (例如: k1d2 s1-1 d s2-1): ").lower().strip()
            logger.debug(f"User input for semantic review: '{action}'")

            # 标记是否需要重新显示列表 (默认为 True，除非输入无效)
            redisplay_list = True

            if action == 'q':
                logger.info("Quitting interactive semantic review.")
                break # 退出语义审核
            elif action == 'save':
                logger.info("User chose 'save' during semantic review.")
                # 调用主工具实例的保存方法
                if self.tool.save_decisions():
                    print(" [*] 决策已保存。您可以继续操作。")
                else:
                    print(" [!] 保存决策时遇到问题。")
                # 保存后不需要重新显示列表，继续循环显示选项
                redisplay_list = False
                continue

            parts = action.split()
            if not parts:
                redisplay_list = False # 无效输入
                continue # 忽略空输入

            command = parts[0]
            item_ids_input = parts[1:]

            if command not in ['k1d2', 'k2d1', 'k', 'd', 'r']:
                logger.warning(f"Invalid command: '{command}'")
                print("[错误] 无效的操作命令。")
                redisplay_list = False # 无效输入
                continue

            if not item_ids_input:
                logger.warning("Command requires at least one block ID.")
                print("[错误] 命令需要至少一个块 ID。")
                redisplay_list = False # 无效输入
                continue

            # 验证输入的块 ID 是否有效
            valid_ids = []
            invalid_ids = []
            for item_id in item_ids_input:
                # 检查 ID 是否在当前的映射中
                if item_id in self.semantic_id_to_key:
                    valid_ids.append(item_id)
                else:
                    invalid_ids.append(item_id)

            if invalid_ids:
                logger.warning(f"Invalid block IDs provided: {invalid_ids}")
                print(f"[错误] 无效的块 ID: {invalid_ids}")
                redisplay_list = False # 无效输入
                continue

            # 处理每个有效的块 ID
            processed_count = 0
            for item_id in valid_ids:
                key = self.semantic_id_to_key[item_id]
                
                # 根据命令设置决策值
                if command == 'k1d2':
                    # 保留第一个，删除第二个
                    pair_id = item_id.split('-')[0] # 例如 's1' from 's1-1'
                    other_id = f"{pair_id}-2" if item_id.endswith('1') else f"{pair_id}-1"
                    if other_id in self.semantic_id_to_key:
                        other_key = self.semantic_id_to_key[other_id]
                        self.tool.block_decisions[key] = 'keep'
                        self.tool.block_decisions[other_key] = 'delete'
                        processed_count += 2
                elif command == 'k2d1':
                    # 保留第二个，删除第一个
                    pair_id = item_id.split('-')[0]
                    other_id = f"{pair_id}-2" if item_id.endswith('1') else f"{pair_id}-1"
                    if other_id in self.semantic_id_to_key:
                        other_key = self.semantic_id_to_key[other_id]
                        self.tool.block_decisions[key] = 'delete'
                        self.tool.block_decisions[other_key] = 'keep'
                        processed_count += 2
                else:
                    # 处理 'k', 'd', 'r' 命令
                    decision_value = {
                        'k': 'keep',
                        'd': 'delete',
                        'r': 'undecided'
                    }[command]
                    self.tool.block_decisions[key] = decision_value
                    processed_count += 1

            print(f" [*] 已处理 {processed_count} 个块的决策。")

            # 如果需要重新显示列表
            if redisplay_list:
                self._display_semantic_duplicates_list()

        logger.info("Finished interactive review of semantic duplicates.")
        print("\n--- 语义相似项处理完成 ---")
