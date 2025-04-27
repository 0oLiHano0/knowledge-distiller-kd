# [DEPENDENCIES]
# 1. Python Standard Library: time, re, hashlib, pathlib, collections
# 2. 需要安装：sentence-transformers # 用于语义相似度计算
# 3. 同项目模块: utils, constants (使用绝对导入)

import time
import re
import hashlib
from pathlib import Path
import collections # 引入 collections

# 尝试导入 sentence_transformers，如果失败则禁用语义功能
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None # 定义一个占位符
    util = None # 定义一个占位符

# 使用绝对导入
import utils
import constants

logger = utils.logger # 使用 utils 中配置好的 logger

class SemanticAnalyzer:
    """处理语义相似性查找和交互式决策的类。"""

    def __init__(self, kd_tool_instance):
        """
        初始化语义分析器。

        Args:
            kd_tool_instance: 主 KDToolCLI 类的实例，用于访问共享数据和方法。
        """
        self.tool = kd_tool_instance # 保存主工具实例的引用
        self.semantic_model = None
        self.semantic_duplicates = [] # [(block_info1, block_info2, score), ...]
        self.semantic_id_to_key = {} # 用于交互式界面的映射 {display_id: decision_key}

        # 检查 sentence_transformers 是否可用
        if not SENTENCE_TRANSFORMERS_AVAILABLE and not self.tool.skip_semantic:
            logger.error("`sentence-transformers` library not found.")
            logger.error("Please install it: pip install sentence-transformers")
            logger.warning("Semantic analysis feature will be disabled.")
            self.tool.skip_semantic = True # 强制跳过语义分析

    def load_semantic_model(self):
        """加载 Sentence Transformer 模型。"""
        # 如果设置了跳过，或者模型已加载，或者库不可用，则直接返回
        if self.tool.skip_semantic or self.semantic_model or not SENTENCE_TRANSFORMERS_AVAILABLE:
            if self.tool.skip_semantic:
                logger.info("Semantic analysis is skipped by configuration.")
            elif self.semantic_model:
                 logger.info("Semantic model already loaded.")
            elif not SENTENCE_TRANSFORMERS_AVAILABLE:
                 logger.warning("Cannot load semantic model because `sentence-transformers` is not installed.")
            return

        try:
            model_name = constants.SEMANTIC_MODEL_NAME
            logger.info(f"Loading semantic model: {model_name} ... (This may take some time)")
            start_time = time.time()
            # 使用 SentenceTransformer 加载模型
            self.semantic_model = SentenceTransformer(model_name)
            end_time = time.time()
            logger.info(f"Semantic model loaded successfully. Time taken: {end_time - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to load semantic model: {constants.SEMANTIC_MODEL_NAME}", exc_info=True)
            logger.warning("Semantic deduplication feature will be unavailable.")
            self.semantic_model = None
            self.tool.skip_semantic = True # 加载失败，强制跳过

    def find_semantic_duplicates(self):
        """
        使用 Sentence Transformer 模型查找语义相似的块。
        优化：跳过对 MD5 精确重复块的向量计算和比较。
        """
        logger.info("Starting semantic similarity analysis (Optimized)...")
        # 检查是否应跳过或是否可以执行
        if self.tool.skip_semantic:
            logger.warning("Semantic analysis skipped as requested or due to previous errors.")
            self.semantic_duplicates = []
            return
        if self.semantic_model is None:
            logger.error("Semantic model not loaded, cannot perform semantic analysis.")
            self.semantic_duplicates = []
            return
        if len(self.tool.blocks_data) < 2:
            logger.info("Not enough blocks (<2) for semantic comparison.")
            self.semantic_duplicates = []
            return
        # 检查 MD5 分析是否已运行并有结果 (虽然本方法不再直接使用 duplicate_blocks, 但逻辑上依赖它先运行)
        if not hasattr(self.tool, 'md5_analyzer'):
             logger.error("MD5 analyzer instance not found. Cannot perform optimized semantic analysis.")
             return

        start_time = time.time()
        self.semantic_duplicates = [] # 清空旧结果

        # --- 优化步骤 1: 识别唯一块 ---
        unique_blocks_info = [] # 存储唯一块的信息 (原始 block_info)
        original_indices_map = [] # 存储 unique_blocks_info 中每个块在原始 blocks_data 中的索引
        md5_hash_to_representative_index = {} # {md5_hash: index_in_unique_blocks_info}
        unique_block_count = 0

        logger.debug("Identifying unique blocks based on MD5 for semantic analysis...")
        for original_idx, block_info in enumerate(self.tool.blocks_data):
             file_path, b_index, b_type, b_text = block_info
             try:
                 hex_dig = hashlib.md5(b_text.encode('utf-8')).hexdigest()
                 # 如果这个 MD5 哈希还没见过，就把它作为代表加入唯一列表
                 if hex_dig not in md5_hash_to_representative_index:
                      md5_hash_to_representative_index[hex_dig] = unique_block_count
                      unique_blocks_info.append(block_info)
                      original_indices_map.append(original_idx) # 记录其在原始列表中的索引
                      unique_block_count += 1
             except Exception as e:
                 logger.warning(f"Error calculating MD5 for block at original index {original_idx} during unique block identification: {e}. Skipping block.")
                 continue # 跳过无法计算 MD5 的块

        num_unique_blocks = len(unique_blocks_info)
        num_original_blocks = len(self.tool.blocks_data)
        num_skipped = num_original_blocks - num_unique_blocks
        logger.info(f"Identified {num_unique_blocks} unique content blocks for semantic analysis (skipped {num_skipped} exact duplicates).")

        if num_unique_blocks < 2:
            logger.info("Not enough unique blocks (<2) for semantic comparison after skipping duplicates.")
            return # 没有足够的唯一块进行比较

        try:
            # --- 优化步骤 2: 只对唯一块计算嵌入 ---
            unique_block_texts = [info[3] for info in unique_blocks_info]
            logger.info(f"Calculating vector embeddings for {num_unique_blocks} unique content blocks...")
            embeddings = self.semantic_model.encode(unique_block_texts, convert_to_tensor=True, show_progress_bar=True)
            embed_time = time.time()
            logger.info(f"Vector embedding calculation complete. Time taken: {embed_time - start_time:.2f} seconds")

            # --- 优化步骤 3: 只对唯一块进行语义搜索 ---
            logger.info(f"Finding unique block pairs with similarity > {self.tool.similarity_threshold}...")
            # top_k 应该相对于唯一块的数量
            hits = util.semantic_search(
                embeddings,
                embeddings,
                query_chunk_size=100,
                corpus_chunk_size=500,
                top_k=min(10, num_unique_blocks), # 限制 K 值不超过唯一块数量
                score_function=util.cos_sim
            )
            find_start_time = time.time()

            # --- 优化步骤 4: 处理结果并映射回原始块 ---
            processed_original_pairs = set() # 使用原始块索引跟踪已处理的对

            for unique_query_idx in range(len(hits)):
                # 获取查询块在原始 blocks_data 中的信息
                original_query_idx = original_indices_map[unique_query_idx]
                query_block_info = self.tool.blocks_data[original_query_idx]

                for hit in hits[unique_query_idx]:
                    unique_corpus_idx = hit['corpus_id']
                    score = hit['score']

                    # 跳过与自身的比较 (在唯一列表中的比较)
                    if unique_query_idx == unique_corpus_idx:
                        continue

                    # 跳过低于阈值的
                    if score < self.tool.similarity_threshold:
                        # 因为 semantic_search 返回按分数排序的结果，后续的也会低于阈值
                        break # 直接跳出内层循环

                    # 获取语料库块在原始 blocks_data 中的信息
                    original_corpus_idx = original_indices_map[unique_corpus_idx]
                    corpus_block_info = self.tool.blocks_data[original_corpus_idx]

                    # 创建原始块对的标识 (排序后的原始索引)
                    original_pair_identity = tuple(sorted((original_query_idx, original_corpus_idx)))

                    # 如果这对原始块之前没有处理过 (避免重复添加)
                    if original_pair_identity not in processed_original_pairs:
                        # 在优化逻辑下，这里找到的对不应该是 MD5 相同的
                        # (可以加断言 assert hashlib.md5(query_block_info[3].encode('utf-8')).hexdigest() != hashlib.md5(corpus_block_info[3].encode('utf-8')).hexdigest())
                        logger.debug(f"Found semantic duplicate pair (Original Indices): {original_query_idx} and {original_corpus_idx}, Score: {score:.4f}")
                        # 添加原始块的信息到结果列表
                        self.semantic_duplicates.append((query_block_info, corpus_block_info, score))
                        # 标记这对原始块为已处理
                        processed_original_pairs.add(original_pair_identity)

            find_end_time = time.time()
            logger.info(f"Semantic similarity search on unique blocks complete. Time taken: {find_end_time - find_start_time:.2f} seconds.")

            if self.semantic_duplicates:
                logger.info(f"Found {len(self.semantic_duplicates)} semantic duplicate pairs among unique blocks (threshold > {self.tool.similarity_threshold}).")
            else:
                logger.info(f"No semantic duplicate pairs found among unique blocks meeting the criteria.")

        except Exception as e:
            logger.error("Error during optimized semantic analysis process.", exc_info=True)
            self.semantic_duplicates = [] # 出错时清空结果

    # _display_semantic_duplicates_list 和 review_semantic_duplicates_interactive 方法保持不变
    # 因为它们处理的是最终的 self.semantic_duplicates 列表，这个列表现在只包含非 MD5 重复的相似对
    def _display_semantic_duplicates_list(self):
        """
        在控制台显示所有语义相似对，并为每个块分配唯一的显示 ID。
        更新 self.semantic_id_to_key 映射。
        返回: True 如果有相似项显示，False 如果没有。
        """
        print("\n--- 语义相似内容块列表 ---")
        if self.tool.skip_semantic:
            print("[*] 语义分析已被跳过。")
            return False
        # 现在 self.semantic_duplicates 只包含非 MD5 重复的相似项
        if not self.semantic_duplicates:
            print(f"[*] 未找到相似度 > {self.tool.similarity_threshold} 的语义相似项。")
            return False

        self.semantic_id_to_key.clear() # 清空旧的映射
        print("对号 | 相似度 | 块ID | 文件名 (类型 #索引) | 当前决策 | 内容预览")
        print("-----|---------|------|-----------------------|------------|----------")
        pair_num = 0

        # 遍历找到的语义相似对 (按分数降序排列显示)
        sorted_duplicates = sorted(self.semantic_duplicates, key=lambda x: x[2], reverse=True)

        for pair_idx, (info1, info2, score) in enumerate(sorted_duplicates):
            pair_num += 1
            # 为每对中的两个块创建显示 ID，例如 s1-1, s1-2
            id1 = f"s{pair_num}-1"
            id2 = f"s{pair_num}-2"

            file_path1, b_index1, b_type1, b_text1 = info1
            file_path2, b_index2, b_type2, b_text2 = info2

            # 创建决策键
            try:
                abs_path1_str = str(Path(file_path1).resolve())
                key1 = utils.create_decision_key(abs_path1_str, b_index1, b_type1)
            except Exception as e:
                 logger.warning(f"无法为块创建决策键 {file_path1}#{b_index1}: {e}. 跳过此对的显示。")
                 continue
            try:
                abs_path2_str = str(Path(file_path2).resolve())
                key2 = utils.create_decision_key(abs_path2_str, b_index2, b_type2)
            except Exception as e:
                 logger.warning(f"无法为块创建决策键 {file_path2}#{b_index2}: {e}. 跳过此对的显示。")
                 continue

            # 存储 显示ID -> 决策键 的映射
            self.semantic_id_to_key[id1] = key1
            self.semantic_id_to_key[id2] = key2

            # 获取当前决策状态
            decision1 = self.tool.block_decisions.get(key1, 'undecided')
            decision2 = self.tool.block_decisions.get(key2, 'undecided')

            # 生成内容预览
            preview1 = utils.display_block_preview(b_text1)
            preview2 = utils.display_block_preview(b_text2)

            # 获取文件名
            file_name1_str = file_path1.name if isinstance(file_path1, Path) else str(file_path1)
            file_name2_str = file_path2.name if isinstance(file_path2, Path) else str(file_path2)

            # 打印这对相似块的信息
            print(f" P{pair_num} | {score:<7.4f} | {id1:<4} | {file_name1_str} ({b_type1} #{b_index1}) | {decision1:<10} | {preview1}")
            print(f"     |         | {id2:<4} | {file_name2_str} ({b_type2} #{b_index2}) | {decision2:<10} | {preview2}")
            print("-----|---------|------|-----------------------|------------|----------") # 对分隔线
        return True

    def review_semantic_duplicates_interactive(self):
        """
        提供交互式界面，让用户处理语义相似对 (列表模式)。
        用户可以标记保留 (keep)、删除 (delete) 或重置 (undecided)。
        也提供了针对整对操作的快捷方式。
        """
        logger.info("Starting interactive review of semantic duplicates (list mode)...")

        # 初始显示列表
        if not self._display_semantic_duplicates_list():
            logger.info("No semantic duplicates to review.")
            print("\n--- 语义相似项处理完成 ---")
            return # 没有可审核项，直接返回

        while True:
            # 显示选项
            print("\n操作选项:")
            print(" k <块ID> [块ID...] - 标记指定块为 '保留 (keep)'")
            print(" d <块ID> [块ID...] - 标记指定块为 '删除 (delete)'")
            print(" r <块ID> [块ID...] - 重置指定块为 'undecided'")
            print(" keep <对号>        - 保留指定对号中的两个块 (例如: keep P1)")
            print(" k1d2 <对号>        - 保留对号中第1个块, 删除第2个 (例如: k1d2 P2)")
            print(" k2d1 <对号>        - 保留对号中第2个块, 删除第1个 (例如: k2d1 P3)")
            print(" save               - 保存当前所有决策到文件")
            print(" q                  - 完成语义处理并退出")

            action = input("请输入操作 (例如: k s1-1 d s1-2 或 k1d2 P1): ").lower().strip()
            logger.debug(f"User input for semantic list review: '{action}'")

            # 标记是否需要重新显示列表 (默认为 True，除非输入无效)
            redisplay_list = True

            if action == 'q':
                logger.info("Quitting interactive semantic review (list mode).")
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
            args = parts[1:]

            if command not in ['k', 'd', 'r', 'keep', 'k1d2', 'k2d1']:
                logger.warning(f"Invalid command: '{command}'")
                print("[错误] 无效的操作命令。")
                redisplay_list = False # 无效输入
                continue

            if not args:
                logger.warning("Command requires arguments (block IDs or pair numbers).")
                print("[错误] 命令需要参数 (块 ID 或对号)。")
                redisplay_list = False # 无效输入
                continue

            # --- 处理 'k', 'd', 'r' (针对单个块) ---
            if command in ['k', 'd', 'r']:
                item_ids_input = args
                valid_ids = []
                invalid_ids = []
                # 验证输入的块 ID
                for item_id in item_ids_input:
                    if item_id in self.semantic_id_to_key:
                        valid_ids.append(item_id)
                    else:
                        invalid_ids.append(item_id)

                if invalid_ids:
                    logger.warning(f"Invalid block IDs provided: {invalid_ids}")
                    print(f"[错误] 无效的块 ID: {invalid_ids}")
                    redisplay_list = False # 无效输入
                    continue

                # 应用决策到每个有效的块 ID
                for item_id in valid_ids:
                    key = self.semantic_id_to_key[item_id]
                    if command == 'k':
                        self.tool.block_decisions[key] = 'keep'
                        logger.info(f"Marked {item_id} ({key}) as 'keep'.")
                        print(f" 已标记 {item_id} 为 '保留'")
                    elif command == 'd':
                        self.tool.block_decisions[key] = 'delete'
                        logger.info(f"Marked {item_id} ({key}) as 'delete'.")
                        print(f" 已标记 {item_id} 为 '删除'")
                    elif command == 'r':
                        self.tool.block_decisions[key] = 'undecided'
                        logger.info(f"Reset {item_id} ({key}) to 'undecided'.")
                        print(f" 已重置 {item_id} 为 'undecided'")
                # 有效处理后，会在下一次循环开始时重新显示列表

            # --- 处理 'keep', 'k1d2', 'k2d1' (针对整对) ---
            elif command in ['keep', 'k1d2', 'k2d1']:
                if len(args) != 1:
                    print(f"[错误] 命令 '{command}' 需要且仅需要一个对号参数 (例如: P1)。")
                    redisplay_list = False # 无效输入
                    continue

                pair_arg = args[0]
                # 使用正则表达式匹配对号格式 P<数字> (不区分大小写)
                match = re.match(r"p(\d+)", pair_arg, re.IGNORECASE)
                if not match:
                    print(f"[错误] 无效的对号格式: '{pair_arg}'。请使用 P<数字> 格式 (例如: P1)。")
                    redisplay_list = False # 无效输入
                    continue

                pair_num = int(match.group(1))
                # 构建这对块的显示 ID
                id1 = f"s{pair_num}-1"
                id2 = f"s{pair_num}-2"

                # 检查这对 ID 是否存在于映射中
                if id1 not in self.semantic_id_to_key or id2 not in self.semantic_id_to_key:
                    print(f"[错误] 找不到对号 P{pair_num} 对应的块 ID。可能这对块已被处理或不存在。")
                    redisplay_list = False # 无效输入
                    continue

                # 获取这对块的决策键
                key1 = self.semantic_id_to_key[id1]
                key2 = self.semantic_id_to_key[id2]

                # 根据命令应用决策
                if command == 'keep':
                    self.tool.block_decisions[key1] = 'keep'
                    self.tool.block_decisions[key2] = 'keep'
                    logger.info(f"Pair P{pair_num}: Marked both {id1} and {id2} as 'keep'.")
                    print(f" 已标记对 P{pair_num} 中的两个块 ({id1}, {id2}) 为 '保留'")
                elif command == 'k1d2':
                    self.tool.block_decisions[key1] = 'keep'
                    self.tool.block_decisions[key2] = 'delete'
                    logger.info(f"Pair P{pair_num}: Marked {id1} as 'keep', {id2} as 'delete'.")
                    print(f" 已标记对 P{pair_num} 中的块 {id1} 为 '保留', {id2} 为 '删除'")
                elif command == 'k2d1':
                    self.tool.block_decisions[key1] = 'delete'
                    self.tool.block_decisions[key2] = 'keep'
                    logger.info(f"Pair P{pair_num}: Marked {id1} as 'delete', {id2} as 'keep'.")
                    print(f" 已标记对 P{pair_num} 中的块 {id1} 为 '删除', {id2} 为 '保留'")
                # 有效处理后，会在下一次循环开始时重新显示列表

            # 只有在需要时才重新显示列表
            if redisplay_list:
                if not self._display_semantic_duplicates_list():
                    # 如果处理完最后一个相似对，列表为空，则退出
                    logger.info("No remaining semantic duplicates to review.")
                    break # 退出循环

        logger.info("Finished interactive review of semantic duplicates (list mode).")
        print("\n--- 语义相似项处理完成 ---")
