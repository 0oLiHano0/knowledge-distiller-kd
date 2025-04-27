import sys
import os
from pathlib import Path
import hashlib
import collections
import mistune
import json
from sentence_transformers import SentenceTransformer, util # 确保已安装
import time
import logging # 引入日志模块
import re # 用于解析输入

# --- 配置日志记录 ---
# 创建一个 logger
logger = logging.getLogger('KDToolLogger')
logger.setLevel(logging.INFO) # 默认记录 INFO 及以上级别的日志
# 创建一个控制台处理器 (handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # 控制台也显示 INFO 及以上
# 创建日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
# 将处理器添加到 logger
if not logger.handlers: # 防止重复添加 handler
    logger.addHandler(console_handler)

# --- 常量定义 ---
SEMANTIC_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
SIMILARITY_THRESHOLD = 0.85 # 语义相似度阈值
DECISION_KEY_SEPARATOR = "::"
DEFAULT_OUTPUT_SUFFIX = "_deduped"

# --- 硬编码路径 (开发阶段) ---
HARDCODED_INPUT_PATH = "/Users/hansen/Documents/Knowledge Distillation/Test_SourceFiles"
HARDCODED_DECISION_PATH = "/Users/hansen/Documents/Knowledge Distillation/Test_other/kd_decisions.json"
HARDCODED_OUTPUT_DIR_PATH = "/Users/hansen/Documents/Knowledge Distillation/Test_other/kd_results"
# ---------------------------

# --- 辅助函数 ---
def create_decision_key(file_path, block_index, block_type):
    """为块创建唯一的字符串键。"""
    file_path_str = str(file_path)
    try: index_int = int(block_index)
    except ValueError: index_int = block_index
    key = f"{file_path_str}{DECISION_KEY_SEPARATOR}{index_int}{DECISION_KEY_SEPARATOR}{str(block_type)}"
    return key

def parse_decision_key(key):
    """从键解析出文件路径、索引和类型。"""
    try:
        parts = key.split(DECISION_KEY_SEPARATOR)
        if len(parts) == 3:
            path_str, index_str, type_str = parts
            try: index_val = int(index_str)
            except ValueError: index_val = index_str
            return path_str, index_val, type_str
        else:
            type_str = parts[-1]; index_str = parts[-2]; path_str = DECISION_KEY_SEPARATOR.join(parts[:-2])
            try: index_val = int(index_str)
            except ValueError: index_val = index_str
            return path_str, index_val, type_str
    except Exception as e:
        logger.error(f"无法解析决策键: {key} - {e}", exc_info=True)
        return None, None, None

def extract_text_from_children(children):
    """(来自原代码) 辅助函数，用于从 mistune 令牌子项中递归提取文本。"""
    text = "";
    if children is None: return ""
    for child in children:
        child_type = child.get('type')
        if child_type == 'text': text += child.get('raw', '')
        elif child_type == 'codespan': text += child.get('raw', '')
        elif child_type in ['link', 'image', 'emphasis', 'strong', 'strikethrough']: text += extract_text_from_children(child.get('children', []))
        elif child_type == 'softbreak' or child_type == 'linebreak': text += ' '
        elif child_type == 'inline_html': pass
    return text

def display_block_preview(text, max_len=80):
    """生成用于控制台显示的块预览。"""
    preview = text.replace('\n', ' ').strip()
    if len(preview) > max_len: return preview[:max_len-3] + "..."
    return preview

# --- 核心逻辑类 ---
class KDToolCLI:
    def __init__(self, skip_semantic=False, similarity_threshold=SIMILARITY_THRESHOLD):
        logger.info("Initializing KDToolCLI instance...")
        self.input_dir = None
        self.output_dir = Path(HARDCODED_OUTPUT_DIR_PATH)
        self.decision_file = Path(HARDCODED_DECISION_PATH)
        self.skip_semantic = skip_semantic
        self.similarity_threshold = similarity_threshold
        logger.info(f"Settings: skip_semantic={self.skip_semantic}, similarity_threshold={self.similarity_threshold}")
        logger.info(f"Hardcoded Decision Path: {self.decision_file}")
        logger.info(f"Hardcoded Output Dir: {self.output_dir}")
        self.markdown_files_content = {}
        self.blocks_data = []
        self.duplicate_blocks = {}
        self.semantic_duplicates = []
        self.block_decisions = {}
        self.semantic_model = None
        self.md5_id_to_key = {}
        self.semantic_id_to_key = {}
        logger.info("KDToolCLI instance initialized.")

    # --- set_input_directory, _load_semantic_model, run_analysis, _read_files, _parse_markdown, _initialize_decisions, _find_md5_duplicates, _find_semantic_duplicates 保持不变 ---
    def set_input_directory(self, input_dir_path):
        """设置输入目录并进行基本验证。"""
        logger.info(f"Attempting to set input directory: {input_dir_path}")
        path = Path(input_dir_path)
        if not path.is_dir():
            logger.error(f"Invalid directory path provided: {input_dir_path}")
            return False
        self.input_dir = path
        logger.info(f"Input directory set successfully: {self.input_dir}")
        try:
            self.decision_file.parent.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Ensured hardcoded decision/output parent directories exist.")
        except Exception as e:
            logger.warning(f"Could not create parent directories for hardcoded paths: {e}", exc_info=False)
        if not self.skip_semantic and self.semantic_model is None:
            self._load_semantic_model()
        return True

    def _load_semantic_model(self):
        """加载 Sentence Transformer 模型。"""
        if self.semantic_model or self.skip_semantic: logger.info("Semantic model already loaded or skipped."); return
        try:
            logger.info(f"Loading semantic model: {SEMANTIC_MODEL_NAME} ...")
            start_time = time.time(); self.semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME); end_time = time.time()
            logger.info(f"Semantic model loaded successfully. Time taken: {end_time - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to load semantic model: {SEMANTIC_MODEL_NAME}", exc_info=True)
            logger.warning("Semantic deduplication feature will be unavailable."); self.semantic_model = None; self.skip_semantic = True

    def run_analysis(self):
        """执行完整的分析流程。"""
        if not self.input_dir: logger.error("Analysis aborted: Input directory not set."); print("[错误] 请先设置有效的输入文件夹。"); return False
        logger.info(f"--- Starting analysis for folder: {self.input_dir} ---"); logger.info("Resetting internal state for new analysis...")
        self.markdown_files_content = {}; self.blocks_data = []; self.duplicate_blocks = {}; self.semantic_duplicates = []; self.block_decisions = {}
        self.md5_id_to_key = {}; self.semantic_id_to_key = {}; logger.info("Internal state reset.")
        logger.info("Step 1: Reading files...")
        if not self._read_files(): logger.error("Analysis stopped: File reading failed."); return False
        logger.info("Step 1: File reading completed.")
        logger.info("Step 2: Parsing Markdown...")
        if not self._parse_markdown(): logger.error("Analysis stopped: Markdown parsing failed."); return False
        logger.info("Step 2: Markdown parsing completed.")
        logger.info("Step 3: Initializing block decisions...")
        self._initialize_decisions(); logger.info("Step 3: Block decisions initialized.")
        logger.info("Step 4: Performing MD5 deduplication...")
        self._find_md5_duplicates(); logger.info("Step 4: MD5 deduplication completed.")
        if not self.skip_semantic:
            logger.info("Step 5: Performing semantic deduplication...")
            self._find_semantic_duplicates(); logger.info("Step 5: Semantic deduplication completed.")
        else: logger.info("Step 5: Semantic deduplication skipped as requested.")
        logger.info("--- Analysis finished ---"); return True

    def _read_files(self):
        """查找并读取目录中的 Markdown 文件。"""
        logger.info(f"Scanning directory for Markdown files: {self.input_dir}")
        try:
            markdown_files = list(self.input_dir.glob('*.md')); logger.info(f"Found {len(markdown_files)} potential Markdown files.")
            files_read_count = 0
            for md_file in markdown_files:
                logger.debug(f"Attempting to read: {md_file.name}")
                try: content = md_file.read_text(encoding='utf-8'); self.markdown_files_content[md_file] = content; files_read_count += 1; logger.debug(f"Successfully read: {md_file.name}")
                except Exception as e: logger.error(f"Error reading file: {md_file.name} - {e}", exc_info=False)
            logger.info(f"Content reading complete: Successfully read {files_read_count} / {len(markdown_files)} files.")
            if files_read_count == 0 and markdown_files: logger.warning("No files were successfully read..."); print("[错误] 未能读取任何文件。"); return False
            elif files_read_count == 0 and not markdown_files: logger.warning("No Markdown files found..."); print("[错误] 未找到任何 Markdown 文件。"); return False
            return True
        except Exception as e: logger.error(f"Error finding files in directory: {self.input_dir}", exc_info=True); return False

    def _parse_markdown(self):
        """解析 Markdown 文件内容并提取非标题块。"""
        logger.info("Parsing Markdown content and extracting blocks (excluding headings)...")
        self.blocks_data = []; markdown_parser = mistune.create_markdown(renderer=None, plugins=['strikethrough', 'footnotes', 'table', 'url', 'task_lists'])
        total_files = len(self.markdown_files_content); parsed_files_count = 0
        try:
            for md_file_path, content in self.markdown_files_content.items():
                logger.debug(f"Parsing file: {md_file_path.name}"); file_block_count = 0
                try:
                    block_tokens = markdown_parser(content)
                    if not block_tokens: logger.debug(f"No tokens found for: {md_file_path.name}"); continue
                    for index, token in enumerate(block_tokens):
                        block_type = token.get('type', 'unknown'); block_text = ""; blocks_to_add = []
                        if block_type == 'heading': continue
                        elif block_type == 'paragraph': block_text = extract_text_from_children(token.get('children', []))
                        elif block_type == 'block_code': block_text = token.get('raw') if token.get('raw') is not None else token.get('text', '')
                        elif block_type == 'block_quote': block_text = extract_text_from_children(token.get('children', []))
                        elif block_type == 'list':
                            list_items = token.get('children', [])
                            for item_index, item_token in enumerate(list_items):
                                if item_token.get('type') == 'list_item':
                                    item_text = extract_text_from_children(item_token.get('children', [])); cleaned_item_text = item_text.strip()
                                    if cleaned_item_text: list_item_block_index = f"{index}_{item_index}"; blocks_to_add.append((md_file_path, list_item_block_index, 'list_item', cleaned_item_text))
                        elif block_type == 'thematic_break': continue
                        elif block_type == 'block_html': block_text = token.get('raw', '')
                        elif block_type == 'table': header_text = extract_text_from_children(token.get('header', [])); body_text = extract_text_from_children(token.get('children', [])); block_text = f"Table Header: {header_text}\nTable Body:\n{body_text}"
                        else: block_text = token.get('raw') if token.get('raw') is not None else token.get('text', '');
                        if not block_text and 'children' in token: block_text = extract_text_from_children(token.get('children', []))
                        cleaned_text = block_text.strip() if block_type not in ['block_code', 'block_html', 'list'] else block_text
                        if cleaned_text and block_type != 'list': blocks_to_add.append((md_file_path, index, block_type, cleaned_text))
                        if blocks_to_add: self.blocks_data.extend(blocks_to_add); file_block_count += len(blocks_to_add); logger.debug(f"Added {len(blocks_to_add)} block(s) from token index {index} in {md_file_path.name}")
                    parsed_files_count += 1; logger.debug(f"Finished parsing {md_file_path.name}, found {file_block_count} valid blocks.")
                except Exception as parse_err: logger.warning(f"Error parsing file {md_file_path.name}: {parse_err}", exc_info=False); continue
            total_blocks = len(self.blocks_data); logger.info(f"Markdown parsing complete: Processed {parsed_files_count}/{total_files} files. Extracted {total_blocks} total valid content blocks.")
            if total_blocks == 0: logger.warning("No valid content blocks were extracted..."); print("[警告] 未能从任何文件中提取有效的内容块。");
            return True
        except Exception as e: logger.error("Unexpected error during Markdown parsing process.", exc_info=True); return False

    def _initialize_decisions(self):
        """为所有解析出的块初始化决策状态为 'undecided'。"""
        logger.info("Initializing decision state for all extracted blocks...")
        self.block_decisions = {}; count = 0
        for block_info in self.blocks_data: key = create_decision_key(block_info[0], block_info[1], block_info[2]); self.block_decisions[key] = 'undecided'; count += 1
        logger.info(f"Initialized decisions for {count} blocks to 'undecided'.")

    def _find_md5_duplicates(self):
        """计算 MD5 哈希并查找完全重复的块。"""
        logger.info("Calculating MD5 hashes and finding exact duplicate blocks...")
        hash_map = collections.defaultdict(list); start_time = time.time(); calculated_hashes = 0
        try:
            for block_info in self.blocks_data:
                try: text_to_hash = block_info[3]; hash_object = hashlib.md5(text_to_hash.encode('utf-8')); hex_dig = hash_object.hexdigest(); hash_map[hex_dig].append(block_info); calculated_hashes += 1
                except Exception as hash_err: logger.warning(f"Error hashing block: Index {block_info[1]} in {block_info[0].name} - {hash_err}", exc_info=False); continue
            logger.debug(f"Calculated {calculated_hashes} MD5 hashes.")
            self.duplicate_blocks = {h: b for h, b in hash_map.items() if len(b) > 1}; end_time = time.time()
            num_groups = len(self.duplicate_blocks); num_duplicates = sum(len(b) for b in self.duplicate_blocks.values()); logger.info(f"MD5 analysis complete. Time taken: {end_time - start_time:.2f} seconds.")
            if num_groups > 0: logger.info(f"Found {num_groups} groups of exact duplicates, involving {num_duplicates} block instances.")
            else: logger.info("No exact duplicate content blocks found.")
        except Exception as e: logger.error("Error during MD5 calculation or duplicate finding.", exc_info=True); self.duplicate_blocks = {}

    def _find_semantic_duplicates(self):
        """使用 Sentence Transformer 模型查找语义相似的块。"""
        logger.info("Starting semantic similarity analysis...")
        if self.semantic_model is None: logger.error("Semantic model not loaded, skipping semantic analysis."); return
        if len(self.blocks_data) < 2: logger.info("Not enough blocks (<2) for semantic comparison."); return
        start_time = time.time()
        try:
            block_texts = [info[3] for info in self.blocks_data]; logger.info(f"Calculating vector embeddings for {len(block_texts)} content blocks...")
            embeddings = self.semantic_model.encode(block_texts, convert_to_tensor=True, show_progress_bar=True); embed_time = time.time()
            logger.info(f"Vector embedding calculation complete. Time taken: {embed_time - start_time:.2f} seconds")
            logger.info(f"Finding block pairs with similarity > {self.similarity_threshold}...")
            hits = util.semantic_search(embeddings, embeddings, query_chunk_size=100, corpus_chunk_size=500, top_k=5, score_function=util.cos_sim); find_start_time = time.time()
            self.semantic_duplicates = []; processed_pairs = set(); md5_hashes = {}; md5_calc_errors = 0
            logger.debug("Calculating MD5 hashes for semantic duplicate filtering...")
            for info in self.blocks_data:
                try: md5_hashes[create_decision_key(info[0], info[1], info[2])] = hashlib.md5(info[3].encode('utf-8')).hexdigest()
                except Exception: md5_calc_errors += 1
            if md5_calc_errors > 0: logger.warning(f"Could not calculate MD5 for {md5_calc_errors} blocks during semantic filtering.")
            md5_skipped_count = 0; potential_pairs_checked = 0
            for query_idx in range(len(hits)):
                query_block_info = self.blocks_data[query_idx]; query_key = create_decision_key(query_block_info[0], query_block_info[1], query_block_info[2]); query_hash = md5_hashes.get(query_key)
                for hit in hits[query_idx]:
                    corpus_idx = hit['corpus_id']; score = hit['score']; potential_pairs_checked += 1
                    if query_idx == corpus_idx: continue
                    if score < self.similarity_threshold: continue
                    corpus_block_info = self.blocks_data[corpus_idx]; corpus_key = create_decision_key(corpus_block_info[0], corpus_block_info[1], corpus_block_info[2]); corpus_hash = md5_hashes.get(corpus_key)
                    if query_hash is not None and corpus_hash is not None and query_hash == corpus_hash:
                        pair_identity_md5 = tuple(sorted((query_idx, corpus_idx)))
                        if pair_identity_md5 not in processed_pairs: md5_skipped_count += 1; processed_pairs.add(pair_identity_md5); logger.debug(f"Skipped MD5 identical pair: {query_idx} and {corpus_idx}")
                        continue
                    pair_identity = tuple(sorted((query_idx, corpus_idx)))
                    if pair_identity not in processed_pairs: logger.debug(f"Found semantic duplicate pair: {query_idx} and {corpus_idx}, Score: {score:.4f}"); self.semantic_duplicates.append((query_block_info, corpus_block_info, score)); processed_pairs.add(pair_identity)
            find_end_time = time.time(); logger.info(f"Semantic similarity search complete. Time taken: {find_end_time - find_start_time:.2f} seconds. Checked {potential_pairs_checked} potential pairs.")
            if md5_skipped_count > 0: logger.info(f"Skipped {md5_skipped_count} pairs during semantic analysis because they were exact MD5 duplicates.")
            if self.semantic_duplicates: logger.info(f"Found {len(self.semantic_duplicates)} semantic duplicate pairs (excluding exact duplicates).")
            else: logger.info("No semantic duplicate pairs found meeting the criteria.")
        except Exception as e: logger.error("Error during semantic analysis process.", exc_info=True); self.semantic_duplicates = []

    # --- 新的交互式处理方法 (列表模式) ---
    def _display_md5_duplicates_list(self):
        """Displays all MD5 duplicate groups with unique IDs."""
        print("\n--- 精确重复内容块 (MD5) 列表 ---")
        if not self.duplicate_blocks:
            print("[*] 未找到精确重复项。")
            return False

        self.md5_id_to_key.clear() # 清空旧映射
        group_num = 0
        print("组号 | 块ID | 文件名 (类型 #索引) | 当前决策 | 内容预览")
        print("-----|------|-----------------------|------------|----------")
        for md5_hash, block_list in self.duplicate_blocks.items():
            group_num += 1
            group_preview_shown = False
            for i, block_info in enumerate(block_list):
                item_id = f"g{group_num}-{i+1}"
                file_path, b_index, b_type, b_text = block_info
                key = create_decision_key(file_path, b_index, b_type)
                self.md5_id_to_key[item_id] = key # 存储 ID 到 key 的映射
                current_decision = self.block_decisions.get(key, 'undecided')
                preview_text = display_block_preview(b_text)
                display_preview = preview_text if not group_preview_shown else ""
                group_preview_shown = True
                file_name_str = file_path.name if isinstance(file_path, Path) else str(file_path)
                print(f" G{group_num} | {item_id:<4} | {file_name_str} ({b_type} #{b_index}) | {current_decision:<10} | {display_preview}")
            print("-----|------|-----------------------|------------|----------") # 组分隔线
        return True

    def review_md5_duplicates_interactive(self):
        """交互处理 MD5 重复组 (列表模式)。"""
        logger.info("Starting interactive review of MD5 duplicates (list mode)...")
        while True:
            has_duplicates = self._display_md5_duplicates_list()
            if not has_duplicates:
                break

            print("\n操作选项:")
            print("  k <块ID> [块ID...]  - 标记指定块为 '保留 (keep)'")
            print("  d <块ID> [块ID...]  - 标记指定块为 '删除 (delete)'")
            print("  r <块ID> [块ID...]  - 重置指定块为 'undecided'")
            print("  a <块ID>           - 保留指定块，删除同组其他块 ('auto-delete')")
            print("  save               - 保存当前所有决策到文件") # 新增
            print("  q                  - 完成 MD5 处理并退出")
            action = input("请输入操作 (例如: k g1-1 d g1-2 g2-1): ").lower().strip()
            logger.debug(f"User input for MD5 list review: '{action}'")

            if action == 'q':
                logger.info("Quitting interactive MD5 review (list mode).")
                break
            elif action == 'save': # 新增保存选项
                logger.info("User chose 'save' during MD5 review.")
                self.save_decisions() # 调用保存函数
                print("  [*] 决策已保存。您可以继续操作。")
                continue # 继续 MD5 审核循环

            parts = action.split()
            if not parts: continue
            command = parts[0]
            item_ids = parts[1:]

            if command not in ['k', 'd', 'r', 'a']:
                logger.warning(f"Invalid command: '{command}'")
                print("[错误] 无效的操作命令。")
                continue

            if not item_ids:
                logger.warning("Command requires at least one block ID.")
                print("[错误] 命令需要至少一个块 ID。")
                continue

            valid_ids = []; invalid_ids = []
            for item_id in item_ids:
                item_id_lower = item_id.lower(); found = False
                for map_id in self.md5_id_to_key:
                    if map_id.lower() == item_id_lower: valid_ids.append(map_id); found = True; break
                if not found: invalid_ids.append(item_id)
            if invalid_ids: logger.warning(f"Invalid block IDs provided: {invalid_ids}"); print(f"[错误] 无效的块 ID: {invalid_ids}"); continue

            if command == 'a':
                if len(valid_ids) != 1: logger.warning("Command 'a' requires exactly one block ID."); print("[错误] 命令 'a' 需要且仅需要一个块 ID。"); continue
                keep_id = valid_ids[0]; keep_key = self.md5_id_to_key[keep_id]
                group_id_prefix = keep_id.split('-')[0]; group_member_keys = []; group_member_ids = []
                for item_id_in_map, key_in_map in self.md5_id_to_key.items():
                    if item_id_in_map.startswith(group_id_prefix): group_member_keys.append(key_in_map); group_member_ids.append(item_id_in_map)
                if not group_member_keys: logger.error(f"Internal error: Could not find group members for ID {keep_id}"); continue
                logger.info(f"Processing 'a' command: Keep {keep_id} ({keep_key}), delete others in group {group_id_prefix}")
                self.block_decisions[keep_key] = 'keep'; print(f"  已标记 {keep_id} 为 '保留'")
                for i, member_key in enumerate(group_member_keys):
                    member_id = group_member_ids[i]
                    if member_key != keep_key: self.block_decisions[member_key] = 'delete'; print(f"  已标记 {member_id} 为 '删除'"); logger.info(f"  Marked {member_id} ({member_key}) as 'delete'")
                continue

            for item_id in valid_ids:
                key = self.md5_id_to_key[item_id]
                if command == 'k': self.block_decisions[key] = 'keep'; logger.info(f"Marked {item_id} ({key}) as 'keep'."); print(f"  已标记 {item_id} 为 '保留'")
                elif command == 'r': self.block_decisions[key] = 'undecided'; logger.info(f"Reset {item_id} ({key}) to 'undecided'."); print(f"  已重置 {item_id} 为 'undecided'")
                elif command == 'd':
                    group_id_prefix = item_id.split('-')[0]; group_member_keys = [k for id_, k in self.md5_id_to_key.items() if id_.startswith(group_id_prefix)]
                    non_delete_count = sum(1 for mk in group_member_keys if self.block_decisions.get(mk, 'undecided') != 'delete'); current_decision = self.block_decisions.get(key, 'undecided'); is_last_one = (non_delete_count <= 1 and current_decision != 'delete')
                    if is_last_one: logger.warning(f"Skipped deleting {item_id} ({key}) as last non-delete."); print(f"  [跳过] {item_id} 是其组中最后一个非删除项，无法删除。")
                    else: self.block_decisions[key] = 'delete'; logger.info(f"Marked {item_id} ({key}) as 'delete'."); print(f"  已标记 {item_id} 为 '删除'")

        logger.info("Finished interactive review of MD5 duplicates (list mode).")
        print("\n--- MD5 重复项处理完成 ---")


    def _display_semantic_duplicates_list(self):
        """Displays all semantic duplicate pairs with unique IDs."""
        print("\n--- 语义相似内容块列表 ---")
        if self.skip_semantic: print("[*] 语义分析已被跳过。"); return False
        if not self.semantic_duplicates: print("[*] 未找到语义相似项。"); return False

        self.semantic_id_to_key.clear() # 清空旧映射
        print("对号 | 相似度 | 块ID | 文件名 (类型 #索引) | 当前决策 | 内容预览")
        print("-----|---------|------|-----------------------|------------|----------")
        pair_num = 0
        for pair_idx, (info1, info2, score) in enumerate(self.semantic_duplicates):
            pair_num += 1
            id1 = f"s{pair_num}-1"; id2 = f"s{pair_num}-2"
            file_path1, b_index1, b_type1, b_text1 = info1; file_path2, b_index2, b_type2, b_text2 = info2
            key1 = create_decision_key(file_path1, b_index1, b_type1); key2 = create_decision_key(file_path2, b_index2, b_type2)
            self.semantic_id_to_key[id1] = key1; self.semantic_id_to_key[id2] = key2
            decision1 = self.block_decisions.get(key1, 'undecided'); decision2 = self.block_decisions.get(key2, 'undecided')
            preview1 = display_block_preview(b_text1); preview2 = display_block_preview(b_text2)
            file_name1_str = file_path1.name if isinstance(file_path1, Path) else str(file_path1)
            file_name2_str = file_path2.name if isinstance(file_path2, Path) else str(file_path2)
            print(f" P{pair_num} | {score:<7.4f} | {id1:<4} | {file_name1_str} ({b_type1} #{b_index1}) | {decision1:<10} | {preview1}")
            print(f"     |         | {id2:<4} | {file_name2_str} ({b_type2} #{b_index2}) | {decision2:<10} | {preview2}")
            print("-----|---------|------|-----------------------|------------|----------") # 对分隔线
        return True

    def review_semantic_duplicates_interactive(self):
        """交互处理语义相似对 (列表模式)。"""
        logger.info("Starting interactive review of semantic duplicates (list mode)...")
        while True:
            has_duplicates = self._display_semantic_duplicates_list()
            if not has_duplicates:
                break

            print("\n操作选项:")
            print("  k <块ID> [块ID...] - 标记指定块为 '保留 (keep)'")
            print("  d <块ID> [块ID...] - 标记指定块为 '删除 (delete)'")
            print("  r <块ID> [块ID...] - 重置指定块为 'undecided'")
            print("  keep <对号>        - 保留指定对号中的两个块 (例如: keep P1)")
            print("  k1d2 <对号>        - 保留对号中第1个块, 删除第2个 (例如: k1d2 P2)")
            print("  k2d1 <对号>        - 保留对号中第2个块, 删除第1个 (例如: k2d1 P3)")
            print("  save               - 保存当前所有决策到文件") # 新增
            print("  q                  - 完成语义处理并退出")
            action = input("请输入操作 (例如: k s1-1 d s1-2 或 k1d2 P1): ").lower().strip()
            logger.debug(f"User input for semantic list review: '{action}'")

            if action == 'q':
                logger.info("Quitting interactive semantic review (list mode).")
                break
            elif action == 'save': # 新增保存选项
                logger.info("User chose 'save' during semantic review.")
                self.save_decisions() # 调用保存函数
                print("  [*] 决策已保存。您可以继续操作。")
                continue # 继续语义审核循环

            parts = action.split()
            if not parts: continue
            command = parts[0]
            args = parts[1:]

            if command not in ['k', 'd', 'r', 'keep', 'k1d2', 'k2d1']:
                logger.warning(f"Invalid command: '{command}'")
                print("[错误] 无效的操作命令。")
                continue

            if not args:
                logger.warning("Command requires arguments (block IDs or pair numbers).")
                print("[错误] 命令需要参数 (块 ID 或对号)。")
                continue

            if command in ['k', 'd', 'r']:
                item_ids = args; valid_ids = []; invalid_ids = []
                for item_id in item_ids:
                    item_id_lower = item_id.lower(); found = False
                    for map_id in self.semantic_id_to_key:
                        if map_id.lower() == item_id_lower: valid_ids.append(map_id); found = True; break
                    if not found: invalid_ids.append(item_id)
                if invalid_ids: logger.warning(f"Invalid block IDs provided: {invalid_ids}"); print(f"[错误] 无效的块 ID: {invalid_ids}"); continue
                for item_id in valid_ids:
                    key = self.semantic_id_to_key[item_id]
                    if command == 'k': self.block_decisions[key] = 'keep'; logger.info(f"Marked {item_id} ({key}) as 'keep'."); print(f"  已标记 {item_id} 为 '保留'")
                    elif command == 'd': self.block_decisions[key] = 'delete'; logger.info(f"Marked {item_id} ({key}) as 'delete'."); print(f"  已标记 {item_id} 为 '删除'")
                    elif command == 'r': self.block_decisions[key] = 'undecided'; logger.info(f"Reset {item_id} ({key}) to 'undecided'."); print(f"  已重置 {item_id} 为 'undecided'")

            elif command in ['keep', 'k1d2', 'k2d1']:
                if len(args) != 1: print(f"[错误] 命令 '{command}' 需要且仅需要一个对号参数 (例如: P1)。"); continue
                pair_arg = args[0]; match = re.match(r"p(\d+)", pair_arg, re.IGNORECASE)
                if not match: print(f"[错误] 无效的对号格式: '{pair_arg}'。请使用 P<数字> 格式 (例如: P1)。"); continue
                pair_num = int(match.group(1)); id1 = f"s{pair_num}-1"; id2 = f"s{pair_num}-2"
                if id1 not in self.semantic_id_to_key or id2 not in self.semantic_id_to_key: print(f"[错误] 找不到对号 P{pair_num} 对应的块 ID。"); continue
                key1 = self.semantic_id_to_key[id1]; key2 = self.semantic_id_to_key[id2]
                if command == 'keep':
                    self.block_decisions[key1] = 'keep'; self.block_decisions[key2] = 'keep'; logger.info(f"Pair P{pair_num}: Marked both {id1} and {id2} as 'keep'."); print(f"  已标记对 P{pair_num} 中的两个块 ({id1}, {id2}) 为 '保留'")
                elif command == 'k1d2':
                    self.block_decisions[key1] = 'keep'; self.block_decisions[key2] = 'delete'; logger.info(f"Pair P{pair_num}: Marked {id1} as 'keep', {id2} as 'delete'."); print(f"  已标记对 P{pair_num} 中的块 {id1} 为 '保留', {id2} 为 '删除'")
                elif command == 'k2d1':
                    self.block_decisions[key1] = 'delete'; self.block_decisions[key2] = 'keep'; logger.info(f"Pair P{pair_num}: Marked {id1} as 'delete', {id2} as 'keep'."); print(f"  已标记对 P{pair_num} 中的块 {id1} 为 '删除', {id2} 为 '保留'")

        logger.info("Finished interactive review of semantic duplicates (list mode).")
        print("\n--- 语义相似项处理完成 ---")


    # --- File Operations (使用硬编码路径, 增加调试输出) ---
    def load_decisions(self, filepath=None):
        """从 JSON 文件加载决策 (使用硬编码路径)。"""
        load_path = self.decision_file; logger.info(f"Attempting to load decisions from hardcoded path: {load_path}")
        if not load_path.exists(): logger.warning(f"Decision file not found: {load_path}"); print(f"[警告] 决策文件不存在: {load_path}"); return False
        try:
            with open(load_path, 'r', encoding='utf-8') as f: loaded_data = json.load(f); logger.info(f"Successfully read {len(loaded_data)} decision records from file.")
        except Exception as e: logger.error(f"Error reading or parsing JSON file: {load_path}", exc_info=True); print(f"[错误] 读取或解析 JSON 时出错: {e}"); return False
        if not self.block_decisions: logger.warning("Analysis data not present."); print("[警告] 分析尚未运行或未找到块。加载的决策可能无法完全应用。")
        loaded_count = 0; skipped_count = 0; unmatched_count = 0; loaded_decision_map = {}; path_resolve_errors = 0
        for item in loaded_data:
            if isinstance(item, dict) and all(k in item for k in ["file", "index", "type", "decision"]):
                try:
                    item_path = Path(item['file']); abs_path = (self.input_dir / item_path).resolve() if not item_path.is_absolute() and self.input_dir else item_path.resolve()
                    key = create_decision_key(str(abs_path), item['index'], item['type'])
                    if item['decision'] in ['keep', 'delete']: loaded_decision_map[key] = item['decision']
                    else: logger.warning(f"Invalid decision value '{item['decision']}' for key {key}. Skipping."); skipped_count += 1
                except Exception as path_e: logger.warning(f"Error processing path from loaded decision: {item['file']} - {path_e}. Skipping.", exc_info=False); path_resolve_errors += 1; skipped_count += 1
            else: logger.warning(f"Invalid format in loaded decision file: {item}. Skipping."); skipped_count += 1
        logger.info(f"Applying {len(loaded_decision_map)} valid loaded decisions...")
        for key in self.block_decisions: self.block_decisions[key] = 'undecided'
        for current_key in list(self.block_decisions.keys()):
            current_path_str, current_index, current_type = parse_decision_key(current_key)
            if current_path_str is None: continue
            try:
                resolved_current_path_str = str(Path(current_path_str).resolve()); match_key = create_decision_key(resolved_current_path_str, current_index, current_type)
                if match_key in loaded_decision_map: self.block_decisions[current_key] = loaded_decision_map[match_key]; logger.debug(f"Applied loaded decision '{loaded_decision_map[match_key]}' to key '{current_key}'"); loaded_count += 1
                else: unmatched_count += 1
            except Exception as resolve_err: logger.warning(f"Could not resolve path for key {current_key}: {resolve_err}")
        logger.info(f"Decision loading finished. Applied: {loaded_count}, Skipped: {skipped_count}, Path errors: {path_resolve_errors}.")
        if unmatched_count > 0: logger.info(f"{unmatched_count} blocks did not have a corresponding loaded decision.")
        print(f"[+] 成功应用 {loaded_count} 条加载的决策。")
        if skipped_count > 0 or path_resolve_errors > 0: print(f"[*] {skipped_count + path_resolve_errors} 条加载的决策被忽略。")
        return loaded_count > 0


    def save_decisions(self, filepath=None):
        """将当前决策保存到 JSON 文件 (使用硬编码路径, 增加调试输出)。"""
        save_path = self.decision_file; logger.info(f"Attempting to save decisions to hardcoded path: {save_path}")
        data_to_save = []; valid_decisions_count = 0; path_errors = 0
        for key, decision in self.block_decisions.items():
            if decision in ['keep', 'delete']:
                path_str, index, type_str = parse_decision_key(key)
                if path_str is not None:
                    try:
                        file_path_obj = Path(path_str); relative_path_str = path_str
                        if self.input_dir and file_path_obj.is_absolute():
                             try: relative_path_str = str(file_path_obj.relative_to(self.input_dir)); logger.debug(f"Saving relative path: {relative_path_str}")
                             except ValueError: relative_path_str = str(file_path_obj); logger.debug(f"Saving absolute path: {path_str}")
                        elif not file_path_obj.is_absolute(): relative_path_str = str(file_path_obj); logger.debug(f"Saving already relative path: {path_str}")
                        else: relative_path_str = str(file_path_obj); logger.debug(f"Saving absolute path: {path_str}")
                        data_to_save.append({"file": relative_path_str, "index": index, "type": type_str, "decision": decision}); valid_decisions_count += 1
                    except Exception as path_e: logger.warning(f"Error processing path for key {key}: {path_e}. Skipping.", exc_info=False); path_errors += 1
                else: logger.warning(f"Could not parse key '{key}'. Skipping."); path_errors += 1
        if not data_to_save: logger.info("No decisions to save."); print("[信息] 没有已标记为 '保留' 或 '删除' 的决策可供保存。"); return False
        logger.info(f"Attempting to save {valid_decisions_count} decisions.");
        if path_errors > 0: logger.warning(f"{path_errors} decisions could not be saved.")
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f: json.dump(data_to_save, f, ensure_ascii=False, indent=4); logger.info(f"Successfully saved {valid_decisions_count} decisions to: {save_path}")
            print(f"[+] {valid_decisions_count} 条决策已成功保存到: {save_path}")

            # --- 新增：在 Debug 模式下显示保存的文件内容 ---
            if logger.level == logging.DEBUG:
                try:
                    saved_content = save_path.read_text(encoding='utf-8')
                    print("\n--- 保存的决策文件内容 (Debug) ---")
                    print(saved_content)
                    print("----------------------------------")
                except Exception as read_err:
                    logger.warning(f"无法读取刚保存的决策文件以供调试显示: {read_err}")
            # --------------------------------------------
            return True
        except Exception as e: logger.error(f"Failed to save decisions file: {save_path}", exc_info=True); print(f"[错误] 保存文件时出错: {e}"); return False

    def apply_decisions(self, output_dir_path=None):
        """根据内存中的决策生成新的去重后的 Markdown 文件 (使用硬编码输出路径, 增加调试输出)。"""
        output_path_obj = self.output_dir; logger.info(f"Attempting to apply decisions and write output files to hardcoded path: {output_path_obj}")
        if not self.blocks_data: logger.error("Cannot apply decisions: No block data."); print("[错误] 没有解析出的块数据。"); return False
        if not self.block_decisions: logger.warning("No decision data available. Defaulting to 'keep'."); print("[警告] 没有任何决策信息，将默认保留所有块。")
        try: output_path_obj.mkdir(parents=True, exist_ok=True); logger.info(f"Output directory confirmed/created: {output_path_obj}")
        except Exception as e: logger.error(f"Failed to create output directory: {output_path_obj}", exc_info=True); print(f"[错误] 无法创建输出目录: {output_path_obj} - {e}"); return False
        file_to_blocks = collections.defaultdict(list); original_paths = set()
        for block_info in self.blocks_data: file_to_blocks[block_info[0]].append(block_info); original_paths.add(block_info[0])
        logger.info(f"Applying decisions for {len(original_paths)} original files.")
        processed_files_count = 0; error_files = []; total_kept_blocks = 0; total_deleted_blocks = 0
        for original_file_path in sorted(list(original_paths), key=str):
            logger.debug(f"Processing file: {original_file_path.name}")
            def sort_key(block_info):
                index_val = block_info[1]
                if isinstance(index_val, str) and '_' in index_val:
                    try: main_idx, sub_idx = map(int, index_val.split('_')); return (main_idx, sub_idx)
                    except ValueError: return (float('inf'), index_val)
                try: return (int(index_val), 0)
                except ValueError: return (float('inf'), index_val)
            blocks_for_this_file = sorted(file_to_blocks.get(original_file_path, []), key=sort_key)
            if not blocks_for_this_file: logger.debug(f"No blocks for {original_file_path.name}, skipping."); continue
            new_content_parts = []; kept_block_count = 0; deleted_block_count = 0
            for block_info in blocks_for_this_file:
                try: abs_path_str = str(block_info[0].resolve()); key = create_decision_key(abs_path_str, block_info[1], block_info[2])
                except Exception as e: logger.warning(f"Could not resolve path for key lookup: {block_info[0]} - {e}. Defaulting keep."); key = None
                decision = self.block_decisions.get(key, 'keep') if key else 'keep'
                if decision != 'delete':
                    file_path, b_index, b_type, b_text = block_info; formatted_text = b_text
                    if b_type == 'block_code':
                        if not b_text.strip().startswith("```"): formatted_text = f"```\n{b_text}\n```"
                    elif b_type == 'list_item': formatted_text = f"- {b_text}"
                    elif b_type == 'block_quote': lines = b_text.splitlines(); formatted_text = "\n".join([f"> {line}" for line in lines])
                    elif b_type == 'block_html': formatted_text = b_text
                    new_content_parts.append(formatted_text); kept_block_count += 1
                else: deleted_block_count += 1; logger.debug(f"Deleting block: Index {block_info[1]} in {original_file_path.name}")
            new_content = "\n\n".join(new_content_parts); output_filename = f"{original_file_path.stem}{DEFAULT_OUTPUT_SUFFIX}{original_file_path.suffix}"; output_filepath = output_path_obj / output_filename
            logger.debug(f"Writing output file: {output_filepath.name} (Kept: {kept_block_count}, Deleted: {deleted_block_count})"); total_kept_blocks += kept_block_count; total_deleted_blocks += deleted_block_count
            try:
                output_filepath.write_text(new_content, encoding='utf-8'); logger.info(f"Successfully wrote output file: {output_filepath.name}"); print(f"  [+] 已生成: {output_filepath.name} (保留 {kept_block_count} 块, 删除 {deleted_block_count} 块)"); processed_files_count += 1
                # --- 新增：在 Debug 模式下显示生成的去重文件内容 ---
                if logger.level == logging.DEBUG:
                    try:
                        written_content = output_filepath.read_text(encoding='utf-8')
                        print(f"\n--- 生成的文件内容 (Debug): {output_filepath.name} ---")
                        print(written_content)
                        print("--------------------------------------------------")
                    except Exception as read_err:
                        logger.warning(f"无法读取刚生成的去重文件以供调试显示: {read_err}")
                # -------------------------------------------------
            except Exception as e: logger.error(f"Error writing output file: {output_filepath.name}", exc_info=True); print(f"  [错误] 写入文件 '{output_filepath.name}' 时出错: {e}"); error_files.append(original_file_path.name)
        logger.info("--- Finished applying decisions ---"); logger.info(f"Processed {len(original_paths)} original files."); logger.info(f"Successfully generated {processed_files_count} deduplicated files."); logger.info(f"Total blocks kept: {total_kept_blocks}, Total blocks deleted: {total_deleted_blocks}")
        print(f"[*] 共处理了 {len(original_paths)} 个原始文件。"); print(f"[+] 成功生成了 {processed_files_count} 个去重后的文件。")
        if error_files: logger.error(f"{len(error_files)} files encountered errors: {', '.join(error_files)}"); print(f"[错误] 以下 {len(error_files)} 个文件在写入时发生错误:\n  - " + "\n  - ".join(error_files))
        return processed_files_count > 0 or not error_files


# --- 菜单函数 (无变化) ---
def get_input_directory():
    """提示用户输入有效的文件夹路径。"""
    while True:
        dir_path = input("请输入包含 Markdown 文件的目标文件夹路径: ").strip()
        if not dir_path: print("[提示] 输入不能为空，请重新输入。"); continue
        path = Path(dir_path)
        if path.is_dir(): logger.info(f"User provided valid input directory: {path}"); return path
        else: print(f"[错误] 输入的路径 '{dir_path}' 不是一个有效的文件夹，请重新输入。"); logger.warning(f"User provided invalid directory path: {dir_path}")

def display_main_menu():
    """显示主菜单选项。"""
    print("\n--- 主菜单 ---"); print("1. 运行分析 (查找重复项)"); print("2. 退出程序"); print("------------")

def display_post_analysis_menu(tool_instance):
    """显示分析完成后的菜单选项。"""
    print("\n--- 分析后菜单 ---")
    print("1. 查看并处理 MD5 精确重复项 (列表模式)")
    if not tool_instance.skip_semantic: print("2. 查看并处理语义相似项 (列表模式)")
    else: print("2. (语义分析已跳过)")
    print(f"3. 从文件加载决策 ({tool_instance.decision_file.name})")
    print(f"4. 保存当前决策到文件 ({tool_instance.decision_file.name})")
    print(f"5. 应用决策 (生成到 '{tool_instance.output_dir.name}' 目录)")
    print("6. 返回主菜单 (可以重新分析或退出)"); print("7. 退出程序"); print("------------------")


# --- 主程序入口 (使用 logger) ---
if __name__ == "__main__":
    print("欢迎使用知识蒸馏 KD Tool (命令行交互版)"); print("======================================"); logger.info("KD Tool started.")
    skip_semantic_analysis = False; similarity_thresh = SIMILARITY_THRESHOLD; log_level = logging.INFO
    hardcoded_test_path = HARDCODED_INPUT_PATH

    if "--debug" in sys.argv:
        log_level = logging.DEBUG; logger.setLevel(log_level); console_handler.setLevel(log_level); logger.info("DEBUG logging enabled."); print("[信息] 检测到 --debug 标志，将输出详细日志和文件内容。") # 更新提示
    if "--skip-semantic" in sys.argv: skip_semantic_analysis = True; logger.info("Semantic analysis skipped."); print("[信息] 检测到 --skip-semantic 标志，将跳过语义分析。")
    try:
        threshold_index = sys.argv.index("--threshold")
        if threshold_index + 1 < len(sys.argv):
            similarity_thresh = float(sys.argv[threshold_index + 1])
            logger.info(f"Similarity threshold set to {similarity_thresh} via command line flag.")
            print(f"[信息] 使用命令行设置的相似度阈值: {similarity_thresh}")
    except (ValueError, IndexError): logger.debug("Could not parse --threshold argument."); pass

    kd_tool = KDToolCLI(skip_semantic=skip_semantic_analysis, similarity_threshold=similarity_thresh)
    logger.info(f"Using hardcoded input directory: {hardcoded_test_path}")
    input_dir = Path(hardcoded_test_path)
    if not input_dir.is_dir(): logger.critical(f"Hardcoded path is invalid: {hardcoded_test_path}. Exiting."); print(f"[错误] 硬编码的路径无效: {hardcoded_test_path}。程序退出。"); sys.exit(1)
    if not kd_tool.set_input_directory(input_dir): logger.critical("Failed to set input directory. Exiting."); print("[!] 无法设置输入目录，程序退出。"); sys.exit(1)

    analysis_run_successfully = False
    logger.info("Entering main menu loop.")
    while True:
        display_main_menu()
        choice = input("请输入选项编号: ").strip(); logger.debug(f"Main menu choice: '{choice}'")
        if choice == '1':
            logger.info("User chose option 1: Run Analysis"); analysis_run_successfully = kd_tool.run_analysis()
            if analysis_run_successfully:
                logger.info("Analysis completed successfully. Entering post-analysis menu.")
                while True:
                    display_post_analysis_menu(kd_tool)
                    sub_choice = input("请输入选项编号: ").strip(); logger.debug(f"Post-analysis menu choice: '{sub_choice}'")
                    if sub_choice == '1': logger.info("User chose option 1: Review MD5 Duplicates (List Mode)"); kd_tool.review_md5_duplicates_interactive()
                    elif sub_choice == '2':
                        if not kd_tool.skip_semantic: logger.info("User chose option 2: Review Semantic Duplicates (List Mode)"); kd_tool.review_semantic_duplicates_interactive()
                        else: logger.info("User chose option 2, but semantic analysis was skipped."); print("[信息] 语义分析已被跳过。")
                    elif sub_choice == '3': logger.info("User chose option 3: Load Decisions (Hardcoded Path)"); kd_tool.load_decisions()
                    elif sub_choice == '4': logger.info("User chose option 4: Save Decisions (Hardcoded Path)"); kd_tool.save_decisions()
                    elif sub_choice == '5': logger.info("User chose option 5: Apply Decisions (Hardcoded Path)"); kd_tool.apply_decisions()
                    elif sub_choice == '6': logger.info("User chose option 6: Return to Main Menu"); print("[*] 返回主菜单..."); break
                    elif sub_choice == '7': logger.info("User chose option 7: Exit Program"); print("[*] 退出程序。"); sys.exit(0)
                    else: logger.warning(f"Invalid post-analysis menu choice: '{sub_choice}'"); print("[错误] 无效的选项，请重新输入。")
            else: logger.error("Analysis failed. Returning to main menu."); print("[!] 分析未能成功完成，请检查错误信息。")
        elif choice == '2': logger.info("User chose option 2: Exit Program from main menu."); print("[*] 退出程序。"); break
        else: logger.warning(f"Invalid main menu choice: '{choice}'"); print("[错误] 无效的选项，请重新输入。")
    logger.info("KD Tool finished.")
    sys.exit(0)
