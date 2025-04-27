# [DEPENDENCIES]
# 1. Python Standard Library: sys, os, json, logging, collections, pathlib, argparse
# 2. 需要安装：mistune, sentence-transformers (可选, 用于语义分析)
# 3. 同项目模块: constants, utils, md5_analyzer, semantic_analyzer (使用绝对导入)

import sys
import os
import json
import logging
import collections
from pathlib import Path
import argparse # 引入 argparse

# 使用绝对导入导入自定义模块
import constants
import utils
from md5_analyzer import MD5Analyzer
import semantic_analyzer # 导入整个模块以访问模块级变量

# 使用 utils 中配置好的 logger
logger = utils.logger

# --- 核心逻辑类 ---
class KDToolCLI:
    """
    知识蒸馏工具的主控制类，协调文件读取、解析、分析和决策应用。
    """
    # 修改 __init__ 以接收路径参数
    def __init__(self, output_dir_path, decision_file_path, skip_semantic=False, similarity_threshold=constants.DEFAULT_SIMILARITY_THRESHOLD):
        logger.info("Initializing KDToolCLI instance...")
        self.input_dir = None # 输入目录路径 (Path object) - 将通过 set_input_directory 设置
        # 从参数接收路径
        self.output_dir = Path(output_dir_path)
        self.decision_file = Path(decision_file_path)

        self.skip_semantic = skip_semantic # 是否跳过语义分析
        self.similarity_threshold = similarity_threshold # 语义相似度阈值

        logger.info(f"Settings: skip_semantic={self.skip_semantic}, similarity_threshold={self.similarity_threshold}")
        logger.info(f"Using Decision Path: {self.decision_file}")
        logger.info(f"Using Output Dir: {self.output_dir}")

        # --- 状态变量 ---
        self.markdown_files_content = {} # {Path_object: file_content_string}
        self.blocks_data = [] # [(Path_object, index, type, text), ...]
        self.block_decisions = {} # {decision_key: 'keep'/'delete'/'undecided'}

        # --- 分析器实例 ---
        # 使用直接导入的类名进行实例化
        self.md5_analyzer = MD5Analyzer(self)
        # 使用 模块名.类名 实例化，确保一致性
        self.semantic_analyzer = semantic_analyzer.SemanticAnalyzer(self) # SemanticAnalyzer 内部会检查库可用性

        # 确保目录存在 (移到 set_input_directory 之后或主程序逻辑中更合适)
        # self._ensure_dirs_exist() # 暂时注释掉，在主逻辑中处理

        logger.info("KDToolCLI instance initialized.")

    def _ensure_dirs_exist(self):
        """确保决策文件和输出目录存在。"""
        try:
            # 确保决策文件所在的目录存在
            self.decision_file.parent.mkdir(parents=True, exist_ok=True)
            # 确保输出目录存在
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Ensured decision/output directories exist.")
            print(f"[信息] 已确认/创建目录:")
            # print(f"  - 输入: {self.input_dir}") # input_dir 可能还未设置
            print(f"  - 决策: {self.decision_file.parent}")
            print(f"  - 输出: {self.output_dir}")
        except OSError as e:
            logger.warning(f"Could not create parent directories for decision/output paths: {e}", exc_info=False)
            print(f"[警告] 无法自动创建决策/输出目录，请检查权限: {e}")
        except Exception as e:
             logger.error(f"Error ensuring directories exist: {e}", exc_info=True)
             print(f"[错误] 检查或创建目录时发生未知错误: {e}")


    def set_input_directory(self, input_dir_path):
        """设置输入目录并进行基本验证。"""
        logger.info(f"Attempting to set input directory: {input_dir_path}")
        try:
            path = Path(input_dir_path).resolve() # 解析为绝对路径
            if not path.is_dir():
                logger.error(f"Invalid directory path provided: {input_dir_path}")
                print(f"[错误] 输入的路径 '{input_dir_path}' 不是一个有效的文件夹。")
                return False

            # 确保输入目录本身存在 (虽然调用前通常会检查，但这里再确认下)
            path.mkdir(parents=True, exist_ok=True) # 尝试创建（如果不存在）

            self.input_dir = path
            logger.info(f"Input directory set successfully: {self.input_dir}")

            # 在设置输入目录后，确保所有需要的目录都存在
            self._ensure_dirs_exist()

            # 尝试加载语义模型（如果需要）
            self.semantic_analyzer.load_semantic_model()

            return True
        except OSError as e:
            logger.error(f"Error setting input directory or ensuring directories: {input_dir_path} - {e}", exc_info=True)
            print(f"[错误] 设置输入目录或创建所需目录时出错: {e}")
            return False
        except Exception as e:
            logger.error(f"Error setting input directory {input_dir_path}: {e}", exc_info=True)
            print(f"[错误] 设置输入目录时发生未知错误: {e}")
            return False

    # ... (run_analysis, _read_files, _parse_markdown, _initialize_decisions 不变) ...
    def run_analysis(self):
        """执行完整的分析流程：读取 -> 解析 -> 初始化决策 -> MD5 -> 语义。"""
        if not self.input_dir:
            logger.error("Analysis aborted: Input directory not set.")
            print("[错误] 输入文件夹未设置。")
            return False

        logger.info(f"--- Starting analysis for folder: {self.input_dir} ---")
        logger.info("Resetting internal state for new analysis...")
        # 重置状态变量
        self.markdown_files_content = {}
        self.blocks_data = []
        self.block_decisions = {}
        self.md5_analyzer.duplicate_blocks = {} # 重置 MD5 分析器的结果
        self.md5_analyzer.md5_id_to_key = {}
        self.semantic_analyzer.semantic_duplicates = [] # 重置语义分析器的结果
        self.semantic_analyzer.semantic_id_to_key = {}
        logger.info("Internal state reset.")

        # --- Step 1: 读取文件 ---
        logger.info("Step 1: Reading files...")
        if not self._read_files():
            logger.error("Analysis stopped: File reading failed.")
            return False
        logger.info("Step 1: File reading completed.")
        if not self.markdown_files_content:
             logger.warning("No Markdown files were read. Stopping analysis.")
             print("[警告] 未能读取任何 Markdown 文件内容，分析中止。")
             return False # 如果没有文件内容，后续步骤无意义

        # --- Step 2: 解析 Markdown ---
        logger.info("Step 2: Parsing Markdown...")
        if not self._parse_markdown():
            logger.error("Analysis stopped: Markdown parsing failed.")
            return False
        logger.info("Step 2: Markdown parsing completed.")
        if not self.blocks_data:
             logger.warning("No content blocks were extracted from files. Analysis finished.")
             print("[警告] 未能从文件中提取任何有效内容块。")
             # 即使没有块，也认为分析“成功”完成，只是没有找到内容
             return True # 返回 True，允许用户查看空结果或加载决策

        # --- Step 3: 初始化决策 ---
        logger.info("Step 3: Initializing block decisions...")
        self._initialize_decisions()
        logger.info("Step 3: Block decisions initialized.")

        # --- Step 4: MD5 去重 ---
        logger.info("Step 4: Performing MD5 deduplication...")
        self.md5_analyzer.find_md5_duplicates() # 调用 MD5 分析器的方法
        logger.info("Step 4: MD5 deduplication completed.")

        # --- Step 5: 语义去重 (如果未跳过) ---
        if not self.skip_semantic:
            logger.info("Step 5: Performing semantic deduplication...")
            self.semantic_analyzer.find_semantic_duplicates() # 调用语义分析器的方法
            logger.info("Step 5: Semantic deduplication completed.")
        else:
            logger.info("Step 5: Semantic deduplication skipped as requested.")

        logger.info("--- Analysis finished ---")
        return True

    def _read_files(self):
        """查找并读取输入目录中的 Markdown 文件。"""
        if not self.input_dir: # 增加检查
             logger.error("Cannot read files: Input directory not set.")
             return False
        logger.info(f"Scanning directory for Markdown files: {self.input_dir}")
        try:
            # 查找所有 .md 文件，不区分大小写, 忽略以 '.' 开头的文件/文件夹
            markdown_files = list(self.input_dir.glob('[!.]*.md')) + list(self.input_dir.glob('[!.]*.MD'))
            # 过滤掉可能的子目录中的文件（如果需要）
            markdown_files = [f for f in markdown_files if f.is_file()]
            logger.info(f"Found {len(markdown_files)} potential Markdown files.")

            if not markdown_files:
                logger.warning("No Markdown files (.md, .MD) found in the input directory.")
                print(f"[错误] 在目录 '{self.input_dir}' 未找到任何 Markdown 文件。")
                return False # 返回 False 表示没有找到文件

            files_read_count = 0
            for md_file in markdown_files:
                logger.debug(f"Attempting to read: {md_file.name}")
                try:
                    # 使用 utf-8 读取文件内容
                    content = md_file.read_text(encoding='utf-8')
                    # 存储文件路径对象和内容
                    self.markdown_files_content[md_file] = content
                    files_read_count += 1
                    logger.debug(f"Successfully read: {md_file.name}")
                except Exception as e:
                    # 记录读取单个文件时的错误
                    logger.error(f"Error reading file: {md_file.name} - {e}", exc_info=False)
                    print(f"[警告] 读取文件失败: {md_file.name} - {e}")

            logger.info(f"Content reading complete: Successfully read {files_read_count} / {len(markdown_files)} files.")

            # 如果找到了文件但一个都没成功读取
            if files_read_count == 0 and markdown_files:
                logger.error("Failed to read content from any of the found Markdown files.")
                print("[错误] 未能成功读取任何找到的 Markdown 文件。")
                return False
            # 如果成功读取了至少一个文件
            elif files_read_count > 0:
                return True
            else: # 如果没有找到文件 (前面已处理)
                return False

        except Exception as e:
            # 记录查找文件过程中的错误
            logger.error(f"Error finding files in directory: {self.input_dir}", exc_info=True)
            print(f"[错误] 查找文件时出错: {e}")
            return False

    def _parse_markdown(self):
        """解析已读取的 Markdown 文件内容并提取非标题块。"""
        logger.info("Parsing Markdown content and extracting blocks (excluding headings)...")
        self.blocks_data = [] # 清空旧数据
        markdown_parser = utils.get_markdown_parser() # 获取解析器实例
        total_files = len(self.markdown_files_content)
        parsed_files_count = 0

        try:
            # 遍历每个已读取的文件
            for md_file_path, content in self.markdown_files_content.items():
                logger.debug(f"Parsing file: {md_file_path.name}")
                file_block_count = 0
                current_token = None # 用于在异常处理中引用
                try:
                    # 使用 mistune 解析 Markdown 内容为令牌 (tokens)
                    block_tokens = markdown_parser(content)
                    if not block_tokens:
                        logger.debug(f"No tokens found for: {md_file_path.name}")
                        continue # 如果文件解析后没有令牌，跳过

                    # 遍历文件的顶层令牌 (通常是块级元素)
                    for index, token in enumerate(block_tokens):
                        current_token = token # 保存当前token供调试
                        block_type = token.get('type', 'unknown')
                        block_text = ""
                        blocks_to_add = [] # 用于收集当前 token 可能产生的多个块 (例如列表项)

                        # --- 根据块类型提取文本 ---
                        if block_type == 'heading':
                            continue # 跳过标题块
                        elif block_type == 'paragraph':
                            # 提取段落内的文本内容
                            block_text = utils.extract_text_from_children(token.get('children', []))
                        elif block_type == 'block_code':
                            # 提取代码块的原始内容 (包括语言标识符和代码)
                            block_text = token.get('raw') if token.get('raw') is not None else token.get('text', '')
                        elif block_type == 'block_quote':
                            # 提取引用块内的文本内容
                            block_text = utils.extract_text_from_children(token.get('children', []))
                        elif block_type == 'list':
                            # 特殊处理列表：将每个列表项作为一个独立的块
                            list_items = token.get('children', [])
                            for item_index, item_token in enumerate(list_items):
                                if item_token.get('type') == 'list_item':
                                    # 提取列表项的文本内容
                                    item_text = utils.extract_text_from_children(item_token.get('children', []))
                                    cleaned_item_text = item_text.strip()
                                    if cleaned_item_text:
                                        # 为列表项创建复合索引，例如 "主索引_列表项索引"
                                        list_item_block_index = f"{index}_{item_index}"
                                        # 添加列表项块信息
                                        blocks_to_add.append((md_file_path, list_item_block_index, 'list_item', cleaned_item_text))
                            # 列表本身不作为一个块添加，其内容已分解为 list_item
                            block_text = None # 防止后续处理
                        elif block_type == 'thematic_break': # 例如 --- 或 ***
                            continue # 跳过分隔线
                        elif block_type == 'block_html':
                            # 提取 HTML 块的原始内容
                            block_text = token.get('raw', '')
                        elif block_type == 'table':
                            # 提取表格的表头和主体文本
                            header_text = utils.extract_text_from_children(token.get('header', []))
                            body_text = utils.extract_text_from_children(token.get('children', []))
                            # 将表头和主体合并为一个文本表示
                            block_text = f"Table Header: {header_text}\nTable Body:\n{body_text}"
                        else:
                             # 对于其他未明确处理的块类型，尝试获取原始文本或子节点文本
                             logger.debug(f"Handling unknown or potentially empty block type '{block_type}' at index {index} in {md_file_path.name}")
                             block_text = token.get('raw') if token.get('raw') is not None else token.get('text', '')
                             if not block_text and 'children' in token:
                                 block_text = utils.extract_text_from_children(token.get('children', []))

                        # --- 清理和添加块 ---
                        # 如果是从 token 直接提取的文本 (非列表项)
                        if block_text is not None:
                             # 对非代码、HTML、列表的块进行 strip() 清理
                             cleaned_text = block_text.strip() if block_type not in ['block_code', 'block_html'] else block_text
                             # 如果清理后仍有内容，则添加到待添加列表
                             if cleaned_text:
                                 blocks_to_add.append((md_file_path, index, block_type, cleaned_text))

                        # 如果当前 token 产生了需要添加的块
                        if blocks_to_add:
                            self.blocks_data.extend(blocks_to_add)
                            file_block_count += len(blocks_to_add)
                            logger.debug(f"Added {len(blocks_to_add)} block(s) from token index {index} (type: {block_type}) in {md_file_path.name}")

                    parsed_files_count += 1
                    logger.debug(f"Finished parsing {md_file_path.name}, found {file_block_count} valid content blocks.")
                except Exception as parse_err:
                    # 记录解析单个文件时的错误
                    logger.warning(f"Error parsing file {md_file_path.name} near token {current_token}: {parse_err}", exc_info=False)
                    print(f"[警告] 解析文件时出错: {md_file_path.name} - {parse_err}")
                    continue # 继续处理下一个文件

            total_blocks = len(self.blocks_data)
            logger.info(f"Markdown parsing complete: Processed {parsed_files_count}/{total_files} files. Extracted {total_blocks} total valid content blocks.")

            if total_blocks == 0 and parsed_files_count > 0:
                logger.warning("No valid content blocks were extracted from the parsed files.")
                # 即使没有块，也认为解析过程本身是成功的
            return True
        except Exception as e:
            # 记录解析过程中的意外错误
            logger.error("Unexpected error during Markdown parsing process.", exc_info=True)
            print(f"[错误] Markdown 解析过程中发生意外错误: {e}")
            return False

    def _initialize_decisions(self):
        """为所有解析出的块初始化决策状态为 'undecided'。"""
        logger.info("Initializing decision state for all extracted blocks...")
        self.block_decisions = {} # 清空旧决策
        count = 0
        path_resolve_errors = 0
        # 遍历所有提取的块信息
        for block_info in self.blocks_data:
            file_path, b_index, b_type, _ = block_info
            try:
                # 创建决策键，确保使用绝对路径
                abs_path_str = str(Path(file_path).resolve())
                key = utils.create_decision_key(abs_path_str, b_index, b_type)
                # 初始化决策为 'undecided'
                self.block_decisions[key] = 'undecided'
                count += 1
            except Exception as e:
                 # 记录创建键时的错误
                 file_name = file_path.name if isinstance(file_path, Path) else str(file_path)
                 logger.warning(f"Could not create decision key for block {file_name}#{b_index}: {e}. Skipping initialization for this block.")
                 path_resolve_errors += 1

        logger.info(f"Initialized decisions for {count} blocks to 'undecided'.")
        if path_resolve_errors > 0:
            logger.warning(f"Failed to initialize decisions for {path_resolve_errors} blocks due to path resolution issues.")

    # --- 文件操作方法 ---

    def load_decisions(self):
        """从 JSON 文件加载决策 (使用 self.decision_file)。"""
        load_path = self.decision_file
        logger.info(f"Attempting to load decisions from: {load_path}")

        if not load_path.exists():
            logger.warning(f"Decision file not found: {load_path}")
            print(f"[警告] 决策文件不存在: {load_path}")
            return False

        try:
            # 读取 JSON 文件
            with open(load_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            logger.info(f"Successfully read {len(loaded_data)} decision records from file.")
        except json.JSONDecodeError as json_err:
             logger.error(f"Error parsing JSON file: {load_path} - {json_err}", exc_info=False)
             print(f"[错误] 解析 JSON 文件时出错: {load_path} - {json_err}")
             return False
        except Exception as e:
            logger.error(f"Error reading decision file: {load_path}", exc_info=True)
            print(f"[错误] 读取决策文件时出错: {e}")
            return False

        if not self.blocks_data:
            logger.warning("Block data is not available (analysis might not have been run). Loaded decisions cannot be applied yet.")
            print("[警告] 分析尚未运行或未找到块。加载的决策将在下次分析后或应用决策时尝试匹配。")
            # 可以在这里将加载的决策暂存起来，但目前的设计是在应用时匹配
            # self.loaded_decisions_cache = loaded_data # 示例性缓存
            return False # 或者返回 True 表示文件已读取，但未应用？当前返回 False

        # --- 应用加载的决策 ---
        loaded_count = 0
        skipped_invalid_format = 0
        skipped_invalid_decision = 0
        path_resolve_errors = 0
        loaded_decision_map = {} # {decision_key: decision_value}

        # 1. 解析加载的数据，构建决策映射 (使用绝对路径作为键)
        for item in loaded_data:
            if isinstance(item, dict) and all(k in item for k in ["file", "index", "type", "decision"]):
                try:
                    # 将加载的文件路径转换为绝对路径
                    # 假设存储的是相对路径 (相对于 input_dir) 或绝对路径
                    item_path_str = item['file']
                    item_path = Path(item_path_str)
                    if not item_path.is_absolute():
                        if self.input_dir:
                            # 如果是相对路径且 input_dir 已设置，则相对于 input_dir 解析
                            abs_path = (self.input_dir / item_path).resolve()
                        else:
                             logger.warning(f"Cannot resolve relative path '{item_path_str}' from decision file because input directory is not set. Skipping.")
                             path_resolve_errors += 1
                             continue # 跳过无法解析的相对路径
                    else:
                        # 如果已经是绝对路径
                        abs_path = item_path.resolve()

                    # 创建决策键
                    key = utils.create_decision_key(str(abs_path), item['index'], item['type'])

                    # 验证决策值
                    decision_value = item['decision']
                    if decision_value in ['keep', 'delete']:
                        loaded_decision_map[key] = decision_value
                    else:
                        logger.warning(f"Invalid decision value '{decision_value}' found for key {key} in loaded file. Skipping.")
                        skipped_invalid_decision += 1

                except Exception as path_e:
                    logger.warning(f"Error processing path from loaded decision: {item.get('file', 'N/A')} - {path_e}. Skipping.", exc_info=False)
                    path_resolve_errors += 1
            else:
                logger.warning(f"Invalid format in loaded decision file: {item}. Skipping.")
                skipped_invalid_format += 1

        total_skipped = skipped_invalid_format + skipped_invalid_decision + path_resolve_errors
        logger.info(f"Parsed {len(loaded_decision_map)} valid decisions from file. Skipped {total_skipped} invalid/unresolvable entries.")

        if not loaded_decision_map:
             print("[信息] 从文件中未加载到有效的决策信息。")
             return False

        # 2. 将加载的决策应用到当前的 self.block_decisions
        # 注意：这里会覆盖现有的 'undecided' 或之前的决策
        applied_count = 0
        unmatched_count = 0
        # apply_path_errors = 0 # 这个变量未使用，移除

        # 先重置所有当前块的决策为 undecided，确保只应用加载的决策
        self._initialize_decisions() # 重置为 undecided

        for current_key in self.block_decisions: # 遍历当前所有块的 key
            # 尝试在加载的决策映射中查找当前键
            if current_key in loaded_decision_map:
                 new_decision = loaded_decision_map[current_key]
                 self.block_decisions[current_key] = new_decision
                 # logger.debug(f"Applied loaded decision '{new_decision}' to key '{current_key}'")
                 applied_count += 1
            else:
                 # 如果当前块的键在加载的决策中找不到, 保持 undecided
                 unmatched_count += 1

        logger.info(f"Decision loading finished. Applied: {applied_count} decisions to current blocks.")
        if unmatched_count > 0:
            logger.info(f"{unmatched_count} current blocks did not have a corresponding loaded decision (remain undecided).")
        if total_skipped > 0:
             logger.warning(f"Skipped loading {total_skipped} entries from the file due to format/path/value issues.")

        print(f"[+] 成功应用 {applied_count} 条加载的决策。")
        if total_skipped > 0:
            print(f"[*] {total_skipped} 条来自文件的决策记录因格式、路径或值无效而被忽略。")
        if unmatched_count > 0:
             print(f"[*] {unmatched_count} 个当前内容块在加载的决策文件中没有找到对应项 (保持未定状态)。")

        return applied_count > 0

    def save_decisions(self):
        """将当前内存中的 'keep' 或 'delete' 决策保存到 JSON 文件 (self.decision_file)。"""
        save_path = self.decision_file
        logger.info(f"Attempting to save decisions to: {save_path}")

        data_to_save = []
        valid_decisions_count = 0
        path_errors = 0

        # 遍历当前所有决策
        for key, decision in self.block_decisions.items():
            # 只保存明确标记为 'keep' 或 'delete' 的决策
            if decision in ['keep', 'delete']:
                # 从决策键解析出文件路径、索引和类型
                path_str, index, type_str = utils.parse_decision_key(key)

                if path_str is not None:
                    try:
                        file_path_obj = Path(path_str)
                        # 尝试将绝对路径转换为相对于 input_dir 的相对路径进行存储
                        relative_path_str = path_str # 默认为原始（绝对）路径
                        if self.input_dir and file_path_obj.is_absolute():
                            try:
                                # 计算相对路径
                                relative_path = file_path_obj.relative_to(self.input_dir)
                                relative_path_str = str(relative_path)
                                logger.debug(f"Saving relative path for {key}: {relative_path_str}")
                            except ValueError:
                                # 如果不在 input_dir 下，无法计算相对路径，保存绝对路径
                                logger.warning(f"Path {path_str} is not inside the input directory {self.input_dir}. Saving absolute path.")
                                relative_path_str = str(file_path_obj) # 保存绝对路径字符串
                        else:
                             # 如果原始路径已经是相对路径或 input_dir 未设置，直接使用
                             relative_path_str = str(file_path_obj)
                             logger.debug(f"Saving non-relative or original path for {key}: {relative_path_str}")


                        # 添加到待保存列表
                        data_to_save.append({
                            "file": relative_path_str, # 保存相对或绝对路径字符串
                            "index": index, # 保存解析出的 index (可能是 int 或 str)
                            "type": type_str,
                            "decision": decision
                        })
                        valid_decisions_count += 1
                    except Exception as path_e:
                        # 记录处理路径时的错误
                        logger.warning(f"Error processing path for key {key}: {path_e}. Skipping decision.", exc_info=False)
                        path_errors += 1
                else:
                    # 记录无法解析键的错误
                    logger.warning(f"Could not parse key '{key}' during saving. Skipping decision.")
                    path_errors += 1

        if not data_to_save:
            logger.info("No decisions marked as 'keep' or 'delete' to save.")
            print("[信息] 没有已标记为 '保留' 或 '删除' 的决策可供保存。")
            return False

        logger.info(f"Attempting to save {valid_decisions_count} decisions.")
        if path_errors > 0:
            logger.warning(f"{path_errors} decisions could not be saved due to key parsing or path processing errors.")

        try:
            # 确保保存目录存在
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # 将决策数据写入 JSON 文件
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            logger.info(f"Successfully saved {valid_decisions_count} decisions to: {save_path}")
            print(f"[+] {valid_decisions_count} 条决策已成功保存到: {save_path}")

            # --- 在 Debug 模式下显示保存的文件内容 ---
            if logger.level == logging.DEBUG:
                try:
                    saved_content = save_path.read_text(encoding='utf-8')
                    print("\n--- 保存的决策文件内容 (Debug) ---")
                    # 限制预览长度，避免刷屏
                    print(saved_content[:1000] + ("..." if len(saved_content) > 1000 else ""))
                    print("----------------------------------")
                except Exception as read_err:
                    logger.warning(f"无法读取刚保存的决策文件以供调试显示: {read_err}")
            # --------------------------------------------

            return True
        except Exception as e:
            # 记录保存文件时的错误
            logger.error(f"Failed to save decisions file: {save_path}", exc_info=True)
            print(f"[错误] 保存文件时出错: {e}")
            return False

    def apply_decisions(self):
        """
        根据内存中的决策 (self.block_decisions) 生成新的去重后的 Markdown 文件。
        输出到 self.output_dir。
        """
        output_path_obj = self.output_dir
        logger.info(f"Attempting to apply decisions and write output files to: {output_path_obj}")

        if not self.blocks_data:
            logger.error("Cannot apply decisions: No block data available (run analysis first).")
            print("[错误] 没有解析出的块数据，无法应用决策。请先运行分析。")
            return False

        # 检查决策是否已初始化
        if not self.block_decisions:
            logger.warning("No decision data available. All blocks will be kept by default.")
            print("[警告] 没有任何决策信息，将默认保留所有块。")
            # 初始化所有块为 'keep' 以便继续
            self._initialize_decisions() # 初始化为 undecided
            for key in self.block_decisions: # 然后全部设为 keep
                self.block_decisions[key] = 'keep'


        try:
            # 确保输出目录存在
            output_path_obj.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory confirmed/created: {output_path_obj}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {output_path_obj}", exc_info=True)
            print(f"[错误] 无法创建输出目录: {output_path_obj} - {e}")
            return False

        # --- 按原文件组织块数据 ---
        file_to_blocks = collections.defaultdict(list)
        original_paths = set()
        for block_info in self.blocks_data:
            file_path = block_info[0]
            file_to_blocks[file_path].append(block_info)
            original_paths.add(file_path)

        logger.info(f"Applying decisions for {len(original_paths)} original files.")
        processed_files_count = 0
        error_files = []
        total_kept_blocks = 0
        total_deleted_blocks = 0

        # --- 遍历每个原始文件 ---
        # 按文件名排序以获得一致的输出顺序
        for original_file_path in sorted(list(original_paths), key=str):
            logger.debug(f"Processing file: {original_file_path.name}")

            # 获取该文件的所有块，并按原始顺序排序
            # 使用 utils 中的排序键函数
            blocks_for_this_file = sorted(file_to_blocks.get(original_file_path, []), key=utils.sort_blocks_key)

            if not blocks_for_this_file:
                logger.debug(f"No blocks found for {original_file_path.name} after sorting, skipping.")
                continue

            new_content_parts = [] # 存储保留下来的块的文本
            kept_block_count = 0
            deleted_block_count = 0
            key_generation_errors = 0

            # --- 遍历文件中的每个块 ---
            for block_info in blocks_for_this_file:
                file_path, b_index, b_type, b_text = block_info
                decision = 'keep' # 默认保留

                try:
                    # 创建决策键以查找决策
                    abs_path_str = str(file_path.resolve())
                    key = utils.create_decision_key(abs_path_str, b_index, b_type)
                    # 获取该块的决策，如果找不到键，则默认为 'keep'
                    # (如果之前没有决策，会保持上面设置的 'keep')
                    decision = self.block_decisions.get(key, 'keep')
                except Exception as e:
                    # 如果创建键失败，也默认保留，并记录警告
                    logger.warning(f"Could not generate decision key for block {file_path.name}#{b_index}: {e}. Defaulting to 'keep'.")
                    key_generation_errors += 1
                    decision = 'keep'

                # --- 根据决策处理块 ---
                if decision != 'delete':
                    # 保留该块
                    formatted_text = b_text # 默认使用原始文本

                    # --- 根据块类型进行必要的格式化，以便重新组合为 Markdown ---
                    # 查找原始 token (这部分比较困难，因为 token 没有存储)
                    # 简单的格式化规则：
                    if b_type == 'block_code':
                        # 尝试从文本猜测是否需要添加 ```
                        lines = b_text.splitlines()
                        if not (lines and lines[0].strip().startswith("```")):
                             # 假设没有语言标识符，因为无法从文本中可靠地恢复
                             formatted_text = f"```\n{b_text}\n```"
                        # else: 保持原样
                    elif b_type == 'list_item':
                        # 为列表项添加 Markdown 标记 (例如 '-')
                        # 假设是无序列表
                        formatted_text = f"- {b_text}"
                    elif b_type == 'block_quote':
                        # 为引用块的每一行添加 '>'
                        lines = b_text.splitlines()
                        formatted_text = "\n".join([f"> {line}" for line in lines])
                    # 其他类型 (paragraph, table, block_html) 保持原样

                    new_content_parts.append(formatted_text)
                    kept_block_count += 1
                else:
                    # 删除该块
                    deleted_block_count += 1
                    logger.debug(f"Deleting block: Index {b_index} in {original_file_path.name} (Key: {key})")

            if key_generation_errors > 0:
                 logger.warning(f"Encountered {key_generation_errors} key generation errors while processing {original_file_path.name}.")

            # --- 组合新内容并写入文件 ---
            # 使用两个换行符分隔块，模拟 Markdown 的段落间距
            new_content = "\n\n".join(new_content_parts)
            # 构建输出文件名
            output_filename = f"{original_file_path.stem}{constants.DEFAULT_OUTPUT_SUFFIX}{original_file_path.suffix}"
            output_filepath = self.output_dir / output_filename # 使用 self.output_dir

            logger.debug(f"Writing output file: {output_filepath.name} (Kept: {kept_block_count}, Deleted: {deleted_block_count})")
            total_kept_blocks += kept_block_count
            total_deleted_blocks += deleted_block_count

            try:
                # 将新内容写入输出文件
                output_filepath.write_text(new_content, encoding='utf-8')
                logger.info(f"Successfully wrote output file: {output_filepath.name}")
                print(f" [+] 已生成: {output_filepath.name} (保留 {kept_block_count} 块, 删除 {deleted_block_count} 块)")
                processed_files_count += 1

                # --- 在 Debug 模式下显示生成的去重文件内容 ---
                if logger.level == logging.DEBUG:
                    try:
                        written_content = output_filepath.read_text(encoding='utf-8')
                        print(f"\n--- 生成的文件内容 (Debug): {output_filepath.name} ---")
                        # 限制预览长度
                        print(written_content[:1000] + ("..." if len(written_content) > 1000 else ""))
                        print("--------------------------------------------------")
                    except Exception as read_err:
                        logger.warning(f"无法读取刚生成的去重文件以供调试显示: {read_err}")
                # -------------------------------------------------

            except Exception as e:
                # 记录写入单个文件时的错误
                logger.error(f"Error writing output file: {output_filepath.name}", exc_info=True)
                print(f" [错误] 写入文件 '{output_filepath.name}' 时出错: {e}")
                error_files.append(original_file_path.name)

        # --- 结束应用决策 ---
        logger.info("--- Finished applying decisions ---")
        logger.info(f"Processed {len(original_paths)} original files.")
        logger.info(f"Successfully generated {processed_files_count} deduplicated files.")
        logger.info(f"Total blocks kept: {total_kept_blocks}, Total blocks deleted: {total_deleted_blocks}")

        print(f"\n[*] 共处理了 {len(original_paths)} 个原始文件。")
        print(f"[+] 成功生成了 {processed_files_count} 个去重后的文件到 '{self.output_dir.name}' 目录。") # 使用 self.output_dir
        print(f"[*] 总计保留了 {total_kept_blocks} 个内容块，删除了 {total_deleted_blocks} 个内容块。")

        if error_files:
            logger.error(f"{len(error_files)} files encountered errors during writing: {', '.join(error_files)}")
            print(f"[错误] 以下 {len(error_files)} 个文件在写入时发生错误:\n - " + "\n - ".join(error_files))

        # 如果有任何文件成功写入，或者没有错误发生，则认为操作成功
        return processed_files_count > 0 or not error_files


# --- 菜单函数 ---

# 不再需要这个函数，因为路径是硬编码的或通过参数传递
# def get_input_directory_from_user(): ...

def display_main_menu():
    """显示主菜单选项。(在 argparse 模式下基本不用)"""
    print("\n--- 主菜单 ---")
    print("1. (已通过命令行参数或默认值指定路径并运行分析)") # 修改提示
    print("2. 退出程序")
    print("------------")

def display_post_analysis_menu(tool_instance):
    """显示分析完成后的菜单选项。"""
    print("\n--- 分析后菜单 ---")
    # 显示当前处理的文件夹和决策/输出路径
    if tool_instance.input_dir:
        # 检查是否使用了默认开发路径
        is_default_input = str(tool_instance.input_dir) == DEV_DEFAULT_INPUT_DIR
        input_source = "(默认开发路径)" if is_default_input else "(来自命令行参数)"
        print(f"当前文件夹: {tool_instance.input_dir} {input_source}")
    else:
        print("当前文件夹: (未设置)")
    print(f"决策文件: {tool_instance.decision_file.name} (在 {tool_instance.decision_file.parent})")
    print(f"输出目录: {tool_instance.output_dir.name} (在 {tool_instance.output_dir.parent})")
    print("-" * 20) # 分隔线

    print("1. 查看并处理 MD5 精确重复项")
    # 根据是否跳过语义分析显示选项
    if not tool_instance.skip_semantic:
        # 检查是否有语义分析结果或是否可用
        if not semantic_analyzer.SENTENCE_TRANSFORMERS_AVAILABLE:
             print("2. (语义分析库不可用)")
        else:
             print("2. 查看并处理语义相似项")
    else:
        print("2. (语义分析已跳过)")

    print(f"3. 从文件加载决策")
    print(f"4. 保存当前决策到文件")
    print(f"5. 应用决策 (生成去重文件)")
    print("6. 重新运行当前文件夹分析")
    print("7. 退出程序") # 调整编号
    print("------------------")

# --- 常量定义 (移到这里以便 argparse 使用默认值) ---
# 开发阶段的默认路径 (硬编码)
DEV_DEFAULT_INPUT_DIR = "/Users/hansen/Documents/Knowledge Distillation/Test_SourceFiles"
DEV_DEFAULT_OUTPUT_DIR = "/Users/hansen/Documents/Knowledge Distillation/Test_other/kd_results"
DEV_DEFAULT_DECISION_DIR = "/Users/hansen/Documents/Knowledge Distillation/Test_other"
DEV_DEFAULT_DECISION_FILE = os.path.join(DEV_DEFAULT_DECISION_DIR, constants.DEFAULT_DECISION_FILENAME)


# --- 主程序入口 ---
# 检查是否作为主程序运行
if __name__ == "__main__":
    print("欢迎使用知识蒸馏 KD Tool (命令行交互版 V2.3 - Argparse)")
    print("=====================================================")

    # --- 设置 ArgumentParser ---
    parser = argparse.ArgumentParser(description="知识蒸馏 KD Tool: 查找并处理 Markdown 文件中的重复内容块。")

    # 定义命令行参数
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        default=DEV_DEFAULT_INPUT_DIR, # 设置默认开发路径
        help=f"包含 Markdown 文件的输入文件夹路径 (默认: '{DEV_DEFAULT_INPUT_DIR}')"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=DEV_DEFAULT_OUTPUT_DIR, # 设置默认开发路径
        help=f"保存去重后文件的输出文件夹路径 (默认: '{DEV_DEFAULT_OUTPUT_DIR}')"
    )
    parser.add_argument(
        "-d", "--decision-file",
        type=str,
        default=DEV_DEFAULT_DECISION_FILE, # 设置默认开发路径
        help=f"加载和保存决策的 JSON 文件路径 (默认: '{DEV_DEFAULT_DECISION_FILE}')"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=constants.DEFAULT_SIMILARITY_THRESHOLD,
        help=f"语义相似度阈值 (默认: {constants.DEFAULT_SIMILARITY_THRESHOLD})"
    )
    parser.add_argument(
        "--skip-semantic",
        action="store_true", # 表示参数存在时值为 True
        help="跳过语义相似度分析"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用详细的调试日志输出"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # --- 初始化日志和工具 ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    utils.setup_logger(log_level)
    logger.info("KD Tool started.")
    logger.debug(f"Parsed arguments: {args}") # 记录解析的参数

    # 使用解析出的参数实例化 KDToolCLI
    kd_tool = KDToolCLI(
        output_dir_path=args.output_dir,
        decision_file_path=args.decision_file,
        skip_semantic=args.skip_semantic,
        similarity_threshold=args.threshold
    )

    # --- 使用解析出的输入目录进行分析 ---
    analysis_run_successfully = False
    input_directory = args.input_dir
    logger.info(f"Using input directory: {input_directory}")

    # 检查输入目录是否存在
    input_path_obj = Path(input_directory)
    if not input_path_obj.is_dir():
         logger.error(f"Input directory specified does not exist or is not a directory: {input_directory}")
         print(f"[错误] 指定的输入目录不存在或不是一个有效的文件夹: {input_directory}")
         sys.exit(1)

    # 设置输入目录并运行分析
    if kd_tool.set_input_directory(input_directory):
        analysis_run_successfully = kd_tool.run_analysis()
    else:
        logger.critical(f"Failed to set input directory: {input_directory}. Exiting.")
        print(f"[错误] 无法设置输入目录 '{input_directory}'。程序退出。")
        sys.exit(1) # 设置失败则直接退出

    # --- 如果分析成功，直接进入分析后菜单循环 ---
    if analysis_run_successfully:
        logger.info("Analysis completed. Entering post-analysis menu.")
        while True: # 分析后菜单循环
            display_post_analysis_menu(kd_tool) # 调用显示菜单函数
            sub_choice = input("请输入选项编号: ").strip()
            logger.debug(f"Post-analysis menu choice: '{sub_choice}'")

            if sub_choice == '1':
                logger.info("User chose option 1: Review MD5 Duplicates")
                kd_tool.md5_analyzer.review_md5_duplicates_interactive()
            elif sub_choice == '2':
                if not kd_tool.skip_semantic and semantic_analyzer.SENTENCE_TRANSFORMERS_AVAILABLE:
                    logger.info("User chose option 2: Review Semantic Duplicates")
                    kd_tool.semantic_analyzer.review_semantic_duplicates_interactive()
                elif kd_tool.skip_semantic:
                    logger.info("User chose option 2, but semantic analysis was skipped.")
                    print("[信息] 语义分析已被跳过。")
                else:
                     logger.info("User chose option 2, but semantic library is unavailable.")
                     print("[信息] 语义分析库 (sentence-transformers) 不可用。")
            elif sub_choice == '3':
                logger.info("User chose option 3: Load Decisions")
                kd_tool.load_decisions()
            elif sub_choice == '4':
                logger.info("User chose option 4: Save Decisions")
                kd_tool.save_decisions()
            elif sub_choice == '5':
                logger.info("User chose option 5: Apply Decisions")
                kd_tool.apply_decisions()
            elif sub_choice == '6': # 重新分析当前文件夹
                logger.info("User chose option 6: Rerun analysis for current directory")
                if kd_tool.input_dir: # 确保目录已设置
                     print(f"[*] 正在重新分析文件夹: {kd_tool.input_dir} ...")
                     analysis_run_successfully = kd_tool.run_analysis() # 重新运行分析
                     if not analysis_run_successfully:
                          print("[!] 重新分析未能成功完成。")
                     # 重新分析后继续显示分析后菜单
                else:
                     # 理论上在 argparse 模式下不太可能发生
                     print("[错误] 当前未设置输入文件夹，无法重新分析。")
                     logger.error("Attempted to rerun analysis but input directory was not set.")

            elif sub_choice == '7': # 退出程序 (原选项 8)
                logger.info("User chose option 7: Exit Program from post-analysis menu")
                print("[*] 退出程序。")
                sys.exit(0)
            else:
                logger.warning(f"Invalid post-analysis menu choice: '{sub_choice}'")
                print("[错误] 无效的选项，请重新输入。")
    else:
        # 如果分析失败
        logger.error("Analysis failed to complete successfully. Exiting.")
        print("[!] 分析未能成功完成，请检查日志信息。程序退出。")
        sys.exit(1) # 分析失败也退出

    # 如果从分析后菜单退出 (理论上只有通过选项 7 退出)
    logger.info("KD Tool finished.")
    sys.exit(0)

