# KD_Tool_CLI/knowledge_distiller_kd/core/kd_tool_CLI.py
"""
知识蒸馏工具CLI模块。

此模块提供命令行界面，用于执行知识蒸馏任务。
"""

import os
import sys
import json
import logging
from collections import defaultdict
import collections
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any, DefaultDict
import argparse
import traceback # 添加导入

from knowledge_distiller_kd.core.error_handler import (
    FileOperationError,
    ConfigurationError,
    handle_error,
    safe_file_operation,
    validate_file_path
)
from knowledge_distiller_kd.core.utils import (
    setup_logger,
    create_decision_key,
    parse_decision_key,
    # extract_text_from_children, # 可能不再需要，因为处理移到 ContentBlock
    display_block_preview,
    get_markdown_parser,
    sort_blocks_key,
    logger # 使用 utils 中配置好的 logger
)
from knowledge_distiller_kd.core.md5_analyzer import MD5Analyzer
from knowledge_distiller_kd.core.semantic_analyzer import SemanticAnalyzer
from knowledge_distiller_kd.core.document_processor import (
    ContentBlock,
    process_file,
    process_directory,
    DocumentProcessingError
)
from knowledge_distiller_kd.core import constants

# [DEPENDENCIES] - 保持更新
# 1. Python Standard Library: sys, os, json, logging, collections, pathlib, argparse, traceback
# 2. 需要安装：unstructured, sentence-transformers, torch, numpy, PyYAML, colorama, markdown (mistune 可能不再直接需要，但unstructured可能依赖)
# 3. 同项目模块: constants, error_handler, utils, md5_analyzer, semantic_analyzer, document_processor

# 开发环境默认路径 (如果需要，可以根据实际情况调整或移除)
DEV_DEFAULT_INPUT_DIR = "input"
DEV_DEFAULT_OUTPUT_DIR = "output"
DEV_DEFAULT_DECISION_DIR = "decisions"
DEV_DEFAULT_DECISION_FILE = os.path.join(DEV_DEFAULT_DECISION_DIR, "decisions.json")

# --- 核心逻辑类 ---
class KDToolCLI:
    """
    知识蒸馏工具的主控制类，协调文件读取、解析、分析和决策应用。
    """

    def __init__(
        self,
        input_dir: Optional[Union[str, Path]] = None,
        decision_file: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        skip_semantic: bool = False,
        similarity_threshold: float = constants.DEFAULT_SIMILARITY_THRESHOLD # 使用常量
    ) -> None:
        """
        初始化知识蒸馏工具。
        """
        logger.info("Initializing KDToolCLI...")
        try:
            # 路径处理与验证
            self.input_dir: Optional[Path] = None
            if input_dir:
                self.input_dir = validate_file_path(Path(input_dir), must_exist=True)
                if not self.input_dir.is_dir():
                     raise ConfigurationError(f"Input path is not a directory: {self.input_dir}")

            # 使用 Path 对象处理路径，并在需要时创建目录
            self.decision_file: Path = Path(decision_file or constants.DEFAULT_DECISION_FILE).resolve()
            self.output_dir: Path = Path(output_dir or constants.DEFAULT_OUTPUT_DIR).resolve()

            # 确保目录存在 (推迟到实际使用时创建，避免空目录)
            # self.decision_file.parent.mkdir(parents=True, exist_ok=True) # 在保存时创建
            # self.output_dir.mkdir(parents=True, exist_ok=True) # 在应用决策时创建

            # 配置参数
            self.skip_semantic: bool = skip_semantic
            # 确保阈值在有效范围 [0.0, 1.0]
            self.similarity_threshold: float = max(0.0, min(1.0, similarity_threshold))

            # 初始化数据存储
            self.blocks_data: List[ContentBlock] = []
            self.block_decisions: Dict[str, str] = {}
            self._decisions_loaded: bool = False
            self._analysis_completed: bool = False # 添加分析完成状态

            # 初始化分析器实例 (在 _reset_state 中进行)
            self.md5_analyzer: Optional[MD5Analyzer] = None
            self.semantic_analyzer: Optional[SemanticAnalyzer] = None
            self._reset_state() # 在初始化时就重置一次状态并创建分析器

            logger.info("知识蒸馏工具初始化完成")
            logger.debug(f"  输入目录: {self.input_dir}")
            logger.debug(f"  决策文件: {self.decision_file}")
            logger.debug(f"  输出目录: {self.output_dir}")
            logger.debug(f"  跳过语义分析: {self.skip_semantic}")
            logger.debug(f"  相似度阈值: {self.similarity_threshold}")

        except Exception as e:
            handle_error(e, "初始化知识蒸馏工具")
            # 提供更具体的错误信息或重新抛出自定义错误
            raise ConfigurationError(f"初始化知识蒸馏工具失败: {e}") from e

    def _reset_state(self) -> None:
        """
        重置工具状态，用于开始新的分析或设置新目录后。
        """
        logger.debug("Resetting internal state...")
        self.blocks_data.clear()
        self.block_decisions.clear()
        self._decisions_loaded = False
        self._analysis_completed = False # 重置分析状态
        # 创建（或重新创建）分析器实例
        self.md5_analyzer = MD5Analyzer(self)
        # 确保传递正确的阈值
        self.semantic_analyzer = SemanticAnalyzer(self, self.similarity_threshold)
        logger.debug("Analyzers re-initialized.")


    def set_input_dir(self, input_dir: Union[str, Path]) -> bool:
        """
        设置输入目录，并重置状态以准备新的分析。

        Args:
            input_dir: 输入目录路径

        Returns:
            bool: 是否成功设置输入目录
        """
        try:
            input_path = Path(input_dir)
            resolved_path = validate_file_path(input_path, must_exist=True)
            if not resolved_path.is_dir():
                 logger.error(f"设置输入目录失败: '{resolved_path}' 不是一个目录。")
                 print(f"[错误] 路径 '{resolved_path}' 不是一个有效的文件夹。")
                 return False

            logger.info(f"设置输入目录: {resolved_path}")
            self.input_dir = resolved_path
            self._reset_state() # 设置新目录后重置状态
            return True
        except FileOperationError as e:
            # 由 validate_file_path 抛出的错误
            handle_error(e, "设置输入目录")
            print(f"[错误] 设置输入目录时出错: {e}")
            return False
        except Exception as e:
            handle_error(e, "设置输入目录")
            print(f"[错误] 设置输入目录时发生意外错误: {e}")
            return False

    # ==================== 新增方法：加载语义模型 ====================
    def _load_semantic_model(self) -> bool:
        """
        尝试加载语义模型。
        """
        if self.skip_semantic:
            logger.info("Semantic analysis is skipped by configuration. Model loading skipped.")
            return True # 跳过也算成功
        if not self.semantic_analyzer:
             logger.error("Semantic analyzer not initialized.")
             return False

        try:
            # 调用 SemanticAnalyzer 的加载方法
            self.semantic_analyzer.load_semantic_model()
            # 检查加载是否成功 (load_semantic_model 失败时会设置 skip_semantic=True)
            if self.semantic_analyzer.model is None and not self.skip_semantic:
                logger.warning("Semantic model loading seems to have failed (model is None).")
                # 根据 load_semantic_model 的逻辑，它失败时会设置 self.skip_semantic = True
                # 所以理论上这里不需要返回 False，但可以加日志
            return True
        except Exception as e:
            handle_error(e, "加载语义模型")
            print(f"[错误] 加载语义模型时出错: {e}")
            # 加载失败，强制跳过后续语义分析
            self.skip_semantic = True
            if self.semantic_analyzer:
                 self.semantic_analyzer.tool.skip_semantic = True # 确保分析器内部状态也同步
            return False
    # =============================================================

    def run_analysis(self) -> bool:
        """
        执行完整的分析流程：读取 -> 解析 -> 初始化决策 -> MD5 -> [加载模型] -> 语义。

        Returns:
            bool: 如果分析成功完成返回 True，否则返回 False
        """
        if not self.input_dir:
            logger.error("Analysis aborted: Input directory not set.")
            print("[错误] 输入文件夹未设置。")
            return False

        logger.info(f"--- Starting analysis for folder: {self.input_dir} ---")
        print(f"\n[*] 开始分析文件夹: {self.input_dir}")

        # 重置状态 (如果需要确保每次运行都是全新的)
        # self._reset_state() # 或者根据需要决定是否重置

        # 确保分析器已创建
        if not self.md5_analyzer or not self.semantic_analyzer:
             logger.error("Analyzers not initialized correctly.")
             print("[错误] 分析器未正确初始化。")
             return False

        # 定义分析步骤
        steps = [
            ("处理文档", self._process_documents),
            ("初始化决策", self._initialize_decisions),
            ("MD5 去重", self.md5_analyzer.find_md5_duplicates),
            # ==================== 修改点：添加加载模型步骤 ====================
            ("加载语义模型", self._load_semantic_model),
            # ==============================================================
            ("语义去重", self.semantic_analyzer.find_semantic_duplicates) # 使用实例方法
        ]

        analysis_successful = True
        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}...")
            print(f"\n[*] 正在执行: {step_name}...")

            # 跳过语义相关步骤（如果配置）
            if step_name in ["加载语义模型", "语义去重"] and self.skip_semantic:
                logger.info(f"Skipping step '{step_name}' as semantic analysis is disabled.")
                print(f"[*] 已跳过步骤: {step_name}。")
                continue

            try:
                # 执行步骤
                result = step_func()
                # 对于 find_... 和 load_... 方法，它们内部处理错误并可能返回 False 或 None
                # 但 run_analysis 应该继续，除非是关键步骤失败
                if result is False and step_name in ["处理文档", "初始化决策"]: # 关键步骤失败则中止
                    logger.error(f"Analysis stopped: Critical step '{step_name}' failed.")
                    print(f"[错误] 关键步骤 '{step_name}' 失败，分析中止。")
                    analysis_successful = False
                    break # 中止循环

                logger.info(f"Step completed: {step_name}")
                print(f"[*] {step_name} 完成。")

            except Exception as e:
                logger.error(f"Error during step '{step_name}': {e}", exc_info=True)
                print(f"[错误] 在步骤 '{step_name}' 时出错: {e}")
                analysis_successful = False
                break # 发生意外错误，中止循环

        if analysis_successful:
             self._analysis_completed = True
             logger.info("Analysis process completed.")
             print("\n[*] 分析流程执行完毕。")
        else:
             self._analysis_completed = False # 标记分析未成功完成
             logger.error("Analysis process failed or was aborted.")
             print("\n[!] 分析流程未能成功完成。")

        return analysis_successful


    def _process_documents(self) -> bool:
        """
        处理输入目录中的所有文档。

        Returns:
            bool: 如果成功处理了至少一个文件并提取了块返回 True，否则返回 False
        """
        if not self.input_dir:
            logger.error("Cannot process documents: Input directory not set.")
            return False
        try:
            # 调用 document_processor 中的函数
            results = process_directory(self.input_dir, recursive=True) # 假设默认递归
            if not results:
                logger.warning(f"在目录 {self.input_dir} 中未找到可处理的 Markdown 文件或未能提取内容。")
                print(f"[*] 在目录 {self.input_dir} 中未找到有效的 Markdown 文件或内容。")
                self.blocks_data = [] # 确保清空
                return True # 没有文件也算步骤成功，但没有块

            # 清空旧数据并填充新数据
            self.blocks_data.clear()
            for file_path, blocks in results.items():
                self.blocks_data.extend(blocks)

            total_blocks = len(self.blocks_data)
            if total_blocks > 0:
                 logger.info(f"成功处理 {len(results)} 个文件，共提取 {total_blocks} 个内容块。")
                 print(f"\n[*] 文档处理完成: 成功处理 {len(results)} 个文件，提取 {total_blocks} 个块。")
                 return True
            else:
                 logger.warning(f"处理了 {len(results)} 个文件，但未提取到任何内容块。")
                 print(f"\n[*] 文档处理完成: 处理了 {len(results)} 个文件，但未提取到内容块。")
                 return True # 没有块也算步骤成功

        except DocumentProcessingError as e:
            handle_error(e, "处理文档")
            print(f"[错误] 处理文档时出错: {e}")
            return False
        except Exception as e:
            handle_error(e, "处理文档时发生意外错误")
            print(f"[错误] 处理文档时发生意外错误: {e}")
            return False

    def _initialize_decisions(self) -> bool:
        """
        初始化所有内容块的决策状态。

        Returns:
            bool: 总是返回 True，除非发生严重错误。初始化零个决策也算成功。
        """
        if not self.blocks_data:
             logger.info("No blocks found, skipping decision initialization.")
             print("[*] 没有内容块，跳过决策初始化。")
             self.block_decisions.clear()
             return True

        logger.info(f"Initializing decisions for {len(self.blocks_data)} blocks...")
        print(f"\n[*] 正在初始化 {len(self.blocks_data)} 个块的决策...")

        self.block_decisions.clear()
        initialized_count = 0
        error_count = 0

        for i, block in enumerate(self.blocks_data):
            try:
                key = create_decision_key(block.file_path, block.block_id, block.block_type)
                self.block_decisions[key] = constants.DECISION_UNDECIDED
                initialized_count += 1
                # if (i + 1) % 100 == 0: # 减少打印频率
                #      logger.debug(f"Initializing decisions progress: {i+1}/{len(self.blocks_data)}")
            except Exception as e:
                error_count += 1
                logger.error(f"无法为块创建决策键: {block.file_path} # {block.block_id} - {e}", exc_info=False) # 不打印完整堆栈

        logger.info(f"成功初始化 {initialized_count} 个块的决策。{error_count} 个块失败。")
        print(f"[*] 决策初始化完成: {initialized_count} 个成功，{error_count} 个失败。")
        # 即使有错误，也认为此步骤逻辑上完成了它能做的
        return True

    def load_decisions(self) -> bool:
        """
        从决策文件加载决策。会覆盖内存中现有的决策。

        Returns:
            bool: 是否成功加载了至少一个决策记录。
        """
        if not self.decision_file.exists():
            logger.warning(f"决策文件不存在: {self.decision_file}。跳过加载。")
            print(f"[*] 决策文件 '{self.decision_file.name}' 不存在，跳过加载。")
            return False

        logger.info(f"开始从文件加载决策: {self.decision_file}")
        loaded_count = 0
        error_count = 0
        try:
            with open(self.decision_file, 'r', encoding='utf-8') as f:
                try:
                    decisions_from_file = json.load(f)
                    if not isinstance(decisions_from_file, list):
                         logger.error("决策文件格式错误：顶层应为列表。")
                         print("[错误] 决策文件格式错误，无法加载。")
                         return False
                except json.JSONDecodeError as e:
                    logger.error(f"无法解析决策文件 {self.decision_file}: {e}")
                    print(f"[错误] 无法解析决策文件 '{self.decision_file.name}': {e}")
                    return False

            # 加载前可以不清空 self.block_decisions，而是进行合并/更新
            # 这里选择清空以完全反映文件状态
            self.block_decisions.clear()
            logger.info(f"成功读取 {len(decisions_from_file)} 条决策记录，现在应用...")

            for record in decisions_from_file:
                if not isinstance(record, dict):
                    logger.warning(f"跳过格式错误的决策记录: {record}")
                    error_count += 1
                    continue

                file_str = record.get('file')
                block_id = record.get('block_id')
                block_type = record.get('type')
                decision = record.get('decision')

                if not all([file_str, block_id, block_type, decision]):
                    logger.warning(f"跳过不完整的决策记录: {record}")
                    error_count += 1
                    continue

                if decision not in [constants.DECISION_KEEP, constants.DECISION_DELETE, constants.DECISION_UNDECIDED]:
                     logger.warning(f"跳过包含无效决策值的记录 ('{decision}'): {record}")
                     error_count += 1
                     continue

                try:
                    # 尝试将文件路径解析为绝对路径以匹配内部表示
                    # 注意：这假设决策文件中的路径是相对于某个基准或已经是绝对路径
                    # 更健壮的方式可能是在保存时就存绝对路径，或提供基准路径选项
                    abs_file_path = Path(file_str)
                    if not abs_file_path.is_absolute():
                         # 如果不是绝对路径，尝试相对于 input_dir 解析？
                         # 或者假设它是相对于决策文件所在目录？
                         # 暂时假设它是可直接使用的路径字符串，create_decision_key会处理
                         logger.debug(f"Decision record file path '{file_str}' is not absolute. Using as is.")
                         pass # create_decision_key 会尝试 resolve

                    key = create_decision_key(str(file_str), block_id, block_type)
                    self.block_decisions[key] = decision
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"处理决策记录时出错 {record}: {e}")
                    error_count += 1
                    continue

            logger.info(f"决策加载完成: 成功应用 {loaded_count} 条决策，{error_count} 条记录处理失败。")
            print(f"[*] 决策加载完成: 应用 {loaded_count} 条，失败 {error_count} 条。")
            self._decisions_loaded = loaded_count > 0
            return self._decisions_loaded

        except FileNotFoundError:
             logger.error(f"决策文件在读取过程中消失: {self.decision_file}")
             print(f"[错误] 决策文件 '{self.decision_file.name}' 找不到了。")
             return False
        except Exception as e:
            handle_error(e, f"加载决策文件 {self.decision_file}")
            print(f"[错误] 加载决策时发生意外错误: {e}")
            return False

    # 稍微修改 save_decisions 以便处理可能的 Path 对象
    def save_decisions(self) -> bool:
        """
        将当前内存中的决策保存到文件。
        """
        if not self.block_decisions:
             logger.warning("没有决策可保存。")
             print("[!] 没有决策可保存。")
             return False # 或者返回 True 表示“无需保存”？返回 False 表示未执行保存。

        logger.info(f"开始保存 {len(self.block_decisions)} 条决策到文件: {self.decision_file}")
        decisions_to_save = []
        saved_count = 0
        error_count = 0

        for key, decision in self.block_decisions.items():
            try:
                file_path_str, block_id, block_type = parse_decision_key(key)
                if file_path_str is None: # 解析失败
                    logger.warning(f"无法解析决策键 '{key}'，跳过保存。")
                    error_count += 1
                    continue

                # 考虑是否将绝对路径转为相对路径保存（相对于 input_dir）
                # 暂时保存绝对路径字符串
                record = {
                    'file': file_path_str, # 保存字符串
                    'block_id': str(block_id), # 确保 block_id 是字符串
                    'type': block_type,
                    'decision': decision
                }
                decisions_to_save.append(record)
                saved_count += 1
            except Exception as e:
                 logger.error(f"处理决策键 '{key}' 进行保存时出错: {e}")
                 error_count += 1
                 continue

        if not decisions_to_save:
             logger.error("没有有效的决策记录可供保存。")
             print("[错误] 没有有效的决策可供保存。")
             return False

        try:
            # 确保目录存在
            self.decision_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.decision_file, 'w', encoding='utf-8') as f:
                json.dump(decisions_to_save, f, indent=2, ensure_ascii=False)

            logger.info(f"成功保存 {saved_count} 条决策到文件: {self.decision_file}。{error_count} 条处理失败。")
            print(f"[*] 成功保存 {saved_count} 条决策。{error_count} 条失败。")
            return True

        except Exception as e:
            handle_error(e, f"保存决策到文件 {self.decision_file}")
            print(f"[错误] 保存决策时发生意外错误: {e}")
            return False

    # 应用决策（保持不变，它处理 ContentBlock）
    @safe_file_operation
    def apply_decisions(self) -> bool:
        """
        应用决策，生成去重后的文件。

        Returns:
            bool: 如果成功应用决策并生成了至少一个输出文件返回 True，否则返回 False
        """
        if not self._analysis_completed:
             logger.warning("无法应用决策：分析尚未成功完成。")
             print("[警告] 请先成功运行分析，再应用决策。")
             return False
        if not self.blocks_data:
            logger.warning("没有内容块可供处理以应用决策。")
            print("[警告] 没有可处理的内容块。")
            return True  # 没有块也算成功应用（没啥可做的）

        logger.info(f"开始应用 {len(self.block_decisions)} 条决策到 {len(set(b.file_path for b in self.blocks_data))} 个文件的内容块...")
        print(f"[*] 正在应用决策以生成输出文件...")

        # 按文件路径分组块数据
        # 使用原始 ContentBlock 对象的 file_path (应该是绝对路径)
        files_blocks: DefaultDict[str, List[ContentBlock]] = defaultdict(list)
        for block in self.blocks_data:
            files_blocks[block.file_path].append(block)

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录 '{self.output_dir}' 已确认存在。")

        processed_files_count = 0
        error_files = []
        total_files_to_process = len(files_blocks)

        for i, (file_path_str, blocks_in_file) in enumerate(files_blocks.items()):
            original_path = Path(file_path_str)
            logger.debug(f"Processing file {i+1}/{total_files_to_process}: {original_path.name}")
            # 构建输出文件名
            output_name = original_path.stem + constants.DEFAULT_OUTPUT_SUFFIX + original_path.suffix
            output_path = self.output_dir / output_name

            kept_blocks_count = 0
            try:
                with open(output_path, 'w', encoding=constants.DEFAULT_ENCODING) as f_out:
                     # 按原始顺序（假设 blocks_in_file 保持了顺序）写入保留的块
                     # 为了更保险，可以先按 block_id 排序（如果 ID 是有序的）
                     # blocks_in_file.sort(key=lambda b: b.block_id) # 取决于 block_id 格式

                     for block in blocks_in_file:
                         try:
                             key = create_decision_key(block.file_path, block.block_id, block.block_type)
                             decision = self.block_decisions.get(key, constants.DECISION_UNDECIDED) # 默认保留未决策的

                             if decision != constants.DECISION_DELETE:
                                 f_out.write(block.original_text)
                                 # 添加适当的分隔符，例如两个换行符
                                 f_out.write('\n\n')
                                 kept_blocks_count += 1
                         except Exception as e_inner:
                              logger.error(f"处理块 {block.block_id} (文件 {original_path.name}) 时出错: {e_inner}")
                              # 选择跳过这个块或停止整个文件处理
                              continue # 跳过这个块

                logger.info(f"成功写入 {kept_blocks_count} 个块到输出文件: {output_path.name}")
                processed_files_count += 1

            except Exception as e:
                logger.error(f"写入输出文件 {output_path} 时失败: {e}", exc_info=True)
                error_files.append(original_path.name)
                # 可以在这里删除可能已部分写入的文件
                # if output_path.exists(): output_path.unlink()

        logger.info(f"决策应用完成: 成功处理 {processed_files_count}/{total_files_to_process} 个文件。")
        print(f"\n[*] 决策应用完成: 成功生成 {processed_files_count} 个输出文件到 '{self.output_dir}'。")
        if error_files:
             print(f"[警告] 以下原始文件的输出处理失败: {', '.join(error_files)}")

        return processed_files_count > 0


    # --- 菜单显示方法 ---
    def display_main_menu(self) -> None:
        """显示主菜单"""
        print("\n" + "=" * 60)
        print("知识蒸馏工具 (Knowledge Distiller)".center(60))
        print(f"版本 {constants.VERSION}".center(60))
        print("=" * 60)
        print("\n当前配置:")
        print(f"  输入目录: {self.input_dir if self.input_dir else '未设置'}")
        print(f"  决策文件: {self.decision_file}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  语义分析: {'启用' if not self.skip_semantic else '跳过'}")
        if not self.skip_semantic:
            print(f"  相似度阈值: {self.similarity_threshold}")
        print("\n请选择操作:")
        print("  1. 设置输入目录")
        print("  2. 运行分析 (MD5 + 语义)")
        print("  3. 仅运行 MD5 分析")
        print("  4. 加载决策")
        print("  5. 保存决策")
        print("  6. 应用决策")
        print("  7. 查看重复块 (需先运行分析)")
        print("  8. 配置选项")
        print("  q. 退出程序")
        print("-" * 60)

    def display_duplicates_menu(self) -> None:
        """显示重复块查看菜单"""
        if not self._analysis_completed:
            print("\n[!] 请先成功运行分析，才能查看重复块。")
            return

        md5_groups = self.md5_analyzer.md5_duplicates if self.md5_analyzer else []
        sem_groups = self.semantic_analyzer.semantic_duplicates if self.semantic_analyzer and not self.skip_semantic else []

        print("\n" + "=" * 50)
        print("重复块查看".center(50))
        print("=" * 50)
        print("\n分析结果统计:")
        print(f"  MD5 精确重复: {len(md5_groups)} 组")
        if not self.skip_semantic:
             print(f"  语义相似重复: {len(sem_groups)} 对")
        else:
             print("  语义分析已跳过")
        print("\n请选择:")
        print("  1. 查看/处理 MD5 重复项")
        if not self.skip_semantic:
             print("  2. 查看/处理 语义相似项")
        print("  b. 返回上一级菜单")
        print("-" * 50)


    def handle_duplicates_view(self) -> None:
        """处理重复块查看的用户交互"""
        if not self._analysis_completed:
            print("\n[!] 请先成功运行分析，才能查看重复块。")
            return
        if not self.md5_analyzer or not self.semantic_analyzer:
             print("[错误] 分析器未初始化。")
             return

        while True:
             self.display_duplicates_menu()
             if not self._analysis_completed: break # 如果在显示时发现分析未完成，退出

             choice = input("请输入选项: ").strip().lower()

             if choice == '1':
                  self.md5_analyzer.review_md5_duplicates_interactive()
             elif choice == '2' and not self.skip_semantic:
                  self.semantic_analyzer.review_semantic_duplicates_interactive()
             elif choice == 'b':
                  break
             else:
                  print("[错误] 无效的选项。")

    def display_config_menu(self) -> None:
        """显示配置选项菜单"""
        print("\n" + "=" * 50)
        print("配置选项".center(50))
        print("=" * 50)
        print(f"  1. 切换语义分析状态 (当前: {'启用' if not self.skip_semantic else '跳过'})")
        print(f"  2. 设置语义相似度阈值 (当前: {self.similarity_threshold:.2f})")
        print(f"  3. 设置决策文件路径 (当前: {self.decision_file})")
        print(f"  4. 设置输出目录路径 (当前: {self.output_dir})")
        print("  b. 返回主菜单")
        print("-" * 50)

    def handle_config(self) -> None:
        """处理配置选项交互"""
        while True:
            self.display_config_menu()
            choice = input("请输入选项: ").strip().lower()

            if choice == '1':
                self.skip_semantic = not self.skip_semantic
                if self.semantic_analyzer:
                     self.semantic_analyzer.tool.skip_semantic = self.skip_semantic # 同步到分析器
                print(f"[*] 语义分析已 {'跳过' if self.skip_semantic else '启用'}.")
            elif choice == '2':
                try:
                    new_threshold_str = input(f"请输入新的相似度阈值 (0.0 到 1.0 之间，当前: {self.similarity_threshold:.2f}): ")
                    new_threshold = float(new_threshold_str)
                    if 0.0 <= new_threshold <= 1.0:
                         self.similarity_threshold = new_threshold
                         if self.semantic_analyzer:
                              self.semantic_analyzer.similarity_threshold = new_threshold # 同步到分析器
                         print(f"[*] 相似度阈值已更新为: {self.similarity_threshold:.2f}")
                    else:
                         print("[错误] 阈值必须在 0.0 到 1.0 之间。")
                except ValueError:
                     print("[错误] 无效的输入，请输入一个数字。")
            elif choice == '3':
                 new_path_str = input(f"请输入新的决策文件路径 (当前: {self.decision_file}): ").strip()
                 if new_path_str:
                     try:
                         new_path = Path(new_path_str).resolve()
                         # 不在此处创建目录，仅更新路径
                         self.decision_file = new_path
                         print(f"[*] 决策文件路径已更新为: {self.decision_file}")
                     except Exception as e:
                          print(f"[错误] 无效的文件路径: {e}")
                 else:
                      print("[!] 输入为空，路径未更改。")
            elif choice == '4':
                 new_path_str = input(f"请输入新的输出目录路径 (当前: {self.output_dir}): ").strip()
                 if new_path_str:
                     try:
                         new_path = Path(new_path_str).resolve()
                         # 不在此处创建目录，仅更新路径
                         self.output_dir = new_path
                         print(f"[*] 输出目录路径已更新为: {self.output_dir}")
                     except Exception as e:
                          print(f"[错误] 无效的目录路径: {e}")
                 else:
                      print("[!] 输入为空，路径未更改。")
            elif choice == 'b':
                 break
            else:
                 print("[错误] 无效的选项。")


    def run_interactive(self) -> None:
        """运行交互式命令行界面"""
        logger.info("Starting interactive mode...")
        while True:
            self.display_main_menu()
            choice = input("请输入选项编号: ").strip().lower()

            if choice == '1':
                 input_dir_str = input("请输入输入目录路径: ").strip()
                 if input_dir_str:
                      self.set_input_dir(input_dir_str)
                 else:
                      print("[!] 输入为空，目录未更改。")
            elif choice == '2': # 运行完整分析
                 if not self.input_dir: print("[错误] 请先设置输入目录。"); continue
                 # 保存当前 skip_semantic 状态
                 original_skip_semantic = self.skip_semantic
                 self.skip_semantic = False # 确保运行语义分析
                 if self.semantic_analyzer: self.semantic_analyzer.tool.skip_semantic = False
                 print("\n[*] 开始运行完整分析 (MD5 + 语义)...")
                 self.run_analysis()
                 # 恢复原始 skip_semantic 状态？或者让用户在配置中管理
                 # self.skip_semantic = original_skip_semantic
                 # if self.semantic_analyzer: self.semantic_analyzer.tool.skip_semantic = original_skip_semantic

            elif choice == '3': # 仅运行 MD5
                 if not self.input_dir: print("[错误] 请先设置输入目录。"); continue
                 original_skip_semantic = self.skip_semantic
                 self.skip_semantic = True # 强制跳过语义
                 if self.semantic_analyzer: self.semantic_analyzer.tool.skip_semantic = True
                 print("\n[*] 开始仅运行 MD5 分析...")
                 self.run_analysis()
                 self.skip_semantic = original_skip_semantic # 恢复
                 if self.semantic_analyzer: self.semantic_analyzer.tool.skip_semantic = original_skip_semantic

            elif choice == '4': # 加载决策
                 self.load_decisions()
            elif choice == '5': # 保存决策
                 self.save_decisions()
            elif choice == '6': # 应用决策
                 # 确保输出目录存在
                 try:
                      self.output_dir.mkdir(parents=True, exist_ok=True)
                      logger.info(f"确保输出目录存在: {self.output_dir}")
                      self.apply_decisions()
                 except Exception as e:
                      handle_error(e, f"创建输出目录或应用决策时出错: {self.output_dir}")
                      print(f"[错误] 创建输出目录或应用决策时出错: {e}")

            elif choice == '7': # 查看重复块
                 self.handle_duplicates_view()
            elif choice == '8': # 配置选项
                 self.handle_config()
            elif choice == 'q':
                 logger.info("Exiting interactive mode.")
                 print("\n[*] 正在退出程序...")
                 break
            else:
                 print("\n[错误] 无效的选项，请重新输入。")


    @staticmethod
    def parse_args() -> argparse.Namespace:
        """解析命令行参数"""
        parser = argparse.ArgumentParser(
            description="知识蒸馏工具 KD Tool: 查找并处理 Markdown 文件中的重复内容块。",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
  # 使用交互模式 (推荐)
  python -m knowledge_distiller_kd.core.kd_tool_CLI

  # 直接运行分析并进入分析后菜单 (需要指定输入目录)
  python -m knowledge_distiller_kd.core.kd_tool_CLI -i ./input

  # 指定输入/输出/决策文件
  python -m knowledge_distiller_kd.core.kd_tool_CLI -i ./input -o ./output -d ./dec/my_decisions.json

  # 运行分析并跳过语义部分
  python -m knowledge_distiller_kd.core.kd_tool_CLI -i ./input --skip-semantic

  # 设置相似度阈值并运行
  python -m knowledge_distiller_kd.core.kd_tool_CLI -i ./input --threshold 0.75
"""
        )

        parser.add_argument(
            "-i", "--input-dir", type=str, default=None, # 默认不指定，由交互模式处理或要求命令行提供
            help="输入文件夹路径 (包含 Markdown 文件)。如果未提供，将进入完整交互模式。"
        )
        parser.add_argument(
            "-o", "--output-dir", type=str, default=constants.DEFAULT_OUTPUT_DIR,
            help=f"保存去重后文件的输出文件夹路径 (默认: '{constants.DEFAULT_OUTPUT_DIR}')"
        )
        parser.add_argument(
            "-d", "--decision-file", type=str, default=constants.DEFAULT_DECISION_FILE,
            help=f"加载和保存决策的 JSON 文件路径 (默认: '{constants.DEFAULT_DECISION_FILE}')"
        )
        parser.add_argument(
            "-t", "--threshold", type=float, default=constants.DEFAULT_SIMILARITY_THRESHOLD,
            help=f"语义相似度阈值 (0.0-1.0, 默认: {constants.DEFAULT_SIMILARITY_THRESHOLD})"
        )
        parser.add_argument(
            "--skip-semantic", action="store_true",
            help="跳过语义相似度分析 (仅执行 MD5 分析)"
        )
        parser.add_argument(
            "--log-level", type=str, default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="设置日志记录级别 (默认: INFO)"
        )
        # 添加一个非交互模式标志？或者根据 input-dir 是否提供来判断
        parser.add_argument(
             "--non-interactive", action="store_true",
             help="运行分析后直接退出，不进入菜单 (需要提供 -i)。" # 可能需要更多选项来控制决策应用等
        )


        args = parser.parse_args()

        # 验证阈值范围
        if not (0.0 <= args.threshold <= 1.0):
            parser.error("相似度阈值必须在 0.0 到 1.0 之间。")

        return args

# --- 主程序入口 ---
def main() -> None:
    """主函数，程序入口点。"""
    args = KDToolCLI.parse_args()

    # 设置日志级别
    log_level_map = {
        "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
        "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL
    }
    setup_logger(log_level_map.get(args.log_level.upper(), logging.INFO))
    logger.info("KD Tool started.")
    logger.debug(f"Parsed arguments: {args}")

    try:
        # 初始化工具，使用命令行参数（如果提供）覆盖默认值
        tool = KDToolCLI(
            input_dir=args.input_dir, # 可能为 None
            decision_file=args.decision_file,
            output_dir=args.output_dir,
            skip_semantic=args.skip_semantic,
            similarity_threshold=args.threshold
        )

        # 判断运行模式
        if args.input_dir:
             # 如果提供了输入目录，直接运行分析
             logger.info(f"Input directory provided ('{args.input_dir}'), running analysis directly.")
             print(f"[*] 使用输入目录 '{args.input_dir}' 直接运行分析...")
             analysis_success = tool.run_analysis()

             if analysis_success and not args.non_interactive:
                 # 分析成功且不是非交互模式，进入分析后菜单 (需要实现)
                 # tool.handle_post_analysis_menu_or_similar() # 需要添加这个方法或调整
                 print("[*] 分析完成。请在交互模式下查看结果或应用决策。") # 简化处理
                 print("[*] （未来可以添加分析后直接保存/应用决策的命令行选项）")
                 # 暂时直接退出，或者可以调用交互查看
                 tool.handle_duplicates_view() # 让用户至少能查看一下

             elif not analysis_success:
                  logger.error("Analysis failed when run from command line.")
                  print("[错误] 从命令行运行时分析失败。")
                  sys.exit(1)
             else:
                  # 分析成功且是 --non-interactive 模式
                  logger.info("Non-interactive mode: Analysis complete, exiting.")
                  print("[*] 非交互模式：分析完成，程序退出。")
                  # 这里可以添加自动保存/应用决策的逻辑（如果需要）
                  # tool.save_decisions()
                  # tool.apply_decisions()
                  sys.exit(0)

        else:
             # 没有提供输入目录，进入完全交互模式
             logger.info("No input directory provided, entering full interactive mode.")
             print("欢迎使用知识蒸馏工具交互模式！")
             tool.run_interactive()

    except ConfigurationError as e:
         logger.critical(f"Tool initialization failed: {e}", exc_info=False)
         print(f"\n[致命错误] 工具初始化失败: {e}")
         sys.exit(1)
    except KeyboardInterrupt:
         print("\n\n[*] 用户中断操作，程序退出。")
         logger.warning("Operation interrupted by user.")
         sys.exit(0)
    except Exception as e:
         # 捕获其他未预料的全局错误
         logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
         # 使用 traceback 打印详细信息到控制台
         print("\n[致命错误] 程序运行过程中发生未预期错误:")
         traceback.print_exc()
         sys.exit(1)

    logger.info("KD Tool finished.")
    sys.exit(0)


if __name__ == "__main__":
    main()
    