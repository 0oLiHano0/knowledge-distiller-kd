"""
知识蒸馏工具CLI模块。

此模块提供命令行界面，用于执行知识蒸馏任务。
"""

import os
import sys
import json
import logging
import collections
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any, DefaultDict
import argparse

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
    extract_text_from_children,
    display_block_preview,
    get_markdown_parser,
    sort_blocks_key,
    logger
)
from knowledge_distiller_kd.core.md5_analyzer import MD5Analyzer
from knowledge_distiller_kd.core.semantic_analyzer import SemanticAnalyzer
from knowledge_distiller_kd.core import constants

# [DEPENDENCIES]
# 1. Python Standard Library: sys, os, json, logging, collections, pathlib, argparse
# 2. 需要安装：mistune, sentence-transformers (可选, 用于语义分析)
# 3. 同项目模块: constants, utils, md5_analyzer, semantic_analyzer (使用绝对导入)

# 使用 utils 中配置好的 logger
logger = logger

# 开发环境配置
DEV_DEFAULT_INPUT_DIR = "input"
DEV_DEFAULT_OUTPUT_DIR = "output"
DEV_DEFAULT_DECISION_DIR = "decisions"
DEV_DEFAULT_DECISION_FILE = os.path.join(DEV_DEFAULT_DECISION_DIR, "decisions.json")

# --- 核心逻辑类 ---
class KDToolCLI:
    """
    知识蒸馏工具的主控制类，协调文件读取、解析、分析和决策应用。

    该类负责：
    1. 管理输入/输出目录和决策文件
    2. 读取和解析 Markdown 文件
    3. 提取内容块并初始化决策
    4. 协调 MD5 和语义分析
    5. 保存和应用用户决策

    Attributes:
        input_dir: 输入目录路径 (Path object)
        output_dir: 输出目录路径 (Path object)
        decision_file: 决策文件路径 (Path object)
        skip_semantic: 是否跳过语义分析
        similarity_threshold: 语义相似度阈值
        markdown_files_content: 存储文件内容的字典 {Path: str}
        blocks_data: 存储内容块信息的列表 [(Path, index, type, text), ...]
        block_decisions: 存储块决策的字典 {decision_key: 'keep'/'delete'/'undecided'}
        md5_analyzer: MD5Analyzer 实例
        semantic_analyzer: SemanticAnalyzer 实例
    """

    def __init__(
        self,
        input_dir: Optional[Union[str, Path]] = None,
        decision_file: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        skip_semantic: bool = False,
        similarity_threshold: float = 0.8
    ) -> None:
        """
        初始化知识蒸馏工具。

        Args:
            input_dir: 输入目录路径，包含要处理的 Markdown 文件
            decision_file: 决策文件路径，用于保存和加载决策
            output_dir: 输出目录路径，用于保存去重后的文件
            skip_semantic: 是否跳过语义分析
            similarity_threshold: 语义相似度阈值（0.0 到 1.0 之间）

        Process:
            1. 初始化路径属性
            2. 设置配置参数
            3. 初始化数据存储
            4. 创建分析器实例

        Note:
            - 所有路径都会被转换为 Path 对象
            - 相似度阈值会被限制在有效范围内
            - 分析器实例会在需要时延迟创建
        """
        try:
            # 初始化路径属性
            self.input_dir = Path(input_dir) if input_dir else None
            self.decision_file = Path(decision_file) if decision_file else Path(constants.DEFAULT_DECISION_FILE)
            self.output_dir = Path(output_dir) if output_dir else Path(constants.DEFAULT_OUTPUT_DIR)

            # 设置配置参数
            self.skip_semantic = skip_semantic
            self.similarity_threshold = max(0.0, min(1.0, similarity_threshold))

            # 初始化数据存储
            self.markdown_files_content = {}
            self.blocks_data = []
            self.block_decisions = {}
            self._decisions_loaded = False

            # 延迟创建分析器实例
            self.md5_analyzer = None
            self.semantic_analyzer = None

            # 确保必要的目录存在
            if self.input_dir:
                self.input_dir = self.input_dir.resolve()
                validate_file_path(self.input_dir, must_exist=True)
            self.decision_file = self.decision_file.resolve()
            validate_file_path(self.decision_file.parent, must_exist=False)
            self.output_dir = self.output_dir.resolve()
            validate_file_path(self.output_dir, must_exist=False)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # 记录初始化信息
            logger.info("知识蒸馏工具初始化完成")
            logger.debug(f"输入目录: {self.input_dir}")
            logger.debug(f"决策文件: {self.decision_file}")
            logger.debug(f"输出目录: {self.output_dir}")
            logger.debug(f"跳过语义分析: {self.skip_semantic}")
            logger.debug(f"相似度阈值: {self.similarity_threshold}")

        except Exception as e:
            handle_error(e, "初始化知识蒸馏工具")
            raise ConfigurationError(
                "初始化知识蒸馏工具失败",
                error_code="INIT_FAILED",
                details={"error": str(e)}
            )

    def _ensure_dirs_exist(self) -> None:
        """
        确保必要的目录存在。

        Process:
            1. 检查输入目录（如果已设置）
            2. 确保决策文件目录存在
            3. 确保输出目录存在

        Note:
            - 如果目录不存在，会自动创建
            - 如果创建失败，会抛出异常
            - 记录目录创建和验证过程
        """
        try:
            # 检查输入目录
            if self.input_dir:
                if not self.input_dir.exists():
                    logger.warning(f"输入目录不存在: {self.input_dir}")
                elif not self.input_dir.is_dir():
                    raise ValueError(f"输入路径不是目录: {self.input_dir}")

            # 确保决策文件目录存在
            decision_dir = self.decision_file.parent
            if not decision_dir.exists():
                logger.info(f"创建决策文件目录: {decision_dir}")
                decision_dir.mkdir(parents=True, exist_ok=True)

            # 确保输出目录存在
            if not self.output_dir.exists():
                logger.info(f"创建输出目录: {self.output_dir}")
                self.output_dir.mkdir(parents=True, exist_ok=True)
            elif not self.output_dir.is_dir():
                raise ValueError(f"输出路径不是目录: {self.output_dir}")

        except Exception as e:
            logger.error(f"确保目录存在时出错: {e}", exc_info=True)
            raise

    def set_input_dir(self, input_dir: Union[str, Path]) -> bool:
        """
        设置输入目录。

        Args:
            input_dir: 输入目录路径

        Returns:
            bool: 是否成功设置输入目录
        """
        try:
            logger.info(f"设置输入目录: {input_dir}")
            input_dir = Path(input_dir).resolve()
            validate_file_path(input_dir, must_exist=True)
            self.input_dir = input_dir
            self._reset_state()
            return True
        except Exception as e:
            handle_error(e, "设置输入目录")
            print(f"[错误] 设置输入目录时出错: {e}")
            return False

    def _reset_state(self) -> None:
        """
        重置工具状态。
        """
        self.markdown_files_content = {}
        self.blocks_data = []
        self.block_decisions = {}
        self._decisions_loaded = False
        self.md5_analyzer = MD5Analyzer(self)
        self.semantic_analyzer = SemanticAnalyzer(self, self.similarity_threshold)

    def run_analysis(self) -> bool:
        """
        执行完整的分析流程：读取 -> 解析 -> 初始化决策 -> MD5 -> 语义。

        Returns:
            bool: 如果分析成功完成返回 True，否则返回 False

        Process:
            1. 检查输入目录是否设置
            2. 重置内部状态
            3. 读取 Markdown 文件
            4. 解析文件内容
            5. 初始化块决策
            6. 执行 MD5 去重
            7. 执行语义去重（如果未跳过）

        Note:
            - 如果任何步骤失败，分析会立即停止
            - 即使没有找到内容块，也认为分析成功完成
            - 语义分析步骤可能会被跳过（取决于 skip_semantic 设置）
            - 每个步骤都有进度显示和错误处理
        """
        # 检查输入目录
        if not self.input_dir:
            logger.error("Analysis aborted: Input directory not set.")
            print("[错误] 输入文件夹未设置。")
            return False

        logger.info(f"--- Starting analysis for folder: {self.input_dir} ---")
        print(f"\n[*] 开始分析文件夹: {self.input_dir}")

        # 重置状态
        logger.info("Resetting internal state for new analysis...")
        self._reset_state()
        logger.info("Internal state reset.")

        # 执行分析步骤
        steps = [
            ("读取文件", self._read_files),
            ("解析 Markdown", self._parse_markdown),
            ("初始化决策", self._initialize_decisions),
            ("MD5 去重", self.md5_analyzer.find_md5_duplicates),
            ("语义去重", self.semantic_analyzer.find_semantic_duplicates)
        ]

        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}...")
            print(f"\n[*] 正在执行: {step_name}...")

            # 跳过语义分析（如果配置）
            if step_name == "语义去重" and self.skip_semantic:
                logger.info("Skipping semantic analysis as requested.")
                print("[*] 已跳过语义分析。")
                continue

            try:
                # 执行步骤
                if not step_func():
                    logger.error(f"Analysis stopped: {step_name} failed.")
                    print(f"[错误] {step_name}失败，分析中止。")
                    return False

                logger.info(f"Step completed: {step_name}")
                print(f"[*] {step_name}完成。")

            except Exception as e:
                logger.error(f"Error during {step_name}: {e}", exc_info=True)
                print(f"[错误] {step_name}时出错: {e}")
                return False

        # 分析完成
        self._analysis_completed = True
        logger.info("Analysis completed successfully.")
        print("\n[*] 分析完成。")
        return True

    @safe_file_operation
    def _read_files(self) -> bool:
        """
        读取输入目录中的所有 Markdown 文件。

        Returns:
            bool: 如果成功读取了至少一个文件返回 True，否则返回 False

        Raises:
            FileOperationError: 当文件操作失败时
        """
        try:
            if not self.input_dir:
                raise ConfigurationError("未设置输入目录")

            # 获取所有 Markdown 文件
            md_files = list(self.input_dir.glob("**/*.md"))
            total_files = len(md_files)
            
            if total_files == 0:
                logger.warning(f"在目录 {self.input_dir} 中未找到 Markdown 文件")
                return False

            # 读取文件内容
            success_count = 0
            error_files = []

            for i, file_path in enumerate(md_files, 1):
                print(f"[*] 正在读取文件 ({i}/{total_files}): {file_path.name}")

                try:
                    # 忽略隐藏文件
                    if file_path.name.startswith('.'):
                        logger.debug(f"忽略隐藏文件: {file_path}")
                        continue

                    # 读取文件内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 保存文件内容
                    self.markdown_files_content[file_path] = content
                    success_count += 1

                except Exception as e:
                    error_msg = f"读取文件 {file_path.name} 时出错: {e}"
                    logger.error(error_msg, exc_info=False)
                    error_files.append((file_path.name, str(e)))

            # 显示读取结果
            logger.info(f"成功读取 {success_count} 个文件")
            print(f"\n[*] 文件读取完成: 成功读取 {success_count}/{total_files} 个文件")

            if error_files:
                print("\n[警告] 以下文件读取失败:")
                for file_name, error in error_files:
                    print(f"  - {file_name}: {error}")

            return success_count > 0

        except Exception as e:
            handle_error(e, "读取文件")
            raise FileOperationError(
                "读取文件失败",
                error_code="READ_FILES_FAILED",
                details={"error": str(e)}
            )

    @safe_file_operation
    def _parse_markdown(self) -> bool:
        """
        解析所有 Markdown 文件的内容。

        Returns:
            bool: 如果成功解析了至少一个文件返回 True，否则返回 False

        Raises:
            AnalysisError: 当解析失败时
        """
        try:
            if not self.markdown_files_content:
                raise ConfigurationError("没有可解析的文件内容")

            # 清空旧的块数据
            self.blocks_data.clear()

            # 解析每个文件的内容
            total_files = len(self.markdown_files_content)
            success_count = 0
            error_files = []

            for i, (file_path, content) in enumerate(self.markdown_files_content.items(), 1):
                print(f"[*] 正在解析文件 ({i}/{total_files}): {file_path.name}")

                try:
                    # 解析 Markdown 内容
                    tokens = get_markdown_parser().parse(content)

                    # 提取内容块
                    block_index = 0
                    for token in tokens:
                        if isinstance(token, list):
                            continue  # 跳过列表类型的 token
                        
                        token_type = token['type']
                        token_text = token['text'].strip()
                        
                        if token_text:  # 只处理非空内容
                            self.blocks_data.append((
                                file_path,
                                block_index,
                                token_type,
                                token_text
                            ))
                            block_index += 1

                    success_count += 1

                except Exception as e:
                    error_msg = f"解析文件 {file_path.name} 时出错: {e}"
                    logger.error(error_msg, exc_info=True)
                    error_files.append((file_path.name, str(e)))

            # 显示解析结果
            total_blocks = len(self.blocks_data)
            logger.info(f"成功解析 {success_count} 个文件，共提取 {total_blocks} 个内容块")
            print(f"\n[*] 文件解析完成: 成功解析 {success_count}/{total_files} 个文件")
            print(f"[*] 共提取 {total_blocks} 个内容块")

            if error_files:
                print("\n[警告] 以下文件解析失败:")
                for file_name, error in error_files:
                    print(f"  - {file_name}: {error}")

            return success_count > 0

        except Exception as e:
            handle_error(e, "解析 Markdown 内容")
            raise AnalysisError(
                "解析 Markdown 内容失败",
                error_code="PARSE_MARKDOWN_FAILED",
                details={"error": str(e)}
            )

    def _initialize_decisions(self) -> None:
        """
        初始化所有内容块的决策状态。

        Process:
            1. 清空旧的决策数据
            2. 为每个内容块创建决策键
            3. 设置初始决策状态为 'undecided'
            4. 统计初始化结果

        Note:
            - 决策键格式为 "file_path::block_index::block_type"
            - 所有块初始状态都设置为 'undecided'
            - 如果创建决策键失败，会记录错误但继续处理其他块
            - 显示初始化进度和结果统计
        """
        logger.info("Initializing block decisions...")
        print("\n[*] 正在初始化块决策...")

        # 清空旧数据
        self.block_decisions.clear()
        total_blocks = len(self.blocks_data)
        error_blocks = []

        try:
            for i, block_info in enumerate(self.blocks_data, 1):
                file_path, b_index, b_type, _ = block_info
                print(f"[*] 正在初始化决策 ({i}/{total_blocks}): {file_path.name}#{b_index}")

                try:
                    # 创建决策键
                    abs_path_str = str(Path(file_path).resolve())
                    key = create_decision_key(abs_path_str, b_index, b_type)
                    # 设置初始决策状态
                    self.block_decisions[key] = 'undecided'
                except Exception as e:
                    # 记录创建决策键时的错误
                    error_msg = f"Error creating decision key for block: {file_path}#{b_index} - {e}"
                    logger.error(error_msg, exc_info=False)
                    print(f"[警告] 无法为块创建决策键 {file_path}#{b_index}: {e}")
                    error_blocks.append((file_path.name, b_index, str(e)))

            # 显示初始化结果
            logger.info(f"Initialized decisions for {len(self.block_decisions)} blocks.")
            print(f"\n[*] 决策初始化完成: 成功初始化 {len(self.block_decisions)}/{total_blocks} 个块的决策。")

            if error_blocks:
                print("\n[警告] 以下块的决策初始化失败:")
                for file_name, block_index, error in error_blocks:
                    print(f"  - {file_name}#{block_index}: {error}")

        except Exception as e:
            # 记录初始化过程中的错误
            error_msg = "Error during decision initialization."
            logger.error(error_msg, exc_info=True)
            print(f"[错误] 初始化块决策时出错: {e}")

    def load_decisions(self) -> bool:
        """
        从决策文件加载决策。

        Returns:
            bool: 是否成功加载决策
        """
        try:
            # 检查决策文件是否存在
            if not self.decision_file.exists():
                logger.warning(f"决策文件不存在: {self.decision_file}")
                return False

            # 读取决策文件
            logger.info(f"正在从文件加载决策: {self.decision_file}")
            with open(self.decision_file, 'r', encoding='utf-8') as f:
                decisions = json.load(f)

            logger.info(f"成功读取 {len(decisions)} 条决策记录。")

            # 处理每条决策记录
            success_count = 0
            failed_records = []
            for i, record in enumerate(decisions, 1):
                try:
                    logger.info(f"正在处理决策记录 ({i}/{len(decisions)})")
                    file_path = Path(record['file'])
                    block_index = record['index']
                    block_type = record['type']
                    decision = record['decision']

                    # 创建决策键
                    key = create_decision_key(str(file_path), block_index, block_type)
                    self.block_decisions[key] = decision
                    success_count += 1

                except Exception as e:
                    logger.error(f"处理记录 {i} 时出错: {e}")
                    failed_records.append((i, str(e)))
                    continue

            # 记录处理结果
            logger.info(f"决策加载完成: 成功应用 {success_count}/{len(decisions)} 条决策。")
            if failed_records:
                logger.warning("以下记录处理失败:")
                for record_num, error in failed_records:
                    logger.warning(f"  - 记录 #{record_num}: {error}")

            return success_count > 0

        except Exception as e:
            handle_error(e, "加载决策")
            return False

    def save_decisions(self) -> bool:
        """
        将当前决策保存到文件。

        Returns:
            bool: 是否成功保存决策
        """
        try:
            # 准备决策数据
            decisions = []
            failed_decisions = []
            for i, (key, decision) in enumerate(self.block_decisions.items(), 1):
                try:
                    logger.info(f"正在处理决策 ({i}/{len(self.block_decisions)})")
                    file_path, block_index, block_type = parse_decision_key(key)
                    decisions.append({
                        'file': str(file_path),
                        'index': block_index,
                        'type': block_type,
                        'decision': decision
                    })
                except Exception as e:
                    logger.error(f"处理决策时出错: {e}")
                    failed_decisions.append((key, str(e)))
                    continue

            # 检查是否有有效决策需要保存
            if not decisions:
                logger.warning("没有有效的决策需要保存。")
                return False

            # 确保决策文件目录存在
            self.decision_file.parent.mkdir(parents=True, exist_ok=True)

            # 保存决策
            with open(self.decision_file, 'w', encoding='utf-8') as f:
                json.dump(decisions, f, indent=2, ensure_ascii=False)

            logger.info(f"成功保存 {len(decisions)} 条决策到文件: {self.decision_file}")
            if failed_decisions:
                logger.warning("以下决策保存失败:")
                for key, error in failed_decisions:
                    logger.warning(f"  - {key}: {error}")

            return True

        except Exception as e:
            handle_error(e, "保存决策")
            return False

    @safe_file_operation
    def apply_decisions(self) -> bool:
        """
        应用决策，生成去重后的文件。

        Returns:
            bool: 如果成功应用决策并生成文件返回 True，否则返回 False

        Raises:
            FileOperationError: 当应用决策失败时
        """
        try:
            if not self.blocks_data:
                logger.warning("No blocks to process.")
                print("[警告] 没有可处理的内容块。")
                return True  # 没有块也算成功

            # 按文件分组块数据
            files_blocks = {}
            for block in self.blocks_data:
                file_path = block[0]
                if file_path not in files_blocks:
                    files_blocks[file_path] = []
                files_blocks[file_path].append(block)

            # 处理每个文件
            total_files = len(files_blocks)
            processed_files = 0
            error_files = []

            for i, (file_path, blocks) in enumerate(files_blocks.items(), 1):
                print(f"[*] 正在处理文件 ({i}/{total_files}): {file_path.name}")

                try:
                    # 创建输出文件
                    output_name = file_path.stem + constants.DEFAULT_OUTPUT_SUFFIX + file_path.suffix
                    output_path = self.output_dir / output_name

                    # 确保输出目录存在
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # 按顺序写入保留的块
                    with open(output_path, 'w', encoding='utf-8') as f:
                        for block in blocks:
                            file_path, block_index, block_type, block_text = block
                            key = create_decision_key(str(file_path), block_index, block_type)
                            decision = self.block_decisions.get(key, 'undecided')

                            if decision != 'delete':  # 保留 'keep' 和 'undecided'
                                f.write(block_text + '\n\n')

                    processed_files += 1

                except Exception as e:
                    error_msg = f"处理文件 {file_path} 时出错: {e}"
                    logger.error(error_msg, exc_info=True)
                    error_files.append((file_path.name, str(e)))

            # 显示处理结果
            logger.info(f"成功处理 {processed_files} 个文件")
            print(f"\n[*] 决策应用完成: 成功处理 {processed_files}/{total_files} 个文件")

            if error_files:
                print("\n[警告] 以下文件处理失败:")
                for file_name, error in error_files:
                    print(f"  - {file_name}: {error}")

            return processed_files > 0

        except Exception as e:
            handle_error(e, "应用决策")
            raise FileOperationError(
                "应用决策失败",
                error_code="APPLY_DECISIONS_FAILED",
                details={"error": str(e)}
            )

    def display_main_menu(self) -> None:
        """
        显示主菜单。

        Process:
            1. 显示程序标题和版本信息
            2. 显示当前配置信息
            3. 显示可用的操作选项
            4. 等待用户输入选择

        Note:
            - 显示当前输入目录、决策文件和输出目录的路径
            - 提供运行分析和退出程序的选项
            - 使用清晰的格式和颜色显示菜单
        """
        print("\n" + "=" * 50)
        print("知识蒸馏工具 (Knowledge Distiller)".center(50))
        print(f"版本 {constants.VERSION}".center(50))
        print("=" * 50)

        # 显示当前配置
        print("\n当前配置:")
        print(f"  输入目录: {self.input_dir if self.input_dir else '未设置'}")
        print(f"  决策文件: {self.decision_file}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  跳过语义分析: {'是' if self.skip_semantic else '否'}")
        print(f"  相似度阈值: {self.similarity_threshold}")

        # 显示菜单选项
        print("\n请选择操作:")
        print("  1. 运行分析")
        print("  2. 退出程序")
        print("\n" + "-" * 50)

    def display_post_analysis_menu(self) -> None:
        """
        显示分析后的菜单。

        Process:
            1. 显示分析结果统计
            2. 显示当前配置信息
            3. 显示可用的操作选项
            4. 等待用户输入选择

        Note:
            - 显示找到的重复块数量和类型
            - 提供查看、保存和应用决策的选项
            - 使用清晰的格式和颜色显示菜单
        """
        print("\n" + "=" * 50)
        print("分析完成".center(50))
        print("=" * 50)

        # 显示分析结果
        print("\n分析结果:")
        print(f"  总文件数: {len(self.markdown_files_content)}")
        print(f"  总块数: {len(self.blocks_data)}")
        print(f"  已决策块数: {sum(1 for d in self.block_decisions.values() if d != 'undecided')}")
        print(f"  未决策块数: {sum(1 for d in self.block_decisions.values() if d == 'undecided')}")

        # 显示当前配置
        print("\n当前配置:")
        print(f"  输入目录: {self.input_dir}")
        print(f"  决策文件: {self.decision_file}")
        print(f"  输出目录: {self.output_dir}")

        # 显示菜单选项
        print("\n请选择操作:")
        print("  1. 查看重复块")
        print("  2. 保存当前决策")
        print("  3. 应用当前决策")
        print("  4. 返回主菜单")
        print("\n" + "-" * 50)

    def display_duplicates_menu(self) -> None:
        """
        显示重复块查看菜单。

        Process:
            1. 显示重复块统计信息
            2. 显示查看选项
            3. 等待用户输入选择

        Note:
            - 显示 MD5 重复和语义重复的数量
            - 提供查看不同类型重复的选项
            - 使用清晰的格式和颜色显示菜单
        """
        print("\n" + "=" * 50)
        print("重复块查看".center(50))
        print("=" * 50)

        # 获取重复块统计
        md5_duplicates = self.md5_analyzer.md5_duplicates if self.md5_analyzer else []
        semantic_duplicates = self.semantic_analyzer.semantic_duplicates if self.semantic_analyzer else []

        # 显示重复块统计
        print("\n重复块统计:")
        print(f"  MD5 重复: {len(md5_duplicates)} 组")
        print(f"  语义重复: {len(semantic_duplicates)} 组")

        # 显示菜单选项
        print("\n请选择查看:")
        print("  1. 查看 MD5 重复")
        print("  2. 查看语义重复")
        print("  3. 返回上一级")

    def handle_user_input(self) -> None:
        """
        处理用户输入并执行相应操作。

        Process:
            1. 显示主菜单
            2. 获取用户输入
            3. 根据输入执行相应操作
            4. 循环直到用户选择退出

        Note:
            - 提供清晰的错误提示
            - 支持返回上级菜单
            - 显示操作进度和结果
        """
        while True:
            self.display_main_menu()
            choice = input("\n请输入选项编号: ").strip()

            if choice == '1':
                # 运行分析
                if not self.input_dir:
                    print("\n[错误] 请先设置输入目录。")
                    continue

                print("\n[*] 开始运行分析...")
                if self.run_analysis():
                    self.handle_post_analysis()
                else:
                    print("\n[错误] 分析过程中出现错误。")

            elif choice == '2':
                # 退出程序
                print("\n[*] 正在退出程序...")
                break

            else:
                print("\n[错误] 无效的选项，请重新输入。")

    def handle_post_analysis(self) -> None:
        """
        处理分析后的用户交互。

        Process:
            1. 显示分析后菜单
            2. 获取用户输入
            3. 根据输入执行相应操作
            4. 支持返回主菜单

        Note:
            - 提供查看、保存和应用决策的选项
            - 显示操作进度和结果
            - 支持返回上级菜单
        """
        while True:
            self.display_post_analysis_menu()
            choice = input("\n请输入选项编号: ").strip()

            if choice == '1':
                # 查看重复块
                self.handle_duplicates_view()

            elif choice == '2':
                # 保存当前决策
                print("\n[*] 正在保存决策...")
                if self.save_decisions():
                    print("\n[+] 决策保存成功。")
                else:
                    print("\n[错误] 保存决策时出现错误。")

            elif choice == '3':
                # 应用当前决策
                print("\n[*] 正在应用决策...")
                if self.apply_decisions():
                    print("\n[+] 决策应用成功。")
                else:
                    print("\n[错误] 应用决策时出现错误。")

            elif choice == '4':
                # 返回主菜单
                break

            else:
                print("\n[错误] 无效的选项，请重新输入。")

    def handle_duplicates_view(self) -> None:
        """
        处理重复块查看的用户交互。

        Process:
            1. 显示重复块菜单
            2. 获取用户输入
            3. 根据输入显示相应类型的重复块
            4. 支持返回上级菜单

        Note:
            - 提供查看 MD5 重复和语义重复的选项
            - 显示重复块的详细信息
            - 支持返回上级菜单
        """
        while True:
            self.display_duplicates_menu()
            choice = input("\n请输入选项编号: ").strip()

            if choice == '1':
                # 查看 MD5 重复
                if self.md5_analyzer and self.md5_analyzer.md5_duplicates:
                    self.md5_analyzer.display_md5_duplicates_list()
                else:
                    print("\n[信息] 没有找到 MD5 重复的块。")

            elif choice == '2':
                # 查看语义重复
                if self.semantic_analyzer and self.semantic_analyzer.semantic_duplicates:
                    self.semantic_analyzer.display_semantic_duplicates_list()
                else:
                    print("\n[信息] 没有找到语义重复的块。")

            elif choice == '3':
                # 返回上一级
                break

            else:
                print("\n[错误] 无效的选项，请重新输入。")

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """
        解析命令行参数。

        Returns:
            argparse.Namespace: 包含解析后的参数的命名空间对象

        Process:
            1. 创建参数解析器
            2. 添加参数定义
            3. 解析命令行参数
            4. 验证参数值

        Note:
            - 支持设置输入目录、决策文件和输出目录
            - 支持跳过语义分析和设置相似度阈值
            - 提供详细的帮助信息
        """
        parser = argparse.ArgumentParser(
            description="知识蒸馏工具 - 用于检测和处理 Markdown 文件中的重复内容",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
  # 使用默认配置运行
  python kd_tool_CLI.py

  # 指定输入目录和输出目录
  python kd_tool_CLI.py -i ./input -o ./output

  # 跳过语义分析
  python kd_tool_CLI.py --skip-semantic

  # 设置相似度阈值
  python kd_tool_CLI.py --similarity-threshold 0.8
            """
        )

        # 添加参数
        parser.add_argument(
            '-i', '--input-dir',
            type=str,
            help='输入目录路径（包含要处理的 Markdown 文件）'
        )
        parser.add_argument(
            '-d', '--decision-file',
            type=str,
            help='决策文件路径（用于保存和加载决策）'
        )
        parser.add_argument(
            '-o', '--output-dir',
            type=str,
            help='输出目录路径（用于保存去重后的文件）'
        )
        parser.add_argument(
            '--skip-semantic',
            action='store_true',
            help='跳过语义分析（只进行 MD5 重复检测）'
        )
        parser.add_argument(
            '--similarity-threshold',
            type=float,
            default=0.8,
            help='语义相似度阈值（0.0 到 1.0 之间，默认 0.8）'
        )
        parser.add_argument(
            '-v', '--verbose',
            action='store_true',
            help='启用详细日志输出'
        )

        # 解析参数
        args = parser.parse_args()

        # 验证参数
        if args.input_dir:
            input_path = Path(args.input_dir)
            if not input_path.exists():
                parser.error(f"输入目录不存在: {input_path}")
            if not input_path.is_dir():
                parser.error(f"输入路径不是目录: {input_path}")

        if args.decision_file:
            decision_path = Path(args.decision_file)
            if decision_path.exists() and not decision_path.is_file():
                parser.error(f"决策文件路径不是文件: {decision_path}")

        if args.output_dir:
            output_path = Path(args.output_dir)
            if output_path.exists() and not output_path.is_dir():
                parser.error(f"输出路径不是目录: {output_path}")

        if args.similarity_threshold < 0 or args.similarity_threshold > 1:
            parser.error("相似度阈值必须在 0.0 到 1.0 之间")

        return args

    def run_from_args(self, args: argparse.Namespace) -> None:
        """
        根据命令行参数运行工具。

        Args:
            args: 解析后的命令行参数

        Process:
            1. 设置配置参数
            2. 运行分析
            3. 处理分析结果

        Note:
            - 支持设置日志级别
            - 处理分析过程中的错误
            - 提供清晰的进度和结果反馈
        """
        # 设置日志级别
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # 设置配置参数
        if args.input_dir:
            self.input_dir = Path(args.input_dir)
        if args.decision_file:
            self.decision_file = Path(args.decision_file)
        if args.output_dir:
            self.output_dir = Path(args.output_dir)
        if args.skip_semantic:
            self.skip_semantic = True
        if args.similarity_threshold:
            self.similarity_threshold = args.similarity_threshold

        # 显示配置信息
        print("\n[*] 当前配置:")
        print(f"  输入目录: {self.input_dir if self.input_dir else '未设置'}")
        print(f"  决策文件: {self.decision_file}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  跳过语义分析: {'是' if self.skip_semantic else '否'}")
        print(f"  相似度阈值: {self.similarity_threshold}")

        # 检查必要参数
        if not self.input_dir:
            print("\n[错误] 请指定输入目录。")
            return

        # 运行分析
        print("\n[*] 开始运行分析...")
        if self.run_analysis():
            self.handle_post_analysis()
        else:
            print("\n[错误] 分析过程中出现错误。")

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
    if kd_tool.set_input_dir(input_directory):
        analysis_run_successfully = kd_tool.run_analysis()
    else:
        logger.critical(f"Failed to set input directory: {input_directory}. Exiting.")
        print(f"[错误] 无法设置输入目录 '{input_directory}'。程序退出。")
        sys.exit(1) # 设置失败则直接退出

    # --- 如果分析成功，直接进入分析后菜单循环 ---
    if analysis_run_successfully:
        logger.info("Analysis completed. Entering post-analysis menu.")
        kd_tool.handle_post_analysis()
    else:
        # 如果分析失败
        logger.error("Analysis failed to complete successfully. Exiting.")
        print("[!] 分析未能成功完成，请检查日志信息。程序退出。")
        sys.exit(1) # 分析失败也退出

    # 如果从分析后菜单退出 (理论上只有通过选项 7 退出)
    logger.info("KD Tool finished.")
    sys.exit(0)

def main() -> None:
    """
    主函数，程序的入口点。

    Process:
        1. 解析命令行参数
        2. 创建工具实例
        3. 根据参数运行工具
        4. 处理异常情况

    Note:
        - 支持命令行参数和交互式模式
        - 提供清晰的错误处理和日志记录
        - 确保资源正确释放
    """
    try:
        # 解析命令行参数
        args = KDToolCLI.parse_args()

        # 创建工具实例
        tool = KDToolCLI()

        # 根据参数运行工具
        if any([args.input_dir, args.decision_file, args.output_dir, args.skip_semantic]):
            # 命令行模式
            tool.run_from_args(args)
        else:
            # 交互式模式
            tool.handle_user_input()

    except KeyboardInterrupt:
        print("\n\n[*] 程序被用户中断。")
        sys.exit(0)
    except Exception as e:
        logger.error("程序运行出错", exc_info=True)
        print(f"\n[错误] 程序运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

