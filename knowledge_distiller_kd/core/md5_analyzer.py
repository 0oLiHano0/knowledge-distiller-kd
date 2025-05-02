# KD_Tool_CLI/knowledge_distiller_kd/core/md5_analyzer.py
"""
MD5分析器模块，用于检测MD5重复内容。

此模块提供以下功能：
1. 计算文本的MD5哈希值
2. 查找MD5重复内容
3. 处理MD5重复内容
"""

import logging
import collections
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, DefaultDict
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import re

# 假设你有一个 ContentBlock 类定义在别处
# from .document_processor import ContentBlock # 确保导入 ContentBlock
# 同样，需要导入相关的工具函数和错误处理
from knowledge_distiller_kd.core.error_handler import (
    KDError, FileOperationError, ModelError, AnalysisError, UserInputError, # 添加 UserInputError
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
    logger # 使用 utils 中配置好的 logger
)
from knowledge_distiller_kd.core import constants
# 确保 ContentBlock 被正确导入，或者如果它在全局命名空间中，则不需要导入
from knowledge_distiller_kd.core.document_processor import ContentBlock

# 使用 utils 中配置好的 logger
logger = logger

class MD5Analyzer:
    """
    MD5分析器类，用于检测完全相同的内容块。

    该类负责：
    1. 计算内容块的MD5哈希值
    2. 检测具有相同哈希值的块
    3. 标记重复块的决策

    Attributes:
        kd_tool: KDToolCLI实例的引用
        md5_duplicates: 存储找到的MD5重复组 [(block1, block2, ...), ...]
    """

    def __init__(self, kd_tool):
        """
        初始化MD5分析器。

        Args:
            kd_tool: KDToolCLI实例的引用
        """
        self.kd_tool = kd_tool
        self.md5_duplicates: List[List[ContentBlock]] = [] # 明确类型

    def find_md5_duplicates(self) -> bool:
        """
        查找具有相同MD5哈希值的内容块。

        Returns:
            bool: 如果成功完成分析返回 True，否则返回 False
        """
        try:
            # 按MD5哈希值分组
            hash_groups: DefaultDict[str, List[ContentBlock]] = defaultdict(list)
            # 确保 blocks_data 存在且是 ContentBlock 列表
            if not hasattr(self.kd_tool, 'blocks_data') or not self.kd_tool.blocks_data:
                 logger.warning("MD5 analysis skipped: No blocks_data found in KDTool instance.")
                 return True # 没有块也算分析成功

            total_blocks = len(self.kd_tool.blocks_data)
            processed_count = 0

            logger.info(f"开始MD5重复检测，共有{total_blocks}个内容块")
            logger.debug(f"当前决策数量: {len(self.kd_tool.block_decisions)}")

            # 计算每个块的哈希值
            for block in self.kd_tool.blocks_data:
                # 确保 block 是 ContentBlock 类型
                if not isinstance(block, ContentBlock):
                    logger.warning(f"Skipping item in blocks_data as it's not a ContentBlock: {type(block)}")
                    processed_count += 1
                    continue

                # 跳过已经有明确决策的块
                try:
                    key = create_decision_key(
                        block.file_path,
                        block.block_id,
                        block.block_type # 使用 ContentBlock 的 block_type 属性
                    )
                except Exception as e:
                    logger.error(f"Error creating decision key for block {block.block_id} in {block.file_path}: {e}")
                    processed_count += 1
                    continue # 跳过无法创建键的块

                decision = self.kd_tool.block_decisions.get(key)
                if decision in ['keep', 'delete']:
                    logger.debug(f"跳过已有决策的块: {key} (决策: {decision})")
                    processed_count += 1
                    continue

                # logger.debug(f"正在计算哈希值 ({processed_count + 1}/{total_blocks}) for block {key}")
                # logger.debug(f"块类型: {block.block_type}")
                # logger.debug(f"原始文本: {block.original_text[:50]}...")
                # logger.debug(f"分析文本: {block.analysis_text[:50]}...")

                # 获取用于计算哈希的文本 (来自 ContentBlock 的 analysis_text)
                text_to_hash = block.analysis_text # 已经过标准化处理

                # ==================== 修改开始 ====================
                # 跳过仅包含代码块结束符的块
                if block.block_type == "CodeSnippet" and text_to_hash.strip() == '```':
                    logger.debug(f"跳过仅包含代码结束符的块: {key}")
                    processed_count += 1
                    continue
                # ==================== 修改结束 ====================

                # 标准化标题块的内容 (这部分逻辑现在可能可以简化或移除，
                # 因为ContentBlock的_normalize_text应该已经处理了标题)
                # 我们暂时保留，以防万一
                if block.block_type == "Title":
                    # 移除标题符号和空白字符 (再次处理以确保)
                    normalized_title_text = re.sub(r'^#+\s*', '', text_to_hash).strip()
                    text_to_hash = normalized_title_text
                    # logger.debug(f"Title block normalized for hashing: '{text_to_hash[:50]}...'")


                # 计算MD5哈希值，包含块类型
                # 注意：确保 text_to_hash 是字符串
                if not isinstance(text_to_hash, str):
                    logger.warning(f"Analysis text for block {key} is not a string: {type(text_to_hash)}. Skipping.")
                    processed_count += 1
                    continue

                # 确保 block_type 是字符串
                block_type_str = str(block.block_type)

                # 使用strip()确保比较的一致性，去除首尾空白
                hash_input = f"{block_type_str}:{text_to_hash.strip()}"
                try:
                    md5_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
                    # logger.debug(f"哈希输入: {hash_input[:100]}...") # 限制长度
                    # logger.debug(f"MD5哈希: {md5_hash}")
                    hash_groups[md5_hash].append(block)
                except Exception as e:
                     logger.error(f"Error calculating hash for block {key}: {e}")
                     # 可以选择跳过这个块或采取其他错误处理

                processed_count += 1
                # if processed_count % 100 == 0: # 每处理100个块打印一次进度
                #     logger.info(f"MD5 hashing progress: {processed_count}/{total_blocks}")


            # 找出重复的组
            duplicates_found = False
            self.md5_duplicates.clear() # 清空之前的结果

            logger.info(f"哈希计算完成，共有 {len(hash_groups)} 个唯一的哈希组。")
            duplicate_group_count = 0
            for md5_hash, blocks in hash_groups.items():
                if len(blocks) > 1:
                    duplicate_group_count += 1
                    logger.info(f"找到第 {duplicate_group_count} 组重复内容，哈希值: {md5_hash}, 块数: {len(blocks)}")
                    # for block in blocks:
                    #     logger.debug(f"  - 重复块: {block.file_path}#{block.block_id} ({block.block_type})")

                    # 按文件路径和块ID排序，确保结果的一致性
                    try:
                        blocks.sort(key=lambda b: (str(b.file_path), str(b.block_id)))
                    except Exception as e:
                         logger.error(f"Error sorting duplicate blocks for hash {md5_hash}: {e}. Skipping this group.")
                         continue

                    duplicates_found = True
                    self.md5_duplicates.append(blocks)

                    # 自动标记重复块的决策（保留第一个，删除其他）
                    # (确保 blocks 列表不为空)
                    if not blocks:
                         logger.warning(f"Empty block list found for hash {md5_hash}. Skipping decision making.")
                         continue

                    first_block = blocks[0]
                    try:
                        first_key = create_decision_key(
                            first_block.file_path,
                            first_block.block_id,
                            first_block.block_type
                        )
                        # 仅当块未被决策时才设置
                        if self.kd_tool.block_decisions.get(first_key) == 'undecided':
                             self.kd_tool.block_decisions[first_key] = constants.DECISION_KEEP
                             logger.debug(f"自动标记保留块: {first_key}")
                        # else:
                        #      logger.debug(f"块 {first_key} 已有决策 {self.kd_tool.block_decisions.get(first_key)}，不再自动标记为 'keep'。")
                    except Exception as e:
                         logger.error(f"Error creating/setting decision key for first block {first_block.block_id} in hash group {md5_hash}: {e}")


                    for block in blocks[1:]:
                        try:
                            key = create_decision_key(
                                block.file_path,
                                block.block_id,
                                block.block_type
                            )
                            # 仅当块未被决策时才设置
                            if self.kd_tool.block_decisions.get(key) == 'undecided':
                                 self.kd_tool.block_decisions[key] = constants.DECISION_DELETE
                                 logger.debug(f"自动标记删除块: {key}")
                            # else:
                            #      logger.debug(f"块 {key} 已有决策 {self.kd_tool.block_decisions.get(key)}，不再自动标记为 'delete'。")
                        except Exception as e:
                             logger.error(f"Error creating/setting decision key for block {block.block_id} in hash group {md5_hash}: {e}")


            if duplicates_found:
                logger.info(f"MD5分析完成，共找到 {len(self.md5_duplicates)} 组精确重复内容。")
            else:
                logger.info("MD5分析完成，未找到精确重复内容。")

            return True

        except Exception as e:
            # 使用更具体的错误类型或添加上下文
            logger.error(f"在查找MD5重复内容时发生未预期错误: {e}", exc_info=True)
            handle_error(e, "查找MD5重复内容")
            # 不直接抛出 AnalysisError，让调用者决定如何处理
            # raise AnalysisError("查找MD5重复内容失败", details={"error": str(e)})
            return False # 表示分析未成功

    def _display_md5_duplicates_list(self) -> None:
        """
        显示找到的MD5重复内容列表。
        """
        if not self.md5_duplicates:
            print("\n[*] 未找到 MD5 精确重复的内容块。")
            return

        print("\n--- MD5 精确重复内容块列表 ---")
        for i, group in enumerate(self.md5_duplicates, 1):
            print(f"\n{'='*50}")
            print(f"重复组 {i} (共 {len(group)} 个块)")
            print(f"{'='*50}")

            # 打印组内第一个块的内容作为参考
            if group:
                print(f"内容示例 (来自块 1 的 analysis_text):")
                print(display_block_preview(group[0].analysis_text, max_len=100)) # 使用 analysis_text
                print("-" * 30)

            for j, block in enumerate(group, 1):
                try:
                    key = create_decision_key(
                        block.file_path,
                        block.block_id,
                        block.block_type
                    )
                    decision = self.kd_tool.block_decisions.get(key, constants.DECISION_UNDECIDED)

                    print(f"\n[块 {j}]")
                    print(f"  文件: {Path(block.file_path).name}")
                    print(f"  块ID: {block.block_id}")
                    print(f"  类型: {block.block_type}")
                    print(f"  决策: {decision}")
                    print(f"  原始文本预览:")
                    print(f"    {display_block_preview(block.original_text)}") # 打印预览
                    # print("-" * 30) # 减少分隔线使输出更紧凑
                except Exception as e:
                    logger.error(f"Error displaying block {getattr(block, 'block_id', 'N/A')} in group {i}: {e}")
                    print(f"\n[块 {j}] - 显示时出错: {e}")

    def review_md5_duplicates_interactive(self) -> None:
        """
        交互式处理MD5重复内容。
        注意：MD5 重复通常是自动处理的 (保留第一个，删除其他)。
              这个交互式审查更多是用于确认或覆盖自动决策。
        """
        try:
            if not self.md5_duplicates:
                logger.info("没有找到 MD5 重复内容可供审查。")
                print("\n[*] 没有找到 MD5 重复内容可供审查。")
                return

            while True:
                self._display_md5_duplicates_list()
                print("\nMD5 重复项审查选项 (通常已自动处理，这里可覆盖):")
                print("  k <组号> <块号> [...] - 保留指定块 (例如: k 1 1 保留第1组块1)")
                print("  d <组号> <块号> [...] - 删除指定块 (例如: d 1 2 删除第1组块2)")
                # print("  r <组号> <块号> [...] - 重置指定块决策为 undecided") # 可能需要这个？
                print("  a <组号> <块号>       - 应用：保留指定块，删除同组其他块 (例如: a 1 2)")
                print("  save                - 保存当前所有决策到文件")
                print("  q                   - 完成 MD5 重复项审查并退出")

                action = input("请输入操作: ").lower().strip()
                if action == 'q':
                    break
                elif action == 'save':
                    if self.kd_tool.save_decisions():
                        print(" [*] 决策已保存。")
                    else:
                        print(" [!] 保存决策失败。")
                    continue # 保存后继续审查

                if not action:
                    continue

                try:
                    if not self._process_user_action_md5(action):
                         print("[错误] 无效的操作或参数，请重试。")
                except UserInputError as e:
                     print(f"[错误] 输入错误: {e}")
                except Exception as e:
                    logger.error(f"处理 MD5 用户操作时出错: {e}", exc_info=True)
                    print(f"[错误] 处理操作时发生意外错误: {e}")

        except Exception as e:
            handle_error(e, "审查MD5重复内容")
            # 不抛出，允许程序继续
            # raise AnalysisError("审查MD5重复内容失败", details={"error": str(e)})

    def _process_user_action_md5(self, action: str) -> bool:
        """
        处理针对 MD5 重复项审查的用户操作。

        Args:
            action: 用户输入的操作字符串 (例如: "k 1 1", "d 1 2", "a 1 2")

        Returns:
            bool: 如果操作有效且被处理则返回 True，否则返回 False

        Raises:
            UserInputError: 当输入格式或参数无效时
        """
        parts = action.split()
        if not parts:
            return False

        command = parts[0]
        args = parts[1:]

        if command not in ['k', 'd', 'a']:
            raise UserInputError(f"无效的命令: '{command}'。请使用 'k', 'd', 'a'。")

        if command in ['k', 'd']:
            # 需要偶数个参数 (组号, 块号, 组号, 块号...)
            if not args or len(args) % 2 != 0:
                raise UserInputError(f"命令 '{command}' 需要成对的 <组号> <块号> 参数。")

            processed_count = 0
            for i in range(0, len(args), 2):
                try:
                    group_idx = int(args[i]) - 1 # 用户输入从1开始
                    block_idx_in_group = int(args[i+1]) - 1 # 用户输入从1开始

                    if not (0 <= group_idx < len(self.md5_duplicates)):
                        raise UserInputError(f"无效的组号: {args[i]}。范围是 1 到 {len(self.md5_duplicates)}。")

                    group = self.md5_duplicates[group_idx]
                    if not (0 <= block_idx_in_group < len(group)):
                         raise UserInputError(f"无效的块号: {args[i+1]}。第 {args[i]} 组只有 {len(group)} 个块。")

                    target_block = group[block_idx_in_group]
                    key = create_decision_key(
                        target_block.file_path,
                        target_block.block_id,
                        target_block.block_type
                    )

                    new_decision = constants.DECISION_KEEP if command == 'k' else constants.DECISION_DELETE
                    self.kd_tool.block_decisions[key] = new_decision
                    print(f"  [*] 第 {args[i]} 组，块 {args[i+1]} ({Path(target_block.file_path).name}#{target_block.block_id}) 的决策已更新为: {new_decision}")
                    processed_count += 1

                except ValueError:
                    raise UserInputError(f"组号和块号必须是数字: '{args[i]}' 或 '{args[i+1]}' 无效。")
                except Exception as e:
                     # 捕获创建键等其他错误
                     logger.error(f"处理 {command} {args[i]} {args[i+1]} 时出错: {e}")
                     print(f"  [!] 处理组 {args[i]} 块 {args[i+1]} 时出错: {e}")
                     continue # 继续处理下一对参数
            return processed_count > 0

        elif command == 'a':
            # 应用：保留指定块，删除同组其他块
            if len(args) != 2:
                raise UserInputError("命令 'a' 需要正好两个参数: <组号> <块号>。")

            try:
                group_idx = int(args[0]) - 1
                block_idx_to_keep = int(args[1]) - 1

                if not (0 <= group_idx < len(self.md5_duplicates)):
                    raise UserInputError(f"无效的组号: {args[0]}。")

                group = self.md5_duplicates[group_idx]
                if not (0 <= block_idx_to_keep < len(group)):
                    raise UserInputError(f"无效的块号: {args[1]}。")

                print(f"  [*] 应用操作到第 {args[0]} 组:")
                for i, block in enumerate(group):
                    key = create_decision_key(block.file_path, block.block_id, block.block_type)
                    if i == block_idx_to_keep:
                        self.kd_tool.block_decisions[key] = constants.DECISION_KEEP
                        print(f"    - 保留块 {i+1} ({Path(block.file_path).name}#{block.block_id})")
                    else:
                        self.kd_tool.block_decisions[key] = constants.DECISION_DELETE
                        print(f"    - 删除块 {i+1} ({Path(block.file_path).name}#{block.block_id})")
                return True

            except ValueError:
                raise UserInputError("组号和块号必须是数字。")
            except Exception as e:
                logger.error(f"处理 'a' 命令时出错: {e}")
                print(f"  [!] 处理应用操作时出错: {e}")
                return False

        return False # 如果命令未被处理