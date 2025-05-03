# KD_Tool_CLI/knowledge_distiller_kd/analysis/md5_analyzer.py (Refactored)
"""
MD5分析器模块，用于检测MD5重复内容。
(Refactored to be independent of KDToolCLI/Engine)
"""
# --- 标准库导入 ---
import logging
import collections
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, DefaultDict, Set
from collections import defaultdict

# --- 项目内部模块导入 ---
# 使用相对导入 (..) 来访问上级目录中的 core 和 processing
from ..core.error_handler import KDError, AnalysisError, handle_error
from ..core.utils import logger, create_decision_key, parse_decision_key, display_block_preview # 导入需要的工具函数
from ..core import constants
from ..processing.document_processor import ContentBlock # 导入 ContentBlock

# --- 类定义开始 ---
logger = logger # 使用 utils 中配置好的 logger

class MD5Analyzer:
    """
    MD5分析器类，用于检测完全相同的内容块。
    (Refactored: Does not hold state related to specific runs, operates on input data)
    """

    def __init__(self):
        """
        初始化MD5分析器。
        (Refactored: No longer takes kd_tool instance)
        """
        # 不再存储 kd_tool 或重复结果的状态
        pass

    def find_md5_duplicates(
        self,
        blocks_data: List[ContentBlock],
        current_decisions: Dict[str, str]
    ) -> Tuple[List[List[ContentBlock]], Dict[str, str]]:
        """
        查找具有相同MD5哈希值的内容块。
        优化：跳过标题块和已被标记为删除的块。

        Args:
            blocks_data: 要分析的 ContentBlock 列表。
            current_decisions: 当前的决策字典 (key -> decision)，用于跳过已删除块。

        Returns:
            Tuple[List[List[ContentBlock]], Dict[str, str]]:
                - 第一个元素是找到的 MD5 重复组列表 [[block1, block2], [blockA, blockB], ...]
                - 第二个元素是建议的决策更新字典 {key_to_keep: "keep", key_to_delete: "delete", ...}
        """
        logger.info("Starting MD5 duplicate detection...")
        duplicate_groups: List[List[ContentBlock]] = []
        suggested_decisions: Dict[str, str] = {}

        if not blocks_data:
            logger.warning("MD5 analysis skipped: No blocks provided.")
            return duplicate_groups, suggested_decisions # 返回空结果

        try:
            # 按MD5哈希值分组
            hash_groups: DefaultDict[str, List[ContentBlock]] = defaultdict(list)
            total_blocks = len(blocks_data)
            processed_count = 0
            skipped_titles = 0
            skipped_code_fences = 0 # 保留这个跳过逻辑，虽然合并后可能不太需要
            skipped_deleted = 0

            logger.info(f"开始MD5重复检测，共有{total_blocks}个内容块")
            logger.debug(f"传入的当前决策数量: {len(current_decisions)}")

            # 计算每个块的哈希值
            for block in blocks_data:
                if not isinstance(block, ContentBlock):
                    logger.warning(f"Skipping item as it's not a ContentBlock: {type(block)}")
                    continue

                # 跳过标题块
                if block.block_type == constants.BLOCK_TYPE_TITLE:
                    logger.debug(f"Skipping Title block for MD5 analysis: {block.file_path}#{block.block_id}")
                    skipped_titles += 1
                    continue

                # --- 检查当前决策 ---
                try:
                    # 使用绝对路径创建 key 来检查决策
                    # 假设 blocks_data 中的 file_path 已经是可靠的（最好是绝对路径）
                    # 如果不是，Engine 在调用此方法前需要确保路径一致性
                    key = create_decision_key(
                        str(Path(block.file_path).resolve()), # 确保使用绝对路径查找决策
                        block.block_id,
                        block.block_type
                    )
                except Exception as e:
                    logger.error(f"Error creating decision key for block {block.block_id} in {block.file_path}: {e}")
                    continue # 跳过无法创建键的块

                # 从传入的 current_decisions 获取决策
                decision = current_decisions.get(key, constants.DECISION_UNDECIDED)
                if decision == constants.DECISION_DELETE:
                    logger.debug(f"Skipping block already marked for deletion: {key}")
                    skipped_deleted += 1
                    continue
                # 注意：这里不再跳过 DECISION_KEEP 的块，因为它们仍需参与比较以识别其他重复项
                # --------------------

                # 获取用于计算哈希的文本 (来自 ContentBlock 的 analysis_text)
                text_to_hash = block.analysis_text

                # 跳过仅包含代码块结束符的块 (这个逻辑在合并后可能冗余，但保留无害)
                if block.block_type == constants.BLOCK_TYPE_CODE and text_to_hash.strip() == '```':
                    logger.debug(f"Skipping block containing only code fence end: {key}")
                    skipped_code_fences += 1
                    continue

                # 计算MD5哈希值，包含块类型
                if not isinstance(text_to_hash, str):
                    logger.warning(f"Analysis text for block {key} is not a string: {type(text_to_hash)}. Skipping.")
                    continue

                block_type_str = str(block.block_type)
                # 使用strip()确保比较的一致性
                hash_input = f"{block_type_str}:{text_to_hash.strip()}"
                try:
                    md5_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
                    hash_groups[md5_hash].append(block) # 将 ContentBlock 对象加入分组
                except Exception as e:
                     logger.error(f"Error calculating hash for block {key}: {e}")

                processed_count += 1

            logger.info(f"MD5 hash calculation complete. Skipped titles: {skipped_titles}, skipped code fences: {skipped_code_fences}, skipped deleted: {skipped_deleted}")
            logger.info(f"Hash calculation resulted in {len(hash_groups)} unique hash groups (excluding titles, etc.).")

            # 找出重复的组并生成建议决策
            duplicate_group_count = 0
            for md5_hash, blocks in hash_groups.items():
                if len(blocks) > 1:
                    duplicate_group_count += 1
                    logger.info(f"Found MD5 duplicate group {duplicate_group_count}, hash: {md5_hash}, blocks: {len(blocks)}")

                    # 对组内块进行排序，以确定保留哪一个（例如按文件路径和ID）
                    try:
                        # 确保使用绝对路径进行排序比较
                        blocks.sort(key=lambda b: (str(Path(b.file_path).resolve()), str(b.block_id)))
                    except Exception as e:
                         logger.error(f"Error sorting duplicate blocks for hash {md5_hash}: {e}. Skipping suggestions for this group.")
                         continue # 跳过这个组的决策建议

                    duplicate_groups.append(blocks) # 添加找到的重复组

                    # --- 生成决策建议 ---
                    if not blocks: continue # 不太可能，但做个检查

                    # 保留第一个块 (如果它当前不是 'delete')
                    first_block = blocks[0]
                    first_key = create_decision_key(str(Path(first_block.file_path).resolve()), first_block.block_id, first_block.block_type)
                    if current_decisions.get(first_key, constants.DECISION_UNDECIDED) != constants.DECISION_DELETE:
                        suggested_decisions[first_key] = constants.DECISION_KEEP
                        logger.debug(f"Suggesting KEEP for block: {first_key}")
                    else:
                         logger.debug(f"First block {first_key} in group {md5_hash} was already marked delete, not suggesting KEEP.")


                    # 删除其他块 (如果它们当前是 'undecided')
                    for block_to_delete in blocks[1:]:
                        delete_key = create_decision_key(str(Path(block_to_delete.file_path).resolve()), block_to_delete.block_id, block_to_delete.block_type)
                        # 只建议删除当前未决定的块，避免覆盖用户手动设置的 KEEP
                        if current_decisions.get(delete_key, constants.DECISION_UNDECIDED) == constants.DECISION_UNDECIDED:
                            suggested_decisions[delete_key] = constants.DECISION_DELETE
                            logger.debug(f"Suggesting DELETE for block: {delete_key}")
                        else:
                             logger.debug(f"Block {delete_key} in group {md5_hash} already has a decision ({current_decisions.get(delete_key)}), not suggesting DELETE.")
                    # --------------------

            if duplicate_groups:
                logger.info(f"MD5 analysis finished. Found {len(duplicate_groups)} groups of exact duplicates (excluding titles).")
            else:
                logger.info("MD5 analysis finished. No exact duplicates found (excluding titles).")

            return duplicate_groups, suggested_decisions

        except Exception as e:
            logger.error(f"Unexpected error during MD5 duplicate finding: {e}", exc_info=True)
            handle_error(e, "finding MD5 duplicates")
            # 在发生错误时返回空结果
            return [], {}

    # --- UI related methods removed ---
    # _display_md5_duplicates_list removed
    # review_md5_duplicates_interactive removed
    # _process_user_action_md5 removed
