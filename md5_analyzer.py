# [DEPENDENCIES]
# 1. Python Standard Library: collections, hashlib, time, pathlib
# 2. 同项目模块: utils, constants (使用绝对导入)

import collections
import hashlib
import time
from pathlib import Path

# 使用绝对导入
import utils
import constants

logger = utils.logger # 使用 utils 中配置好的 logger

class MD5Analyzer:
    """处理 MD5 精确重复查找和交互式决策的类。"""

    def __init__(self, kd_tool_instance):
        """
        初始化 MD5 分析器。

        Args:
            kd_tool_instance: 主 KDToolCLI 类的实例，用于访问共享数据和方法。
        """
        self.tool = kd_tool_instance # 保存主工具实例的引用
        self.duplicate_blocks = {} # {md5_hash: [block_info1, block_info2, ...]}
        self.md5_id_to_key = {} # 用于交互式界面的映射 {display_id: decision_key}

    def find_md5_duplicates(self):
        """
        计算 MD5 哈希并查找完全重复的块。
        结果存储在 self.duplicate_blocks 中。
        """
        logger.info("Calculating MD5 hashes and finding exact duplicate blocks...")
        hash_map = collections.defaultdict(list)
        start_time = time.time()
        calculated_hashes = 0

        if not self.tool.blocks_data:
            logger.warning("No blocks data available to perform MD5 analysis.")
            self.duplicate_blocks = {}
            return

        try:
            # 遍历所有已解析的块
            for block_info in self.tool.blocks_data:
                file_path, b_index, b_type, text_to_hash = block_info
                try:
                    # 计算文本的 MD5 哈希值
                    hash_object = hashlib.md5(text_to_hash.encode('utf-8'))
                    hex_dig = hash_object.hexdigest()
                    # 将块信息添加到对应哈希值的列表中
                    hash_map[hex_dig].append(block_info)
                    calculated_hashes += 1
                except Exception as hash_err:
                    # 记录哈希计算过程中的错误
                    file_name = file_path.name if isinstance(file_path, Path) else str(file_path)
                    logger.warning(f"Error hashing block: Index {b_index} in {file_name} - {hash_err}", exc_info=False)
                    continue

            logger.debug(f"Calculated {calculated_hashes} MD5 hashes.")

            # 筛选出具有多个块的哈希值（即重复项）
            self.duplicate_blocks = {h: b for h, b in hash_map.items() if len(b) > 1}
            end_time = time.time()

            num_groups = len(self.duplicate_blocks)
            num_duplicates = sum(len(b) for b in self.duplicate_blocks.values())
            logger.info(f"MD5 analysis complete. Time taken: {end_time - start_time:.2f} seconds.")

            if num_groups > 0:
                logger.info(f"Found {num_groups} groups of exact duplicates, involving {num_duplicates} block instances.")
            else:
                logger.info("No exact duplicate content blocks found.")

        except Exception as e:
            # 记录 MD5 分析过程中的意外错误
            logger.error("Error during MD5 calculation or duplicate finding.", exc_info=True)
            self.duplicate_blocks = {} # 出错时清空结果

    def _display_md5_duplicates_list(self):
        """
        在控制台显示所有 MD5 重复组，并为每个块分配唯一的显示 ID。
        更新 self.md5_id_to_key 映射。
        返回: True 如果有重复项显示，False 如果没有。
        """
        print("\n--- 精确重复内容块 (MD5) 列表 ---")
        if not self.duplicate_blocks:
            print("[*] 未找到精确重复项。")
            return False

        self.md5_id_to_key.clear() # 清空旧的映射
        group_num = 0
        print("组号 | 块ID | 文件名 (类型 #索引) | 当前决策 | 内容预览")
        print("-----|------|-----------------------|------------|----------")

        # 遍历每个重复组
        # 按第一个块的文件名和索引排序组，保证显示顺序稳定
        sorted_groups = sorted(self.duplicate_blocks.items(), key=lambda item: utils.sort_blocks_key(item[1][0]))

        # for md5_hash, block_list in self.duplicate_blocks.items():
        for md5_hash, block_list in sorted_groups:
            group_num += 1
            group_preview_shown = False # 每个组只显示一次预览

            # 遍历组内的每个块 (按原始顺序排序)
            sorted_block_list = sorted(block_list, key=utils.sort_blocks_key)
            # for i, block_info in enumerate(block_list):
            for i, block_info in enumerate(sorted_block_list):
                item_id = f"g{group_num}-{i+1}" # 创建显示 ID，例如 g1-1, g1-2
                file_path, b_index, b_type, b_text = block_info

                # 创建用于存储决策的唯一键
                # 使用 resolve() 来获取绝对路径字符串，增加键的稳定性
                try:
                    abs_path_str = str(Path(file_path).resolve())
                    key = utils.create_decision_key(abs_path_str, b_index, b_type)
                except Exception as e:
                     logger.warning(f"无法为块创建决策键 {file_path}#{b_index}: {e}. 跳过此块的显示。")
                     continue # 跳过无法生成 key 的块

                self.md5_id_to_key[item_id] = key # 存储 显示ID -> 决策键 的映射

                # 获取当前块的决策状态
                current_decision = self.tool.block_decisions.get(key, 'undecided')
                # 生成内容预览
                preview_text = utils.display_block_preview(b_text)
                # 只显示组内第一个块的预览，以减少冗余
                display_preview = preview_text if not group_preview_shown else ""
                group_preview_shown = True

                # 获取文件名字符串
                file_name_str = file_path.name if isinstance(file_path, Path) else str(file_path)

                # 打印块信息
                print(f" G{group_num} | {item_id:<4} | {file_name_str} ({b_type} #{b_index}) | {current_decision:<10} | {display_preview}")

            print("-----|------|-----------------------|------------|----------") # 组分隔线
        return True

    def review_md5_duplicates_interactive(self):
        """
        提供交互式界面，让用户处理 MD5 重复组 (列表模式)。
        用户可以标记保留 (keep)、删除 (delete) 或重置 (undecided)。
        """
        logger.info("Starting interactive review of MD5 duplicates (list mode)...")

        # 初始显示列表
        if not self._display_md5_duplicates_list():
            logger.info("No MD5 duplicates to review.")
            print("\n--- MD5 重复项处理完成 ---")
            return # 没有可审核项，直接返回

        while True:
            # 显示操作选项
            print("\n操作选项:")
            print(" k <块ID> [块ID...] - 标记指定块为 '保留 (keep)'")
            print(" d <块ID> [块ID...] - 标记指定块为 '删除 (delete)'")
            print(" r <块ID> [块ID...] - 重置指定块为 'undecided'")
            print(" a <块ID>          - 保留指定块，删除同组其他块 ('auto-delete')")
            print(" save              - 保存当前所有决策到文件")
            print(" q                 - 完成 MD5 处理并退出")

            action = input("请输入操作 (例如: k g1-1 d g1-2 g2-1): ").lower().strip()
            logger.debug(f"User input for MD5 list review: '{action}'")

            # 标记是否需要重新显示列表 (默认为 True，除非输入无效)
            redisplay_list = True

            if action == 'q':
                logger.info("Quitting interactive MD5 review (list mode).")
                break # 退出 MD5 审核
            elif action == 'save':
                logger.info("User chose 'save' during MD5 review.")
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

            if command not in ['k', 'd', 'r', 'a']:
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
                if item_id in self.md5_id_to_key:
                    valid_ids.append(item_id)
                else:
                    invalid_ids.append(item_id)

            if invalid_ids:
                logger.warning(f"Invalid block IDs provided: {invalid_ids}")
                print(f"[错误] 无效的块 ID: {invalid_ids}")
                redisplay_list = False # 无效输入
                continue

            # --- 处理 'a' (auto-delete) 命令 ---
            if command == 'a':
                if len(valid_ids) != 1:
                    logger.warning("Command 'a' requires exactly one block ID.")
                    print("[错误] 命令 'a' 需要且仅需要一个块 ID。")
                    redisplay_list = False # 无效输入
                    continue

                keep_id = valid_ids[0]
                keep_key = self.md5_id_to_key[keep_id]

                # 从块 ID 中提取组号前缀 (例如 'g1')
                group_id_prefix = keep_id.split('-')[0]
                group_member_keys = []
                group_member_ids = []

                # 找到属于同一组的所有块的 ID 和 key
                for item_id_in_map, key_in_map in self.md5_id_to_key.items():
                    if item_id_in_map.startswith(group_id_prefix):
                        group_member_keys.append(key_in_map)
                        group_member_ids.append(item_id_in_map)

                if not group_member_keys:
                    logger.error(f"Internal error: Could not find group members for ID {keep_id}")
                    redisplay_list = False # 内部错误，避免无限循环
                    continue

                logger.info(f"Processing 'a' command: Keep {keep_id} ({keep_key}), delete others in group {group_id_prefix}")

                # 标记要保留的块
                self.tool.block_decisions[keep_key] = 'keep'
                print(f" 已标记 {keep_id} 为 '保留'")

                # 标记同组的其他块为删除
                for i, member_key in enumerate(group_member_keys):
                    member_id = group_member_ids[i]
                    if member_key != keep_key:
                        self.tool.block_decisions[member_key] = 'delete'
                        print(f" 已标记 {member_id} 为 '删除'")
                        logger.info(f" Marked {member_id} ({member_key}) as 'delete'")
                # 处理完 'a' 命令后，会在下次循环开始时重新显示列表

            # --- 处理 'k', 'd', 'r' 命令 ---
            elif command in ['k', 'd', 'r']: # 明确处理 k, d, r
                for item_id in valid_ids:
                    key = self.md5_id_to_key[item_id]
                    if command == 'k':
                        self.tool.block_decisions[key] = 'keep'
                        logger.info(f"Marked {item_id} ({key}) as 'keep'.")
                        print(f" 已标记 {item_id} 为 '保留'")
                    elif command == 'r':
                        self.tool.block_decisions[key] = 'undecided'
                        logger.info(f"Reset {item_id} ({key}) to 'undecided'.")
                        print(f" 已重置 {item_id} 为 'undecided'")
                    elif command == 'd':
                        # --- 防止删除组内最后一个非删除项 ---
                        group_id_prefix = item_id.split('-')[0]
                        # 找到同组的所有 key
                        group_member_keys = [k for id_, k in self.md5_id_to_key.items() if id_.startswith(group_id_prefix)]

                        # 计算当前组内非 'delete' 状态的块的数量
                        non_delete_count = 0
                        for mk in group_member_keys:
                            if self.tool.block_decisions.get(mk, 'undecided') != 'delete':
                                non_delete_count += 1

                        # 获取当前要删除块的决策状态
                        current_decision = self.tool.block_decisions.get(key, 'undecided')
                        # 判断这个块是否是组内最后一个非删除项
                        is_last_one = (non_delete_count <= 1 and current_decision != 'delete')

                        if is_last_one:
                            logger.warning(f"Skipped deleting {item_id} ({key}) as it's the last non-delete item in its group.")
                            print(f" [跳过] {item_id} 是其组中最后一个非删除项，无法删除。")
                            # 即使跳过，也认为用户尝试了有效操作，允许重显列表
                        else:
                            # 可以安全删除
                            self.tool.block_decisions[key] = 'delete'
                            logger.info(f"Marked {item_id} ({key}) as 'delete'.")
                            print(f" 已标记 {item_id} 为 '删除'")
                        # --- 结束防止删除最后一个非删除项的逻辑 ---
                # 有效处理后，会在下一次循环开始时重新显示列表

            # *** 修改点：只有在需要时才重新显示列表 ***
            if redisplay_list:
                if not self._display_md5_duplicates_list():
                    # 如果处理完最后一个重复组，列表为空，则退出
                    logger.info("No remaining MD5 duplicates to review.")
                    break # 退出循环

        logger.info("Finished interactive review of MD5 duplicates (list mode).")
        print("\n--- MD5 重复项处理完成 ---")
