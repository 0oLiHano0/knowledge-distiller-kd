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
import re

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

# [DEPENDENCIES]
# 1. Python Standard Library: collections, hashlib, time
# 2. 同项目模块: constants, utils, error_handler (使用绝对导入)

# 使用 utils 中配置好的 logger
logger = logger

class MD5Analyzer:
    """
    MD5分析器类，用于查找和处理MD5重复内容。

    该类负责：
    1. 计算内容块的MD5哈希值
    2. 查找具有相同哈希值的内容块
    3. 生成重复内容的报告
    4. 管理重复内容的决策

    Attributes:
        tool: KDToolCLI 实例的引用
        duplicate_blocks: 存储重复内容块的字典 {hash: [(file, index, type, text), ...]}
        md5_id_to_key: 存储 MD5 ID 到决策键的映射 {id: key}
        md5_duplicates: 存储MD5重复内容的列表
    """

    def __init__(self, tool) -> None:
        """
        初始化 MD5 分析器。

        Args:
            tool: KDToolCLI 实例的引用
        """
        self.tool = tool
        self.duplicate_blocks = {}
        self.md5_id_to_key = {}
        self.md5_duplicates = []

    def find_md5_duplicates(self) -> bool:
        """
        使用MD5哈希查找重复内容。

        Returns:
            bool: 如果成功找到重复内容返回 True，否则返回 False

        Raises:
            AnalysisError: 当分析过程失败时
        """
        try:
            if not self.tool.blocks_data:
                return False  # 没有内容块时返回False而不是抛出异常

            # 初始化哈希映射
            hash_map: DefaultDict[str, List[Tuple[Union[str, Path], Union[int, str], str, str]]] = collections.defaultdict(list)
            self.md5_id_to_key.clear()  # 清空旧的映射
            block_id = 1  # 用于生成块ID

            # 遍历所有块
            for block_info in self.tool.blocks_data:
                file_path, b_index, b_type, text_to_hash = block_info
                try:
                    # 标准化文本内容
                    # 1. 按行分割
                    lines = text_to_hash.split('\n')
                    # 2. 处理每一行，保留代码块的结构
                    normalized_lines = []
                    in_code_block = False
                    code_block_lines = []
                    
                    for line in lines:
                        # 检测代码块开始
                        if line.strip().startswith('```'):
                            if not in_code_block:
                                # 开始新的代码块
                                in_code_block = True
                                code_block_lines = [line]
                            else:
                                # 结束当前代码块
                                in_code_block = False
                                code_block_lines.append(line)
                                # 将整个代码块作为一个单元添加
                                normalized_lines.append('\n'.join(code_block_lines))
                                code_block_lines = []
                            continue
                        
                        if in_code_block:
                            # 在代码块内，收集所有行
                            code_block_lines.append(line)
                        else:
                            # 非代码块内容，按原有逻辑处理
                            if line.strip():
                                normalized_lines.append(line)
                            elif normalized_lines and normalized_lines[-1].strip():
                                normalized_lines.append(line)
                    
                    # 3. 合并处理后的行，保留换行符
                    normalized_text = '\n'.join(normalized_lines)
                    
                    # 4. 移除开头和结尾的连续空行
                    normalized_text = normalized_text.strip('\n')
                    
                    # 5. 组合块类型和标准化文本
                    combined_text = f"{b_type}:{normalized_text}"

                    # 计算MD5哈希
                    hash_object = hashlib.md5(combined_text.encode('utf-8'))
                    hex_dig = hash_object.hexdigest()

                    # 更新哈希映射
                    hash_map[hex_dig].append(block_info)

                    # 创建决策键映射
                    try:
                        abs_path_str = str(Path(file_path).resolve())
                        key = create_decision_key(abs_path_str, b_index, b_type)
                        display_id = f"b{block_id}"  # 使用简单的递增ID
                        self.md5_id_to_key[display_id] = key
                        block_id += 1
                    except Exception as e:
                        logger.warning(f"创建决策键失败: {e}")
                        continue

                except Exception as e:
                    logger.warning(f"计算哈希失败: {e}")
                    continue

            # 筛选重复内容
            self.duplicate_blocks = {
                h: b for h, b in hash_map.items()
                if len(b) > 1
            }

            # 更新md5_duplicates列表
            self.md5_duplicates = []
            for blocks in self.duplicate_blocks.values():
                self.md5_duplicates.extend(blocks)

            logger.info(f"计算了 {block_id-1} 个MD5哈希")
            logger.info(f"找到 {len(self.duplicate_blocks)} 组重复内容")

            return len(self.duplicate_blocks) > 0

        except Exception as e:
            handle_error(e, "查找MD5重复内容")
            raise AnalysisError(
                "查找MD5重复内容失败",
                error_code="FIND_MD5_DUPLICATES_FAILED",
                details={"error": str(e)}
            )

    def review_md5_duplicates_interactive(self) -> None:
        """
        交互式处理MD5重复内容。

        Raises:
            UserInputError: 当用户输入无效时
            AnalysisError: 当处理过程失败时
        """
        try:
            if not self.duplicate_blocks:
                logger.info("没有找到重复内容")
                return

            # 显示重复内容并处理用户输入
            while True:
                self._display_md5_duplicates_list()

                # 获取用户输入
                action = input("\n请输入操作 (例如: k b1 d b2): ").lower().strip()
                if action == 'q':
                    break

                try:
                    # 处理用户输入
                    if not self._process_user_action(action):
                        print("[错误] 无效的操作，请重试。")
                        continue

                except Exception as e:
                    handle_error(e, "处理用户操作")
                    print(f"[错误] 处理操作时出错: {e}")
                    continue

        except Exception as e:
            handle_error(e, "处理MD5重复内容")
            raise AnalysisError(
                "处理MD5重复内容失败",
                error_code="REVIEW_MD5_DUPLICATES_FAILED",
                details={"error": str(e)}
            )

    def _display_md5_duplicates_list(self) -> bool:
        """
        显示MD5重复内容列表。

        Returns:
            bool: 如果成功显示返回 True，否则返回 False

        Raises:
            AnalysisError: 当显示过程失败时
        """
        try:
            if not self.duplicate_blocks:
                logger.info("没有重复内容可显示")
                return False

            # 显示操作说明
            print("\n操作说明:")
            print(" k <块ID> - 保留指定块")
            print(" d <块ID> - 删除指定块")
            print(" r <块ID> - 重置指定块的决策")
            print(" a <块ID> - 保留指定块，删除同组其他块")
            print(" save     - 保存当前决策")
            print(" q        - 退出")

            # 显示重复内容
            for i, (hash_value, blocks) in enumerate(self.duplicate_blocks.items(), 1):
                print(f"\n重复组 #{i}:")
                for block in blocks:
                    file_path, b_index, b_type, text = block
                    # 查找该块对应的ID
                    block_key = create_decision_key(str(Path(file_path).resolve()), b_index, b_type)
                    display_id = next(
                        (bid for bid, key in self.md5_id_to_key.items() if key == block_key),
                        "未知ID"
                    )
                    decision = self.tool.block_decisions.get(block_key, 'undecided')
                    print(f"  [{display_id}] {file_path.name}#{b_index} ({b_type}) - {decision}")
                    # 显示内容预览
                    preview = text[:100] + "..." if len(text) > 100 else text
                    print(f"      {preview}")

            return True

        except Exception as e:
            handle_error(e, "显示MD5重复内容")
            raise AnalysisError(
                "显示MD5重复内容失败",
                error_code="DISPLAY_MD5_DUPLICATES_FAILED",
                details={"error": str(e)}
            )

    def _process_user_action(self, action: str) -> bool:
        """
        处理用户的操作输入。

        Args:
            action: 用户输入的操作字符串

        Returns:
            bool: 如果操作有效返回 True，否则返回 False

        Raises:
            UserInputError: 当输入无效时
        """
        try:
            if not action:
                raise UserInputError("操作不能为空")

            # 解析操作
            parts = action.split()
            if not parts:
                raise UserInputError("无效的操作格式")

            command = parts[0]
            item_ids = parts[1:] if len(parts) > 1 else []

            # 验证命令
            if command not in ['k', 'd', 'r', 'a', 'save']:
                raise UserInputError(f"无效的命令: {command}")

            # 处理保存命令
            if command == 'save':
                if self.tool.save_decisions():
                    print("[*] 决策已保存")
                    return True
                else:
                    raise UserInputError("保存决策失败")

            # 验证块ID
            if not item_ids:
                raise UserInputError("需要至少一个块ID")

            # 处理每个块ID
            for item_id in item_ids:
                if item_id not in self.md5_id_to_key:
                    raise UserInputError(f"无效的块ID: {item_id}")

                key = self.md5_id_to_key[item_id]
                if command == 'k':
                    self.tool.block_decisions[key] = 'keep'
                elif command == 'd':
                    self.tool.block_decisions[key] = 'delete'
                elif command == 'r':
                    self.tool.block_decisions[key] = 'undecided'
                elif command == 'a':
                    # 获取组号前缀
                    group_prefix = item_id.split('_')[0]
                    # 找到同组的所有块
                    group_keys = [
                        k for k, v in self.md5_id_to_key.items()
                        if k.startswith(group_prefix)
                    ]
                    # 保留当前块，删除其他块
                    for group_key in group_keys:
                        if group_key == item_id:
                            self.tool.block_decisions[self.md5_id_to_key[group_key]] = 'keep'
                        else:
                            self.tool.block_decisions[self.md5_id_to_key[group_key]] = 'delete'

            return True

        except Exception as e:
            handle_error(e, "处理用户操作")
            raise UserInputError(
                "处理用户操作失败",
                error_code="PROCESS_USER_ACTION_FAILED",
                details={"error": str(e)}
            )
