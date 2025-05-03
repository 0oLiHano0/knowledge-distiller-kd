# knowledge_distiller_kd/ui/cli_interface.py
"""
Command Line Interface (CLI) for the Knowledge Distiller tool.
Handles user interaction, displays information, and calls the core engine.
"""
import sys
import logging
import re # Import regular expressions for parsing commands
from pathlib import Path
from typing import Optional, List, Tuple, Dict # Added List, Tuple, Dict

# 使用相对导入访问 core 模块
from ..core.engine import KnowledgeDistillerEngine
from ..core import constants
from ..core.utils import logger, display_block_preview, create_decision_key # 导入需要的工具函数
# 可能还需要导入 ContentBlock 等用于类型提示或显示
from ..processing.document_processor import ContentBlock


class CliInterface:
    """Handles Command Line Interface interactions."""

    def __init__(self, engine: KnowledgeDistillerEngine):
        """
        Initialize the CLI Interface.

        Args:
            engine: An instance of the KnowledgeDistillerEngine.
        """
        self.engine = engine
        logger.info("CliInterface initialized.")

    def run(self) -> None:
        """
        Runs the main interactive loop for the CLI.
        """
        print("\n欢迎使用知识蒸馏工具 (KD Tool)！")
        logger.info("Starting interactive CLI loop.")

        while True:
            self._display_main_menu()
            choice = input("请输入选项: ").strip().lower()

            if choice == 'q':
                print("正在退出...")
                logger.info("Exiting interactive loop.")
                break
            # ==================== 添加选项处理 ====================
            elif choice == '1':
                self._handle_set_input_dir()
            elif choice == '2':
                 # Check if input dir is set before allowing analysis
                 # (Menu display logic already handles this visibility)
                 if self.engine.get_status_summary().get('input_dir', '未设置') != '未设置':
                     self._handle_run_analysis()
                 else:
                     print("[错误] 请先使用选项 '1' 设置有效的输入目录。")
            elif choice == '3':
                 if self.engine.get_status_summary().get('analysis_completed', False):
                     self._handle_view_process_duplicates()
                 else:
                     print("[错误] 请先运行分析 (选项 '2')。")
            # --- 后续添加处理 4, 5, 6, c 的逻辑 ---
            elif choice == '4':
                 if self.engine.get_status_summary().get('analysis_completed', False):
                     self._handle_load_decisions()
                 else:
                      print("[错误] 请先运行分析 (选项 '2')。")
            elif choice == '5':
                 if self.engine.get_status_summary().get('analysis_completed', False):
                     self._handle_save_decisions()
                 else:
                      print("[错误] 请先运行分析 (选项 '2')。")
            elif choice == '6':
                 if self.engine.get_status_summary().get('analysis_completed', False):
                     self._handle_apply_decisions()
                 else:
                      print("[错误] 请先运行分析 (选项 '2')。")
            elif choice == 'c':
                 self._handle_config()
            # =========================================================
            else:
                print(f"无效选项: '{choice}'. 请重新输入。")
                logger.warning(f"Invalid menu choice entered: {choice}")

        print("感谢使用 KD Tool！")


    def _display_main_menu(self) -> None:
        """Displays the main menu options and current status."""
        try: status = self.engine.get_status_summary()
        except Exception as e: logger.error(f"Failed to get status summary from engine: {e}"); status = {}
        print("\n--- KD Tool 主菜单 ---")
        print(f"当前输入目录: {status.get('input_dir', '未设置')}")
        analysis_status = "已完成" if status.get('analysis_completed', False) else "未完成"
        print(f"分析状态: {analysis_status}")
        print(f"总内容块: {status.get('total_blocks', 0)}")
        md5_groups = status.get('md5_duplicates_groups', 0)
        semantic_pairs = status.get('semantic_duplicates_pairs', 0)
        skip_semantic = status.get('skip_semantic', False)
        print(f"MD5 重复组: {md5_groups}")
        if skip_semantic: print("语义相似对: (已跳过)")
        else: print(f"语义相似对: {semantic_pairs}")
        decided_count = status.get('decided_blocks', 0)
        print(f"已决策块数: {decided_count} / {status.get('total_blocks', 0)}")
        print("\n可用操作:")
        print("1. 设置输入目录")
        if status.get('input_dir', '未设置') != '未设置': print("2. 运行分析")
        if status.get('analysis_completed', False):
            print("3. 查看/处理重复项")
            print("4. 加载决策文件")
            print("5. 保存当前决策")
            print("6. 应用决策 (生成去重文件)")
        print("c. 配置")
        print("q. 退出")
        print("--------------------")

    def _handle_set_input_dir(self) -> None:
        """Handles the 'Set Input Directory' action."""
        current_dir = self.engine.get_status_summary().get('input_dir', '未设置')
        print(f"\n当前输入目录: {current_dir}")
        path_str = input("请输入输入目录的路径: ").strip()
        if not path_str: print("[取消] 未输入路径."); return
        input_path = Path(path_str)
        success = self.engine.set_input_dir(input_path)
        if success:
             new_status = self.engine.get_status_summary()
             resolved_path_str = new_status.get('input_dir', str(input_path.resolve()))
             print(f"[*] 输入目录已成功设置为: {resolved_path_str}")
        else:
            print("[错误] 设置输入目录失败。请检查日志或路径是否有效且为目录。")

    def _handle_run_analysis(self) -> None:
        """Handles the 'Run Analysis' action."""
        print("\n[*] 正在运行分析...")
        logger.info("User triggered analysis run.")
        success = self.engine.run_analysis()
        if success: print("[*] 分析完成。"); logger.info("Analysis run completed successfully via UI.")
        else: print("[错误] 分析过程中发生错误。请检查日志获取详细信息。"); logger.error("Analysis run failed via UI.")

    def _display_duplicates_menu(self) -> None:
        """Displays the sub-menu for choosing duplicate type to review."""
        print("\n--- 查看/处理重复项 ---")
        print("m. 处理 MD5 重复项")
        print("s. 处理语义相似项")
        print("q. 返回主菜单")
        print("-----------------------")

    def _handle_view_process_duplicates(self) -> None:
        """Handles the 'View/Process Duplicates' action."""
        while True:
            self._display_duplicates_menu()
            choice = input("请选择处理类型 (m/s, q 返回): ").strip().lower()
            if choice == 'q': break
            elif choice == 'm': self.review_md5_duplicates()
            elif choice == 's': self.review_semantic_duplicates()
            else: print(f"无效选项: '{choice}'.")

    def review_md5_duplicates(self) -> None:
        """Interactive review process for MD5 duplicates."""
        duplicates: List[List[ContentBlock]] = self.engine.get_md5_duplicates()
        if not duplicates: print("\n[*] 未找到 MD5 重复项。"); return

        total_groups = len(duplicates)
        current_group_index = 0

        while True:
            if not (0 <= current_group_index < total_groups):
                logger.warning(f"MD5 review loop index out of bounds: {current_group_index}")
                break

            current_group = duplicates[current_group_index]
            group_size = len(current_group)
            print(f"\n--- MD5 重复项审查 (共 {total_groups} 组) ---")
            print(f"组 {current_group_index + 1} / {total_groups} (共 {group_size} 项):")

            for i, block in enumerate(current_group):
                 try:
                     key = create_decision_key(str(Path(block.file_path).resolve()), block.block_id, block.block_type)
                     decision = self.engine.block_decisions.get(key, constants.DECISION_UNDECIDED)
                 except Exception as key_err:
                     logger.error(f"Error getting decision for block {block.block_id}: {key_err}")
                     decision = "错误"
                 decision_marker = f"[{decision.upper()}]" if decision != constants.DECISION_UNDECIDED else "[ ]"
                 display_path = Path(block.file_path).resolve()
                 print(f"  {i+1}. {decision_marker} {display_path} # {block.block_id} ({block.block_type})")
                 preview = block.original_text[:constants.PREVIEW_LENGTH].replace('\n', ' ') + ('...' if len(block.original_text) > constants.PREVIEW_LENGTH else '')
                 print(f"      预览: {preview}")

            action = input("操作 (k[索引] 保留, d[索引] 删除, a 全删, n 下一组, p 上一组, q 退出): ").strip().lower()

            if action == 'q': break
            elif action == 'n':
                if current_group_index < total_groups - 1: current_group_index += 1
                else: print("[提示] 已经是最后一组。")
            elif action == 'p':
                if current_group_index > 0: current_group_index -= 1
                else: print("[提示] 已经是第一组。")
            elif action.startswith('k') or action.startswith('d'):
                match = re.match(r"([kd])(\d+)", action)
                if match:
                    cmd, index_str = match.groups()
                    try:
                        index = int(index_str) - 1
                        if 0 <= index < group_size:
                            target_block = current_group[index]
                            decision = constants.DECISION_KEEP if cmd == 'k' else constants.DECISION_DELETE
                            try:
                                key = create_decision_key(str(Path(target_block.file_path).resolve()), target_block.block_id, target_block.block_type)
                                if self.engine.update_decision(key, decision):
                                    print(f"[*] 已将 {index + 1} 标记为 [{decision.upper()}]")
                                    if hasattr(self.engine, 'block_decisions'): self.engine.block_decisions[key] = decision
                                else: print(f"[错误] 更新决策失败: {key}")
                            except Exception as e: logger.error(f"Error creating key or updating decision for block {index+1}: {e}"); print(f"[错误] 处理块 {index + 1} 时出错。")
                        else: print(f"[错误] 无效的索引: {index + 1} (应在 1 到 {group_size} 之间)")
                    except ValueError: print(f"[错误] 无效的索引格式: '{index_str}'")
                else: print(f"无效操作格式: '{action}'. 使用 k[索引] 或 d[索引].")
            elif action == 'a':
                print("[*] 正在将除第一个之外的所有项标记为 [DELETE]...")
                success_count = 0; fail_count = 0
                for i in range(1, group_size):
                    block_to_delete = current_group[i]
                    try:
                        key = create_decision_key(str(Path(block_to_delete.file_path).resolve()), block_to_delete.block_id, block_to_delete.block_type)
                        if self.engine.update_decision(key, constants.DECISION_DELETE):
                            success_count += 1
                            if hasattr(self.engine, 'block_decisions'): self.engine.block_decisions[key] = constants.DECISION_DELETE
                        else: fail_count += 1; print(f"[警告] 更新块 {i+1} 的决策失败: {key}")
                    except Exception as e: fail_count += 1; logger.error(f"Error creating key or updating decision for block {i+1} in 'delete all': {e}"); print(f"[错误] 处理块 {i + 1} 时出错。")
                print(f"[*] 操作完成: {success_count} 个已标记, {fail_count} 个失败。")
            else:
                print(f"无效操作: '{action}'")

    # ==================== 实现语义审查框架 ====================
    def review_semantic_duplicates(self) -> None:
        """Interactive review process for semantic duplicates."""
        pairs: List[Tuple[ContentBlock, ContentBlock, float]] = self.engine.get_semantic_duplicates()
        if not pairs: print("\n[*] 未找到语义相似对。"); return

        total_pairs = len(pairs)
        current_pair_index = 0

        while True:
            if not (0 <= current_pair_index < total_pairs):
                logger.warning(f"Semantic review loop index out of bounds: {current_pair_index}")
                break

            block1, block2, similarity = pairs[current_pair_index]
            print(f"\n--- 语义相似项审查 (共 {total_pairs} 对) ---")
            print(f"对 {current_pair_index + 1} / {total_pairs} (相似度: {similarity:.4f}):")

            # Display block 1 info
            try:
                key1 = create_decision_key(str(Path(block1.file_path).resolve()), block1.block_id, block1.block_type)
                decision1 = self.engine.block_decisions.get(key1, constants.DECISION_UNDECIDED)
            except Exception as e: logger.error(f"Error getting decision for block1 {block1.block_id}: {e}"); decision1 = "错误"
            decision_marker1 = f"[{decision1.upper()}]" if decision1 != constants.DECISION_UNDECIDED else "[ ]"
            print(f"  1. {decision_marker1} {Path(block1.file_path).resolve()} # {block1.block_id} ({block1.block_type})")
            preview1 = block1.original_text[:constants.PREVIEW_LENGTH].replace('\n', ' ') + ('...' if len(block1.original_text) > constants.PREVIEW_LENGTH else '')
            print(f"      预览: {preview1}")

            # Display block 2 info
            try:
                key2 = create_decision_key(str(Path(block2.file_path).resolve()), block2.block_id, block2.block_type)
                decision2 = self.engine.block_decisions.get(key2, constants.DECISION_UNDECIDED)
            except Exception as e: logger.error(f"Error getting decision for block2 {block2.block_id}: {e}"); decision2 = "错误"
            decision_marker2 = f"[{decision2.upper()}]" if decision2 != constants.DECISION_UNDECIDED else "[ ]"
            print(f"  2. {decision_marker2} {Path(block2.file_path).resolve()} # {block2.block_id} ({block2.block_type})")
            preview2 = block2.original_text[:constants.PREVIEW_LENGTH].replace('\n', ' ') + ('...' if len(block2.original_text) > constants.PREVIEW_LENGTH else '')
            print(f"      预览: {preview2}")

            action = input("操作 (k1/k2 保留, d1/d2 删除, skip 跳过, n 下一对, p 上一对, q 退出): ").strip().lower()

            if action == 'q': break
            elif action == 'n':
                if current_pair_index < total_pairs - 1: current_pair_index += 1
                else: print("[提示] 已经是最后一对。")
            elif action == 'p':
                if current_pair_index > 0: current_pair_index -= 1
                else: print("[提示] 已经是第一对。")
            elif action == 'skip':
                print("[*] 跳过当前对。")
                if current_pair_index < total_pairs - 1: current_pair_index += 1
                else: print("[提示] 已经是最后一对。") # Optionally loop back or exit? For now, just stays.
            # ==================== 添加决策处理逻辑 ====================
            elif action.startswith('k') or action.startswith('d'):
                 match = re.match(r"([kd])([12])", action)
                 if match:
                     cmd, index_str = match.groups()
                     index = int(index_str) - 1 # 0 for block1, 1 for block2
                     target_block = block1 if index == 0 else block2
                     decision = constants.DECISION_KEEP if cmd == 'k' else constants.DECISION_DELETE
                     try:
                         # Ensure path is absolute for consistent key generation
                         key = create_decision_key(str(Path(target_block.file_path).resolve()), target_block.block_id, target_block.block_type)
                         if self.engine.update_decision(key, decision):
                             print(f"[*] 已将块 {index + 1} 标记为 [{decision.upper()}]")
                             # Update mock engine state for display in tests
                             if hasattr(self.engine, 'block_decisions'): self.engine.block_decisions[key] = decision
                         else:
                             print(f"[错误] 更新决策失败: {key}")
                     except Exception as e:
                         logger.error(f"Error creating key or updating decision for semantic block {index+1}: {e}")
                         print(f"[错误] 处理块 {index + 1} 时出错。")
                 else:
                     print(f"无效操作格式: '{action}'. 使用 k1, k2, d1, d2.")
            # =========================================================
            else:
                print(f"无效操作: '{action}'")
    # =========================================================

    # ==================== 添加最终处理方法 ====================
    def _handle_load_decisions(self) -> None:
        """Handles the 'Load Decisions' action."""
        print("\n[*] 正在加载决策文件...")
        logger.info("User triggered load decisions.")
        self.engine.load_decisions()

    def _handle_save_decisions(self) -> None:
        """Handles the 'Save Decisions' action."""
        print("\n[*] 正在保存当前决策...")
        logger.info("User triggered save decisions.")
        self.engine.save_decisions()

    def _handle_apply_decisions(self) -> None:
        """Handles the 'Apply Decisions' action."""
        print("\n[*] 正在应用决策生成输出文件...")
        logger.info("User triggered apply decisions.")
        self.engine.apply_decisions()

    def _display_config_menu(self) -> None:
        """Displays the configuration sub-menu."""
        status = self.engine.get_status_summary()
        print("\n--- 配置菜单 ---")
        print(f"1. 切换语义分析: {'已跳过' if status.get('skip_semantic') else '已启用'}")
        print(f"2. 设置相似度阈值 (当前: {status.get('similarity_threshold', constants.DEFAULT_SIMILARITY_THRESHOLD):.2f})")
        print("q. 返回主菜单")
        print("----------------")

    def _handle_config(self) -> None:
        """Handles the configuration sub-menu."""
        while True:
            self._display_config_menu()
            choice = input("请选择配置项: ").strip().lower()

            if choice == 'q': break
            elif choice == '1':
                current_skip = self.engine.get_status_summary().get('skip_semantic', False)
                new_skip = not current_skip
                self.engine.set_skip_semantic(new_skip)
                print(f"[*] 语义分析已 {'跳过' if new_skip else '启用'}.")
            elif choice == '2':
                current_threshold = self.engine.get_status_summary().get('similarity_threshold', constants.DEFAULT_SIMILARITY_THRESHOLD)
                threshold_str = input(f"请输入新的相似度阈值 (0.0-1.0, 当前: {current_threshold:.2f}): ").strip()
                try:
                    new_threshold = float(threshold_str)
                    if 0.0 <= new_threshold <= 1.0:
                        if self.engine.set_similarity_threshold(new_threshold):
                            print(f"[*] 相似度阈值已设置为: {new_threshold:.2f}")
                        else: print("[错误] 设置阈值失败。")
                    else: print("[错误] 阈值必须在 0.0 到 1.0 之间。")
                except ValueError: print(f"[错误] 无效的数字格式: '{threshold_str}'")
            else: print(f"无效选项: '{choice}'.")
    # =========================================================

