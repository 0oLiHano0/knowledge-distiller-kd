import sys
import os
from pathlib import Path
import hashlib
import collections
import mistune
import json

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QTextEdit, QLabel, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

# 定义主窗口类
class KDToolWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("知识蒸馏 KD Tool v1.4 - Better Output") # Update title
        self.setGeometry(200, 200, 750, 650)

        # --- 定义颜色常量 ---
        self.color_keep = QColor("lightgreen")
        self.color_delete = QColor(255, 200, 200) # Light red
        self.color_undecided = None

        # --- 创建界面控件 (No changes here) ---
        self.folder_label = QLabel("目标文件夹:")
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setReadOnly(True)
        self.browse_button = QPushButton("选择文件夹...")
        self.start_button = QPushButton("开始分析")
        self.status_label = QLabel("重复项列表 (可勾选):")
        self.results_table = QTableWidget()
        self.setup_results_table()
        self.keep_button = QPushButton("标记选中项为 '保留'")
        self.delete_button = QPushButton("标记选中项为 '删除'")
        self.save_button = QPushButton("保存决策...")
        self.load_button = QPushButton("加载决策...")
        self.apply_button = QPushButton("应用决策 (生成新文件)...")

        # --- 设置布局 (No changes here) ---
        main_layout = QVBoxLayout(self)
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(self.folder_path_edit)
        folder_layout.addWidget(self.browse_button)
        main_layout.addLayout(folder_layout)
        main_layout.addWidget(self.start_button)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.results_table)
        action_button_layout = QHBoxLayout()
        action_button_layout.addWidget(self.keep_button)
        action_button_layout.addWidget(self.delete_button)
        action_button_layout.addStretch(1)
        action_button_layout.addWidget(self.load_button)
        action_button_layout.addWidget(self.save_button)
        action_button_layout.addWidget(self.apply_button)
        main_layout.addLayout(action_button_layout)

        # --- 连接信号与槽 (No changes here) ---
        self.browse_button.clicked.connect(self.browse_folder)
        self.start_button.clicked.connect(self.start_analysis)
        self.keep_button.clicked.connect(self.mark_selected_keep)
        self.delete_button.clicked.connect(self.mark_selected_delete)
        self.save_button.clicked.connect(self.save_decisions)
        self.load_button.clicked.connect(self.load_decisions)
        self.apply_button.clicked.connect(self.apply_decisions)

        # --- 类成员变量 (No changes here) ---
        self.markdown_files_content = {}
        self.blocks_data = []
        self.duplicate_blocks = {}
        self.row_to_block_info = {}
        self.block_decisions = {}

    def setup_results_table(self):
        """Configures the QTableWidget."""
        # (No changes needed here)
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(["选择", "重复组", "文件名", "类型", "块索引", "内容预览"])
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSortingEnabled(True)
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.results_table.setAlternatingRowColors(True)

    # --- 定义槽函数 ---

    def browse_folder(self):
        """Opens a directory selection dialog."""
        # (No changes needed here)
        folder_path = QFileDialog.getExistingDirectory(self, "选择包含 Markdown 文件的文件夹")
        if folder_path:
            self.folder_path_edit.setText(folder_path)
            self.results_table.setRowCount(0)
            self.row_to_block_info = {}
            self.block_decisions = {}
            print(f"已选择文件夹: {folder_path}")
        else:
            print("用户取消选择文件夹。")

    def start_analysis(self):
        """Finds MD files, reads, parses, finds MD5 duplicates, populates table."""
        # (No changes needed here)
        # ... (Reset variables, find/read files, parse blocks, find duplicates, populate table) ...
        selected_folder_str = self.folder_path_edit.text()
        self.markdown_files_content = {}
        self.blocks_data = []
        self.duplicate_blocks = {}
        self.results_table.setRowCount(0)
        self.row_to_block_info = {}
        self.block_decisions = {}

        if not selected_folder_str: print("[错误] 请先选择一个文件夹！"); return
        selected_folder = Path(selected_folder_str)
        if not selected_folder.is_dir(): print(f"[错误] 选择的路径不是一个有效的文件夹: {selected_folder_str}"); return

        print(f"开始分析文件夹: {selected_folder}\n---")
        # 1. Find/Read
        try:
            markdown_files = list(selected_folder.glob('*.md'))
            print(f"找到 {len(markdown_files)} 个 Markdown 文件，开始读取内容...")
            files_read_count = 0
            for md_file in markdown_files:
                try: content = md_file.read_text(encoding='utf-8'); self.markdown_files_content[md_file] = content; files_read_count += 1
                except Exception as e: print(f"[错误] 读取文件时出错: {md_file.name} - {e}")
            print(f"内容读取完成：成功读取 {files_read_count} / {len(markdown_files)} 个文件。")
            if files_read_count == 0: return
        except Exception as e: print(f"查找文件时出错: {e}"); return
        # 2. Parse
        print("---")
        print("开始解析 Markdown 并提取内容块...")
        self.blocks_data = []
        markdown_parser = mistune.create_markdown(renderer=None)
        try:
            self.processed_files_in_run = set() # Ensure this is tracked
            for md_file_path, content in self.markdown_files_content.items():
                self.processed_files_in_run.add(md_file_path)
                block_tokens = markdown_parser(content)
                for index, token in enumerate(block_tokens):
                    block_type = token['type']; block_text = ""; blocks_to_add = []
                    if block_type == 'paragraph': block_text = self._extract_text_from_children(token.get('children', []));
                    elif block_type == 'heading': block_text = self._extract_text_from_children(token.get('children', []));
                    elif block_type == 'block_code': block_text = token.get('raw') or token.get('text', '');
                    elif block_type == 'block_quote': block_text = self._extract_text_from_children(token.get('children', []));
                    elif block_type == 'list':
                        list_items = token.get('children', [])
                        for item_index, item_token in enumerate(list_items):
                            if item_token.get('type') == 'list_item':
                                item_text = self._extract_text_from_children(item_token.get('children', [])); cleaned_item_text = item_text.strip()
                                if cleaned_item_text: blocks_to_add.append((md_file_path, index, 'list_item', cleaned_item_text))
                    cleaned_text = block_text.strip() if block_type not in ['block_code', 'list'] else block_text
                    if cleaned_text and block_type not in ['list']: blocks_to_add.append((md_file_path, index, block_type, cleaned_text))
                    if blocks_to_add: self.blocks_data.extend(blocks_to_add)
            total_blocks = len(self.blocks_data)
            print(f"Markdown 解析完成：共获得 {total_blocks} 个内容块。")
        except Exception as e: print(f"解析 Markdown 时出错: {e}"); return
        # 3. MD5 Duplicates
        print("---")
        print("开始计算 MD5 并查找重复内容块...")
        hash_map = collections.defaultdict(list)
        try:
            for block_info in self.blocks_data: self.block_decisions[block_info] = 'undecided' # Initialize decisions
            for block_info in self.blocks_data: hash_object = hashlib.md5(block_info[3].encode('utf-8')); hex_dig = hash_object.hexdigest(); hash_map[hex_dig].append(block_info)
            self.duplicate_blocks = {h: b for h, b in hash_map.items() if len(b) > 1}
            print(f"找到 {len(self.duplicate_blocks)} 组完全重复的内容块。")
            # 4. Populate Table
            self.results_table.setSortingEnabled(False); self.results_table.setRowCount(0); current_row = 0; group_id_counter = 1; self.row_to_block_info = {}
            if not self.duplicate_blocks: print("未找到完全重复的内容块。")
            else:
                print("开始填充重复项表格...")
                for hash_val, b_list in self.duplicate_blocks.items():
                    group_id_str = f"组 {group_id_counter}"
                    for block_info in b_list: self._populate_table_row(current_row, group_id_str, block_info); self.row_to_block_info[current_row] = block_info; current_row += 1
                    group_id_counter += 1
            self.results_table.setSortingEnabled(True); print("表格填充完成。"); print("---"); print("MD5 去重分析完成。")
        except Exception as e: print(f"计算 MD5、查找或填充表格时出错: {e}")

    def _populate_table_row(self, row, group_id_str, block_info):
        """Helper method to populate a single row in the results table."""
        # (No changes needed here)
        file_path, b_index, b_type, b_text = block_info
        preview_text = (b_text[:100] + '...') if len(b_text) > 100 else b_text; preview_text = preview_text.replace('\n', ' ')
        checkbox_item = QTableWidgetItem(); checkbox_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled); checkbox_item.setCheckState(Qt.CheckState.Unchecked)
        item_group = QTableWidgetItem(group_id_str); item_file = QTableWidgetItem(file_path.name); item_type = QTableWidgetItem(b_type); item_index = QTableWidgetItem(str(b_index)); item_preview = QTableWidgetItem(preview_text)
        self.results_table.insertRow(row)
        self.results_table.setItem(row, 0, checkbox_item); self.results_table.setItem(row, 1, item_group); self.results_table.setItem(row, 2, item_file); self.results_table.setItem(row, 3, item_type); self.results_table.setItem(row, 4, item_index); self.results_table.setItem(row, 5, item_preview)

    # --- Slot Methods for Action Buttons ---
    def mark_selected_keep(self):
        """Handles 'Keep' button click."""
        # (No changes needed here)
        print("\n--- '标记为保留'按钮点击 ---")
        selected_blocks = self._get_selected_block_info(); count = 0; affected_rows = []
        if not selected_blocks: print("没有选中任何项。"); return
        print(f"共选中 {len(selected_blocks)} 项。将更新这些项的决策为 '保留':")
        for block_info in selected_blocks:
            if block_info in self.block_decisions: self.block_decisions[block_info] = 'keep'; count += 1
            for row, info in self.row_to_block_info.items():
                if info == block_info: affected_rows.append(row)
        for row in affected_rows: self._update_row_visuals(row, 'keep')
        print(f"已将 {count} 项标记为 '保留' (状态已在内存中更新，行背景已更改)。")
        self.clear_selection()

    def mark_selected_delete(self):
        """Handles 'Delete' button click."""
        # (No changes needed here)
        print("\n--- '标记为删除'按钮点击 ---")
        selected_blocks_to_delete = self._get_selected_block_info(); actually_marked_count = 0; skipped_count = 0; marked_block_infos = []; skipped_block_infos = []
        if not selected_blocks_to_delete: print("没有选中任何项。"); return
        print(f"共选中 {len(selected_blocks_to_delete)} 项尝试标记为 '删除'...")
        hash_to_all_duplicates = {}
        for hash_val, block_list in self.duplicate_blocks.items():
             for block_info in block_list:
                 current_hash = hashlib.md5(block_info[3].encode('utf-8')).hexdigest()
                 if current_hash not in hash_to_all_duplicates and hash_val == current_hash : hash_to_all_duplicates[current_hash] = block_list
        for block_info_to_delete in selected_blocks_to_delete:
            block_hash = hashlib.md5(block_info_to_delete[3].encode('utf-8')).hexdigest()
            if block_hash in hash_to_all_duplicates:
                group_members = hash_to_all_duplicates[block_hash]
                non_delete_count = sum(1 for member in group_members if self.block_decisions.get(member, 'undecided') != 'delete')
                current_decision = self.block_decisions.get(block_info_to_delete, 'undecided')
                is_last_one = (non_delete_count <= 1 and current_decision != 'delete')
                if is_last_one: skipped_count += 1; skipped_block_infos.append(block_info_to_delete); print(f"[跳过] '{block_info_to_delete[0].name}' (#{block_info_to_delete[1]}) 是其重复组中最后一个非删除项。")
                else:
                    if block_info_to_delete in self.block_decisions: self.block_decisions[block_info_to_delete] = 'delete'; marked_block_infos.append(block_info_to_delete); actually_marked_count += 1
                    else: print(f"[警告] 无法找到选中项的决策记录: {block_info_to_delete[0].name} #{block_info_to_delete[1]}")
            else: print(f"[警告] 选中的块未在重复组中找到: {block_info_to_delete[0].name} #{block_info_to_delete[1]}")
        affected_rows_delete = []; affected_rows_skipped = []
        for row, info in self.row_to_block_info.items():
            if info in marked_block_infos: affected_rows_delete.append(row)
            elif info in skipped_block_infos: affected_rows_skipped.append(row)
        for row in affected_rows_delete: self._update_row_visuals(row, 'delete')
        print(f"操作完成：成功将 {actually_marked_count} 项标记为 '删除'。")
        if skipped_count > 0: print(f"有 {skipped_count} 项因需保留至少一个实例而被跳过。")
        self.clear_selection()

    def clear_selection(self):
        """Unchecks all checkboxes in the results table."""
        # (No changes needed here)
        for row in range(self.results_table.rowCount()):
            item = self.results_table.item(row, 0)
            if item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable: item.setCheckState(Qt.CheckState.Unchecked)
        # print("已清除表格中的所有勾选项。") # Make less verbose

    def _update_row_visuals(self, row, decision):
        """Sets the background color of a table row based on the decision."""
        # (No changes needed here)
        color_to_set = self.color_undecided
        if decision == 'keep': color_to_set = self.color_keep
        elif decision == 'delete': color_to_set = self.color_delete
        for col in range(self.results_table.columnCount()):
            item = self.results_table.item(row, col)
            if item:
                if color_to_set: item.setBackground(color_to_set)
                else: item.setBackground(QColor()) # Reset color

    def _get_selected_block_info(self):
        """Iterates through the table and returns a list of block_info tuples for checked rows."""
        # (No changes needed here)
        selected = []
        for row in range(self.results_table.rowCount()):
            checkbox_item = self.results_table.item(row, 0)
            if checkbox_item and checkbox_item.checkState() == Qt.CheckState.Checked:
                block_info = self.row_to_block_info.get(row)
                if block_info: selected.append(block_info)
        return selected

    def _extract_text_from_children(self, children):
        """Helper function to recursively extract text from mistune token children."""
        # (No changes needed here)
        text = ""
        if children is None: return ""
        for child in children:
            child_type = child.get('type')
            if child_type == 'text': text += child.get('raw', '')
            elif child_type == 'codespan': text += child.get('raw', '')
            elif child_type in ['link', 'image', 'emphasis', 'strong', 'strikethrough']:
                text += self._extract_text_from_children(child.get('children', []))
            elif child_type == 'softbreak' or child_type == 'linebreak': text += ' '
            elif child_type == 'inline_html': pass
        return text.strip()

    # --- Slot Method for Save Button ---
    def save_decisions(self):
        """Saves the current block decisions to a JSON file."""
        # (No changes needed here)
        print("\n--- '保存决策'按钮点击 ---")
        if not self.block_decisions: QMessageBox.information(self, "无决策", "没有可保存的决策。"); return
        data_to_save = []
        for block_info, decision in self.block_decisions.items():
            if decision != 'undecided':
                file_path, b_index, b_type, b_text = block_info
                data_to_save.append({"file": str(file_path), "index": b_index, "type": b_type, "decision": decision})
        if not data_to_save: QMessageBox.information(self, "无已标记决策", "没有已标记为 '保留' 或 '删除' 的决策可供保存。"); return
        default_folder = self.folder_path_edit.text() or os.path.expanduser("~"); suggested_filename = os.path.join(default_folder, "kd_decisions.json")
        file_path, _ = QFileDialog.getSaveFileName(self, "保存决策文件", suggested_filename, "JSON Files (*.json);;All Files (*)")
        if not file_path: print("用户取消保存。"); return
        try:
            with open(file_path, 'w', encoding='utf-8') as f: json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            print(f"决策已成功保存到: {file_path}"); QMessageBox.information(self, "保存成功", f"决策已成功保存到:\n{file_path}")
        except Exception as e: print(f"[错误] 保存文件时出错: {e}"); QMessageBox.critical(self, "保存失败", f"保存文件时出错:\n{e}")

    # --- Slot Method for Load Button ---
    def load_decisions(self):
        """Loads block decisions from a JSON file and updates the table visuals."""
        # (No changes needed here)
        print("\n--- '加载决策'按钮点击 ---")
        if not self.row_to_block_info: QMessageBox.warning(self, "无分析结果", "请先运行分析并填充表格后，再加载决策。"); return
        default_folder = self.folder_path_edit.text() or os.path.expanduser("~")
        file_path, _ = QFileDialog.getOpenFileName(self, "加载决策文件", default_folder, "JSON Files (*.json);;All Files (*)")
        if not file_path: print("用户取消加载。"); return
        try:
            print(f"尝试从以下文件加载决策: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f: loaded_data = json.load(f)
            print(f"成功读取 {len(loaded_data)} 条决策记录。")
        except Exception as e: print(f"[错误] 读取或解析JSON时出错: {e}"); QMessageBox.critical(self, "加载失败", f"读取或解析JSON文件时出错:\n{e}"); return
        loaded_decision_map = {}
        for item in loaded_data:
            if isinstance(item, dict) and all(k in item for k in ["file", "index", "type", "decision"]):
                 key = (item['file'], item['index'], item['type']); loaded_decision_map[key] = item['decision']
            else: print(f"[警告] 加载的决策文件中发现格式不正确的条目: {item}")
        print(f"将 {len(loaded_decision_map)} 条有效决策应用到当前分析结果...")
        applied_count = 0
        for block_info in self.block_decisions: self.block_decisions[block_info] = 'undecided' # Reset first
        for block_info in self.block_decisions.keys():
             lookup_key = (str(block_info[0]), block_info[1], block_info[2])
             if lookup_key in loaded_decision_map:
                 loaded_decision = loaded_decision_map[lookup_key]
                 if loaded_decision in ['keep', 'delete']: self.block_decisions[block_info] = loaded_decision; applied_count += 1
                 else: print(f"[警告] 加载的决策值无效 '{loaded_decision}' for {lookup_key}, 保持为 'undecided'.")
        print(f"成功应用 {applied_count} 条加载的决策。")
        print("正在刷新表格视觉效果以匹配加载的决策...")
        self.results_table.setSortingEnabled(False)
        for row, block_info in self.row_to_block_info.items():
             decision = self.block_decisions.get(block_info, 'undecided')
             self._update_row_visuals(row, decision)
             item = self.results_table.item(row, 0);
             if item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable: item.setCheckState(Qt.CheckState.Unchecked)
        self.results_table.setSortingEnabled(True)
        print("表格视觉效果刷新完成。")
        QMessageBox.information(self, "加载完成", f"已成功加载并应用 {applied_count} 条决策。\n表格视觉效果已更新。")

    # --- NEW Slot Method for Apply Button (UPDATED) ---
    def apply_decisions(self):
        """Applies decisions by generating new files containing 'kept' blocks with basic formatting."""
        print("\n--- '应用决策'按钮点击 ---")

        if not self.blocks_data:
             QMessageBox.warning(self, "无分析结果", "请先运行分析，再应用决策。")
             return
        # Allow applying even if no decisions made (will just keep everything)
        # if not self.block_decisions:
        #      QMessageBox.warning(self, "无决策", "没有决策可应用。请先标记重复项。")
        #      return

        output_dir = QFileDialog.getExistingDirectory(self, "选择保存去重后文件的目标文件夹")
        if not output_dir:
            print("用户取消选择目标文件夹。")
            return

        output_dir_path = Path(output_dir)
        print(f"将把去重后的文件保存到: {output_dir_path}")

        # Ensure we know which files were processed in the *current* analysis run
        if not hasattr(self, 'processed_files_in_run') or not self.processed_files_in_run:
             # Fallback: get unique file paths from blocks_data if needed
             processed_files = {info[0] for info in self.blocks_data}
             if not processed_files:
                  QMessageBox.critical(self, "错误", "无法确定处理了哪些文件，请重新运行分析。")
                  return
             self.processed_files_in_run = processed_files # Store it for consistency
             print("[警告] 未找到运行时的文件列表，从当前块数据推断。")


        processed_files_count = 0
        error_files = []

        # Process each original file that was part of the analysis
        for original_file_path in self.processed_files_in_run:
            # Filter blocks belonging to the current file and sort them by index
            blocks_for_this_file = sorted(
                [b for b in self.blocks_data if b[0] == original_file_path],
                key=lambda x: x[1] # Sort by block index
            )

            if not blocks_for_this_file:
                 print(f"文件 {original_file_path.name} 没有提取到内容块，跳过生成。")
                 continue

            new_content_parts = []
            # Iterate through all blocks of the original file in order
            for block_info in blocks_for_this_file:
                file_path, b_index, b_type, b_text = block_info
                # Get the decision, default to 'keep' if somehow missing (or 'undecided')
                decision = self.block_decisions.get(block_info, 'keep') # Default to keep 'undecided'

                if decision != 'delete':
                    # --- Apply Heuristic Formatting ---
                    formatted_text = b_text # Default is the text itself
                    if b_type == 'heading':
                        # Simple assumption: use level 2 heading marker
                        formatted_text = f"## {b_text}"
                    elif b_type == 'block_code':
                        # Assume it's fenced code block
                        # Check if fences are already present in raw text (unlikely for mistune raw)
                        if b_text.startswith("```") and b_text.endswith("```"):
                             formatted_text = b_text # Keep existing fences
                        else:
                             # Add fences (language info might be missing)
                             formatted_text = f"```\n{b_text}\n```"
                    elif b_type == 'list_item':
                        # Simple assumption: use unordered list marker
                        formatted_text = f"- {b_text}"
                    elif b_type == 'block_quote':
                         # Add '>' prefix to each line
                         lines = b_text.splitlines()
                         formatted_text = "\n".join([f"> {line}" for line in lines])
                    # Paragraphs and other types use text as is
                    # ------------------------------------
                    new_content_parts.append(formatted_text)

            # Join the formatted parts with double newlines
            new_content = "\n\n".join(new_content_parts)

            # Construct output filename
            output_filename = f"{original_file_path.stem}_deduped{original_file_path.suffix}"
            output_filepath = output_dir_path / output_filename

            # Write the new file
            try:
                output_filepath.write_text(new_content, encoding='utf-8')
                print(f"已生成新文件: {output_filepath}")
                processed_files_count += 1
            except Exception as e:
                print(f"[错误] 写入文件 '{output_filepath.name}' 时出错: {e}")
                error_files.append(original_file_path.name)

        # Report completion
        if not error_files:
            QMessageBox.information(self, "应用完成", f"已成功为 {processed_files_count} 个文件生成去重后的版本（包含基本格式），保存在:\n{output_dir_path}")
        else:
            QMessageBox.warning(self, "部分完成", f"为 {processed_files_count} 个文件生成了新版本，但以下文件处理失败:\n" + "\n".join(error_files) + f"\n\n请检查目标文件夹权限或错误日志。\n输出目录: {output_dir_path}")

        print("--- 应用决策完成 ---")


    def _extract_text_from_children(self, children):
        """Helper function to recursively extract text from mistune token children."""
        # (No changes needed here)
        text = ""
        if children is None: return ""
        for child in children:
            child_type = child.get('type')
            if child_type == 'text': text += child.get('raw', '')
            elif child_type == 'codespan': text += child.get('raw', '')
            elif child_type in ['link', 'image', 'emphasis', 'strong', 'strikethrough']:
                text += self._extract_text_from_children(child.get('children', []))
            elif child_type == 'softbreak' or child_type == 'linebreak': text += ' '
            elif child_type == 'inline_html': pass
        return text.strip()

# --- 程序入口 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KDToolWindow()
    window.show()
    sys.exit(app.exec())

