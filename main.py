import sys
import os
from pathlib import Path
import hashlib # Still needed for apply_decisions logic if reconstructing hashes
import collections
import mistune
import json
import time

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QTextEdit, QLabel, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QSizePolicy, QDialog, QDialogButtonBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette, QTextCharFormat, QTextCursor, QFont

# Import the new analyzer modules
from md5_analyzer import find_md5_duplicates
from semantic_analyzer import SemanticAnalyzer # Using a class for semantic analysis

# --- Dialog Class (Remains in main.py as it's UI specific) ---
class SemanticDetailDialog(QDialog):
    """显示语义相似块详情（包括上下文）的对话框。"""
    def __init__(self, score, info1, info2, full_text1, full_text2, parent=None):
        super().__init__(parent)
        self.setWindowTitle("语义相似块详情 (含上下文)")
        self.setMinimumSize(800, 600)

        # Store data needed for highlighting
        self.info1 = info1
        self.info2 = info2
        self.full_text1 = full_text1 if full_text1 is not None else "[错误] 无法加载文件内容"
        self.full_text2 = full_text2 if full_text2 is not None else "[错误] 无法加载文件内容"
        file_path1, self.b_index1, self.b_type1, self.b_text1 = info1
        file_path2, self.b_index2, self.b_type2, self.b_text2 = info2

        # --- UI Elements ---
        info_label1 = QLabel(f"<b>文件 1:</b> {file_path1.name} ({self.b_type1} #{self.b_index1})")
        info_label2 = QLabel(f"<b>文件 2:</b> {file_path2.name} ({self.b_type2} #{self.b_index2})")
        score_label = QLabel(f"<b>相似度:</b> {score:.4f}")

        self.text_edit1 = QTextEdit()
        self.text_edit1.setPlainText(self.full_text1)
        self.text_edit1.setReadOnly(True)

        self.text_edit2 = QTextEdit()
        self.text_edit2.setPlainText(self.full_text2)
        self.text_edit2.setReadOnly(True)

        # Use a monospaced font for better alignment and code viewing
        font = QFont("Courier New", 11)
        self.text_edit1.setFont(font)
        self.text_edit2.setFont(font)

        # --- Layout ---
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(score_label)

        content_layout = QHBoxLayout()
        layout1 = QVBoxLayout()
        layout1.addWidget(info_label1)
        layout1.addWidget(self.text_edit1)
        layout2 = QVBoxLayout()
        layout2.addWidget(info_label2)
        layout2.addWidget(self.text_edit2)
        content_layout.addLayout(layout1)
        content_layout.addLayout(layout2)

        main_layout.addLayout(content_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        main_layout.addWidget(button_box)

    def highlight_and_scroll(self):
        """在文本框中高亮显示相应的文本块并滚动到该位置。"""
        highlight_color = QColor("yellow")
        fmt = QTextCharFormat()
        fmt.setBackground(highlight_color)

        # Highlight in Text Edit 1
        cursor1 = self.text_edit1.document().find(self.b_text1)
        if not cursor1.isNull():
            cursor1.mergeCharFormat(fmt)
            self.text_edit1.setTextCursor(cursor1) # Move cursor to the start of the highlight
            self.text_edit1.ensureCursorVisible()
        else:
            print(f"[警告] 无法在文件1的完整内容中定位块文本: {self.b_text1[:50]}...")
            # Fallback: Try finding line by line if exact match fails? (More complex)

        # Highlight in Text Edit 2
        cursor2 = self.text_edit2.document().find(self.b_text2)
        if not cursor2.isNull():
            cursor2.mergeCharFormat(fmt)
            self.text_edit2.setTextCursor(cursor2) # Move cursor to the start of the highlight
            self.text_edit2.ensureCursorVisible()
        else:
            print(f"[警告] 无法在文件2的完整内容中定位块文本: {self.b_text2[:50]}...")


# --- Main Window Class ---
class KDToolWindow(QWidget):
    # Constants remain here as they are specific to the application's use
    SIMILARITY_THRESHOLD = 0.85
    DECISION_KEY_SEPARATOR = "::" # Separator for creating unique block keys

    def __init__(self):
        super().__init__()
        self.setWindowTitle("知识蒸馏 KD Tool v1.25 - Reset on Load") # Update title
        self.setGeometry(200, 100, 800, 800) # Position and size

        # --- 定义颜色常量 ---
        self.color_keep = QColor("lightgreen")
        self.color_delete = QColor(255, 200, 200) # Light red
        self.color_semantic_processed = QColor(220, 220, 220) # Light grey for processed semantic pairs
        self.color_undecided = None # Will use default table colors

        # --- 创建界面控件 ---
        self.folder_label = QLabel("目标文件夹:")
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setReadOnly(True) # Prevent manual editing
        self.browse_button = QPushButton("选择文件夹...")
        self.start_button = QPushButton("开始分析")

        self.md5_status_label = QLabel("精确重复项列表:")
        self.md5_results_table = QTableWidget()
        self.setup_md5_results_table() # Configure the table appearance

        self.keep_button = QPushButton("标记选中项为 '保留'")
        self.delete_button = QPushButton("标记选中项为 '删除'")

        self.semantic_status_label = QLabel("语义相似项列表 (双击查看详情):")
        self.semantic_results_table = QTableWidget()
        self.setup_semantic_table() # Configure the table appearance

        self.semantic_keep1_del2_button = QPushButton("保留块1, 删除块2")
        self.semantic_keep2_del1_button = QPushButton("保留块2, 删除块1")
        self.semantic_keep_both_button = QPushButton("全部保留 (忽略此对)")

        self.save_button = QPushButton("保存决策...")
        self.load_button = QPushButton("加载决策...")
        self.apply_button = QPushButton("应用决策 (生成新文件)...")

        # --- 设置布局 ---
        main_layout = QVBoxLayout(self)

        # Folder selection layout
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(self.folder_path_edit)
        folder_layout.addWidget(self.browse_button)
        main_layout.addLayout(folder_layout)

        main_layout.addWidget(self.start_button)

        # MD5 results section
        main_layout.addWidget(self.md5_status_label)
        main_layout.addWidget(self.md5_results_table)
        self.md5_results_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) # Allow table to grow

        md5_action_layout = QHBoxLayout()
        md5_action_layout.addWidget(self.keep_button)
        md5_action_layout.addWidget(self.delete_button)
        md5_action_layout.addStretch() # Push buttons to the left
        main_layout.addLayout(md5_action_layout)

        # Semantic results section
        main_layout.addWidget(self.semantic_status_label)
        main_layout.addWidget(self.semantic_results_table)
        self.semantic_results_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) # Allow table to grow

        semantic_action_layout = QHBoxLayout()
        semantic_action_layout.addWidget(self.semantic_keep1_del2_button)
        semantic_action_layout.addWidget(self.semantic_keep2_del1_button)
        semantic_action_layout.addWidget(self.semantic_keep_both_button)
        semantic_action_layout.addStretch() # Push buttons to the left
        main_layout.addLayout(semantic_action_layout)

        # Global actions section (Save, Load, Apply)
        global_action_layout = QHBoxLayout()
        global_action_layout.addStretch() # Push buttons to the right
        global_action_layout.addWidget(self.load_button)
        global_action_layout.addWidget(self.save_button)
        global_action_layout.addWidget(self.apply_button)
        main_layout.addLayout(global_action_layout)

        # Set stretch factors for tables to take up available space
        main_layout.setStretchFactor(self.md5_results_table, 1)
        main_layout.setStretchFactor(self.semantic_results_table, 1)

        # --- 连接信号与槽 ---
        self.browse_button.clicked.connect(self.browse_folder)
        self.start_button.clicked.connect(self.start_analysis)
        self.keep_button.clicked.connect(self.mark_selected_keep)
        self.delete_button.clicked.connect(self.mark_selected_delete)
        self.semantic_results_table.itemDoubleClicked.connect(self.show_semantic_detail)
        self.semantic_keep1_del2_button.clicked.connect(self.mark_keep1_delete2)
        self.semantic_keep2_del1_button.clicked.connect(self.mark_keep2_delete1)
        self.semantic_keep_both_button.clicked.connect(self.mark_both_keep)
        self.save_button.clicked.connect(self.save_decisions)
        self.load_button.clicked.connect(self.load_decisions)
        self.apply_button.clicked.connect(self.apply_decisions)

        # --- 类成员变量 (Application State) ---
        self.markdown_files_content = {} # {Path: content_string}
        self.blocks_data = [] # List of tuples: (Path, block_index, block_type, block_text)
        self.duplicate_blocks = {} # {hash: [block_info1, block_info2,...]} from MD5 analysis
        self.row_to_block_info = {} # {md5_table_row_index: block_info} mapping for MD5 table
        self.block_decisions = {} # {decision_key: 'keep'/'delete'/'undecided'} - Master decisions
        self.semantic_duplicates = [] # List of tuples: (block_info1, block_info2, score) from semantic analysis
        self.semantic_row_to_pair_info = {} # {semantic_table_row_index: (block_info1, block_info2, score)} mapping
        # self.semantic_pair_decisions = {} # This might not be needed if decisions map directly to block_decisions
        self.processed_files_in_run = set() # Keep track of files processed in the current analysis run

        # --- Initialize Semantic Analyzer ---
        # Load the model once when the application starts
        self.semantic_analyzer = None # Initialize as None
        try:
            print("Initializing Semantic Analyzer...")
            self.semantic_analyzer = SemanticAnalyzer() # Create instance
            print("Semantic Analyzer initialized successfully.")
        except Exception as e:
            print(f"[严重错误] 初始化 Semantic Analyzer 失败: {e}")
            QMessageBox.critical(self, "模型加载失败", f"无法加载或初始化语义分析模块。\n错误: {e}\n\n语义去重功能将不可用。")
            # The application can continue without semantic analysis


    # --- Helper method to create the decision key ---
    def _create_decision_key(self, file_path, block_index, block_type):
        """Creates a unique string key for a block based on its origin."""
        # Ensure parts are strings for consistent joining
        return f"{str(file_path)}{self.DECISION_KEY_SEPARATOR}{int(block_index)}{self.DECISION_KEY_SEPARATOR}{str(block_type)}"

    # --- Table Setup Methods (UI Specific) ---
    def setup_md5_results_table(self):
        """Configures the QTableWidget for MD5 duplicates."""
        self.md5_results_table.setColumnCount(6)
        self.md5_results_table.setHorizontalHeaderLabels(["选择", "重复组", "文件名", "类型", "块索引", "内容预览"])
        self.md5_results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers) # Read-only
        self.md5_results_table.setSortingEnabled(True) # Allow sorting
        header = self.md5_results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents) # Checkbox column
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch) # Preview column takes extra space
        self.md5_results_table.setAlternatingRowColors(True) # Improve readability

    def setup_semantic_table(self):
        """Configures the QTableWidget for semantic duplicates."""
        self.semantic_results_table.setColumnCount(6)
        self.semantic_results_table.setHorizontalHeaderLabels(["选择", "相似度", "块 1 (文件/类型/索引)", "块 1 预览", "块 2 (文件/类型/索引)", "块 2 预览"])
        self.semantic_results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers) # Read-only
        self.semantic_results_table.setSortingEnabled(True) # Allow sorting
        header = self.semantic_results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents) # Checkbox
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive) # Score
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive) # Info 1
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch) # Preview 1
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive) # Info 2
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch) # Preview 2
        self.semantic_results_table.setAlternatingRowColors(True) # Improve readability

    # --- Core Application Logic ---

    def browse_folder(self):
        """Opens a directory selection dialog and resets state if a folder is chosen."""
        folder_path = QFileDialog.getExistingDirectory(self, "选择包含 Markdown 文件的文件夹")
        if folder_path:
            self.folder_path_edit.setText(folder_path)
            # --- Reset application state ---
            self.markdown_files_content = {}
            self.blocks_data = []
            self.duplicate_blocks = {}
            self.md5_results_table.setRowCount(0) # Clear table
            self.semantic_results_table.setRowCount(0) # Clear table
            self.row_to_block_info = {}
            self.semantic_row_to_pair_info = {}
            self.block_decisions = {}
            self.semantic_duplicates = []
            self.processed_files_in_run = set()
            # self.semantic_pair_decisions = {} # Reset if used
            print(f"已选择文件夹: {folder_path}. 状态已重置。")
        else:
            print("用户取消选择文件夹。")

    def start_analysis(self):
        """Orchestrates the full analysis pipeline."""
        selected_folder_str = self.folder_path_edit.text()

        # --- 0. Pre-checks and State Reset ---
        if not selected_folder_str:
            QMessageBox.warning(self, "缺少文件夹", "请先选择一个包含 Markdown 文件的文件夹。")
            print("[错误] 请先选择一个文件夹！")
            return
        selected_folder = Path(selected_folder_str)
        if not selected_folder.is_dir():
            QMessageBox.critical(self, "路径无效", f"选择的路径不是一个有效的文件夹:\n{selected_folder_str}")
            print(f"[错误] 选择的路径不是一个有效的文件夹: {selected_folder_str}")
            return

        # Reset state before starting analysis
        self.markdown_files_content = {}
        self.blocks_data = []
        self.duplicate_blocks = {}
        self.md5_results_table.setRowCount(0)
        self.semantic_results_table.setRowCount(0)
        self.row_to_block_info = {}
        self.semantic_row_to_pair_info = {}
        self.block_decisions = {}
        self.semantic_duplicates = []
        self.processed_files_in_run = set()
        # self.semantic_pair_decisions = {} # Reset if used
        print(f"开始分析文件夹: {selected_folder}\n---")
        QApplication.setOverrideCursor(Qt.WaitCursor) # Show busy cursor

        # --- 1. Find and Read Files ---
        try:
            print("正在查找和读取 Markdown 文件...")
            markdown_files = list(selected_folder.glob('*.md'))
            print(f"找到 {len(markdown_files)} 个 Markdown 文件。")
            files_read_count = 0
            for md_file in markdown_files:
                try:
                    # Attempt to read with UTF-8 first, add fallbacks if needed
                    content = md_file.read_text(encoding='utf-8')
                    self.markdown_files_content[md_file] = content
                    self.processed_files_in_run.add(md_file) # Track processed files
                    files_read_count += 1
                except Exception as e:
                    print(f"[错误] 读取文件时出错: {md_file.name} - {e}")
            print(f"内容读取完成：成功读取 {files_read_count} / {len(markdown_files)} 个文件。")
            if files_read_count == 0:
                 QMessageBox.warning(self, "无文件读取", "未能成功读取任何 Markdown 文件。请检查文件权限或编码。")
                 QApplication.restoreOverrideCursor()
                 return
        except Exception as e:
            QMessageBox.critical(self, "查找文件失败", f"查找 Markdown 文件时出错:\n{e}")
            print(f"查找文件时出错: {e}")
            QApplication.restoreOverrideCursor()
            return

        # --- 2. Parse Markdown and Extract Blocks (Exclude Headings) ---
        print("---")
        print("开始解析 Markdown 并提取内容块 (排除标题)...")
        self.blocks_data = [] # Ensure it's clear before parsing
        # Mistune setup - consider making the renderer configurable if needed
        markdown_parser = mistune.create_markdown(renderer=None) # Use the block-level parser
        try:
            for md_file_path, content in self.markdown_files_content.items():
                block_tokens = markdown_parser(content)
                for index, token in enumerate(block_tokens):
                    block_type = token['type']
                    block_text = ""
                    blocks_to_add = [] # Handle lists creating multiple blocks

                    if block_type == 'heading':
                        continue # Skip headings as per requirement

                    elif block_type == 'paragraph':
                        # Extract text, handling potential inline elements
                        block_text = self._extract_text_from_children(token.get('children', []))
                    elif block_type == 'block_code':
                        # Use 'raw' if available (includes language), otherwise 'text'
                        block_text = token.get('raw') or token.get('text', '')
                    elif block_type == 'block_quote':
                        block_text = self._extract_text_from_children(token.get('children', []))
                    elif block_type == 'list':
                        list_items = token.get('children', [])
                        for item_index, item_token in enumerate(list_items):
                             # Check if it's a list item token
                            if item_token.get('type') == 'list_item':
                                # Treat each list item as a separate block
                                item_text = self._extract_text_from_children(item_token.get('children', []))
                                cleaned_item_text = item_text.strip()
                                if cleaned_item_text: # Only add non-empty items
                                    # Use a combined index or a sub-index if needed, here using main block index
                                    blocks_to_add.append((md_file_path, index, 'list_item', cleaned_item_text))
                        # Skip adding the 'list' block itself if we added items
                        if blocks_to_add:
                            block_type = None # Prevent adding the parent 'list' block
                    # Add other block types as needed (e.g., thematic_break, html_block)

                    # Clean and add the primary block if it wasn't a list handled above
                    if block_type not in ['list', None]: # Check type again
                        cleaned_text = block_text.strip() if block_type != 'block_code' else block_text
                        if cleaned_text: # Only add non-empty blocks
                             blocks_to_add.append((md_file_path, index, block_type, cleaned_text))

                    # Extend the main blocks_data list
                    if blocks_to_add:
                        self.blocks_data.extend(blocks_to_add)

            total_blocks = len(self.blocks_data)
            print(f"Markdown 解析完成：共获得 {total_blocks} 个有效内容块 (已排除标题)。")
            if total_blocks == 0:
                QMessageBox.information(self, "无内容块", "在读取的文件中未能解析出有效的内容块（已排除标题）。")
                QApplication.restoreOverrideCursor()
                return

        except Exception as e:
            QMessageBox.critical(self, "解析失败", f"解析 Markdown 时出错:\n{e}")
            print(f"解析 Markdown 时出错: {e}")
            QApplication.restoreOverrideCursor()
            return

        # --- 3. Initialize Decisions for ALL Parsed Blocks ---
        print("---")
        print("初始化所有有效块的决策状态为 'undecided'...")
        self.block_decisions = {} # Ensure it's clear
        try:
            for block_info in self.blocks_data: # Iterate through ALL extracted blocks
                key = self._create_decision_key(block_info[0], block_info[1], block_info[2])
                self.block_decisions[key] = 'undecided' # Default state
            print(f"已为 {len(self.block_decisions)} 个有效块初始化决策。")
        except Exception as e:
             QMessageBox.critical(self, "初始化失败", f"初始化决策时出错:\n{e}")
             print(f"[错误] 初始化决策时出错: {e}")
             QApplication.restoreOverrideCursor()
             return

        # --- 4. Perform MD5 Deduplication ---
        print("---")
        print("开始计算 MD5 并查找精确重复内容块...")
        try:
            # Call the function from the md5_analyzer module
            self.duplicate_blocks = find_md5_duplicates(self.blocks_data)
            print(f"找到 {len(self.duplicate_blocks)} 组完全重复的内容块 (基于 MD5)。")

            # --- 5. Populate MD5 Table ---
            self.md5_results_table.setSortingEnabled(False) # Disable sorting during population
            self.md5_results_table.setRowCount(0) # Clear previous results
            current_row = 0
            group_id_counter = 1
            self.row_to_block_info = {} # Reset mapping

            if not self.duplicate_blocks:
                print("未找到完全重复的内容块。")
            else:
                print("开始填充 MD5 重复项表格...")
                for hash_val, b_list in self.duplicate_blocks.items():
                    group_id_str = f"组 {group_id_counter}"
                    for block_info in b_list:
                        # Populate the table row using the helper method
                        self._populate_md5_table_row(current_row, group_id_str, block_info)
                        # Store mapping from row index to block info for later use
                        self.row_to_block_info[current_row] = block_info
                        current_row += 1
                    group_id_counter += 1
                print(f"MD5 表格填充完成，共 {current_row} 行。")

            self.md5_results_table.setSortingEnabled(True) # Re-enable sorting
            print("--- MD5 分析完成 ---")

        except Exception as e:
            QMessageBox.critical(self, "MD5 分析失败", f"计算 MD5、查找重复或填充表格时出错:\n{e}")
            print(f"计算 MD5、查找或填充表格时出错: {e}")
            # Decide if you want to continue to semantic analysis or stop
            # For now, let's try to continue if possible, but semantic might also fail

        # --- 6. Perform Semantic Deduplication ---
        print("---")
        print("开始进行语义相似度分析...")
        self.semantic_duplicates = [] # Clear previous results
        self.semantic_results_table.setRowCount(0) # Clear table
        self.semantic_row_to_pair_info = {} # Reset mapping

        # Check if semantic analyzer was loaded successfully and if there's data
        if self.semantic_analyzer is None or not self.semantic_analyzer.is_model_loaded():
            print("[信息] 语义模型未加载或初始化失败，跳过语义分析。")
            # Optionally inform the user via status bar or a non-blocking message
        elif len(self.blocks_data) < 2:
            print("内容块数量不足 (<2)，无法进行语义比较。")
        else:
            try:
                start_time = time.time()
                print(f"正在使用阈值 {self.SIMILARITY_THRESHOLD} 查找语义相似对...")

                # Call the method from the semantic_analyzer instance
                # Pass only the necessary data (block info list)
                self.semantic_duplicates = self.semantic_analyzer.find_similar_blocks(
                    self.blocks_data, self.SIMILARITY_THRESHOLD
                )

                find_time = time.time()
                print(f"语义相似对查找完成，找到 {len(self.semantic_duplicates)} 对，耗时: {find_time - start_time:.2f} 秒")

                # --- 7. Populate Semantic Table ---
                self.semantic_results_table.setSortingEnabled(False) # Disable sorting
                # Row count is already 0 from reset
                current_semantic_row = 0

                if not self.semantic_duplicates:
                    print("未找到满足条件的语义相似内容块对。")
                else:
                    print(f"开始填充语义相似项表格 ({len(self.semantic_duplicates)} 对)...")
                    for info1, info2, score in self.semantic_duplicates:
                        # Populate the table row
                        self._populate_semantic_table_row(current_semantic_row, score, info1, info2)
                        # Store mapping from row index to pair info
                        self.semantic_row_to_pair_info[current_semantic_row] = (info1, info2, score)
                        current_semantic_row += 1
                    print(f"语义表格填充完成，共 {current_semantic_row} 行。")

                self.semantic_results_table.setSortingEnabled(True) # Re-enable sorting

            except Exception as e:
                 QMessageBox.critical(self, "语义分析失败", f"语义分析或填充表格时出错:\n{e}")
                 print(f"[错误] 语义分析或填充表格时出错: {e}")
                 # Analysis continues, but semantic results might be incomplete/missing

        QApplication.restoreOverrideCursor() # Restore normal cursor
        print("--- 全部分析流程结束 ---")
        QMessageBox.information(self, "分析完成", "MD5 和语义分析已完成。请查看表格结果并进行决策。")


    # --- Table Population Helpers (UI Specific) ---

    def _populate_md5_table_row(self, row, group_id_str, block_info):
        """Helper method to populate a single row in the MD5 results table."""
        file_path, b_index, b_type, b_text = block_info

        # Create items for the row
        checkbox_item = QTableWidgetItem()
        checkbox_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        checkbox_item.setCheckState(Qt.CheckState.Unchecked) # Default to unchecked

        item_group = QTableWidgetItem(group_id_str)
        item_file = QTableWidgetItem(file_path.name) # Show only filename
        item_type = QTableWidgetItem(b_type)
        item_index = QTableWidgetItem(str(b_index)) # Index needs to be string

        # Create a preview, limit length and remove newlines for table view
        preview_text = (b_text[:100] + '...') if len(b_text) > 100 else b_text
        preview_text = preview_text.replace('\n', ' ') # Replace newlines with spaces
        item_preview = QTableWidgetItem(preview_text)

        # Insert row and set items
        self.md5_results_table.insertRow(row)
        self.md5_results_table.setItem(row, 0, checkbox_item)
        self.md5_results_table.setItem(row, 1, item_group)
        self.md5_results_table.setItem(row, 2, item_file)
        self.md5_results_table.setItem(row, 3, item_type)
        self.md5_results_table.setItem(row, 4, item_index)
        self.md5_results_table.setItem(row, 5, item_preview)
        # Apply initial background based on loaded/existing decisions if any
        key = self._create_decision_key(file_path, b_index, b_type)
        decision = self.block_decisions.get(key, 'undecided')
        self._update_md5_row_visuals(row, decision)


    def _populate_semantic_table_row(self, row, score, block_info1, block_info2):
        """Helper method to populate a single row in the semantic results table."""
        file_path1, b_index1, b_type1, b_text1 = block_info1
        file_path2, b_index2, b_type2, b_text2 = block_info2

        # Create items for the row
        checkbox_item = QTableWidgetItem()
        checkbox_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled) # Make checkable
        checkbox_item.setCheckState(Qt.CheckState.Unchecked)

        score_str = f"{score:.4f}" # Format score
        item_score = QTableWidgetItem(score_str)
        item_score.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter) # Align score right

        info1_str = f"{file_path1.name} ({b_type1} #{b_index1})"
        item_info1 = QTableWidgetItem(info1_str)
        preview1 = (b_text1[:70] + '...') if len(b_text1) > 70 else b_text1
        preview1 = preview1.replace('\n', ' ')
        item_preview1 = QTableWidgetItem(preview1)

        info2_str = f"{file_path2.name} ({b_type2} #{b_index2})"
        item_info2 = QTableWidgetItem(info2_str)
        preview2 = (b_text2[:70] + '...') if len(b_text2) > 70 else b_text2
        preview2 = preview2.replace('\n', ' ')
        item_preview2 = QTableWidgetItem(preview2)

        # Insert row and set items
        self.semantic_results_table.insertRow(row)
        self.semantic_results_table.setItem(row, 0, checkbox_item)
        self.semantic_results_table.setItem(row, 1, item_score)
        self.semantic_results_table.setItem(row, 2, item_info1)
        self.semantic_results_table.setItem(row, 3, item_preview1)
        self.semantic_results_table.setItem(row, 4, item_info2)
        self.semantic_results_table.setItem(row, 5, item_preview2)
        # Apply initial background (likely undecided/default at this point unless loading decisions)
        # For semantic pairs, the initial state is usually just the alternating row color.
        # We update visuals when actions are taken (keep1/del2 etc.)
        self._update_semantic_row_visuals(row, 'undecided') # Use 'undecided' or similar for default


    # --- Slot Methods for MD5 Action Buttons ---

    def mark_selected_keep(self):
        """Handles 'Keep' button click FOR MD5 DUPLICATES."""
        print("\n--- '标记为保留'按钮点击 (精确重复) ---")
        selected_full_block_infos = self._get_selected_md5_block_info() # Get info for checked rows

        if not selected_full_block_infos:
            print("没有选中任何精确重复项。")
            QMessageBox.information(self, "未选择", "请在上方表格中勾选要保留的项。")
            return

        print(f"共选中 {len(selected_full_block_infos)} 项。将更新这些项的决策为 '保留':")
        count = 0
        affected_rows = [] # Keep track of rows to update visuals

        for full_block_info in selected_full_block_infos:
            # Create the unique key for this block
            decision_key = self._create_decision_key(full_block_info[0], full_block_info[1], full_block_info[2])

            # Update the master decision dictionary
            if decision_key in self.block_decisions:
                if self.block_decisions[decision_key] != 'keep':
                    self.block_decisions[decision_key] = 'keep'
                    count += 1
                    # Find all rows in the MD5 table corresponding to this key
                    for row, info in self.row_to_block_info.items():
                        current_key = self._create_decision_key(info[0], info[1], info[2])
                        if current_key == decision_key:
                            affected_rows.append(row)
            else:
                # This should ideally not happen if initialization is correct
                print(f"[警告] 尝试更新未初始化的决策键: {decision_key}")

        # Update visuals for affected rows (remove duplicates before iterating)
        unique_affected_rows = sorted(list(set(affected_rows)))
        for row in unique_affected_rows:
            self._update_md5_row_visuals(row, 'keep') # Update background color

        print(f"已将 {count} 项新标记为 '保留' (状态已在内存中更新，行背景已更改)。")

        # Update semantic table visuals if any of these blocks are part of a pair
        self._sync_semantic_visuals_from_block_decisions()

        self.clear_md5_selection() # Uncheck boxes after processing
        QApplication.processEvents() # Ensure UI updates immediately

    def mark_selected_delete(self):
        """Handles 'Delete' button click FOR MD5 DUPLICATES, ensuring at least one in a group is kept."""
        print("\n--- '标记为删除'按钮点击 (精确重复) ---")
        selected_full_block_infos = self._get_selected_md5_block_info()

        if not selected_full_block_infos:
            print("没有选中任何项。")
            QMessageBox.information(self, "未选择", "请在上方表格中勾选要删除的项。")
            return

        print(f"共选中 {len(selected_full_block_infos)} 项尝试标记为 '删除'...")
        actually_marked_count = 0
        skipped_count = 0
        marked_keys = [] # Keys successfully marked for deletion
        skipped_keys = [] # Keys skipped because they were the last non-delete item

        # Need to check within each MD5 duplicate group
        hash_to_all_keys_in_group = collections.defaultdict(list)
        if not self.duplicate_blocks: # Should be populated if we have rows
             print("[警告] duplicate_blocks 字典为空，无法执行删除检查。")
             # Fallback: maybe allow deletion but warn? Or prevent? For safety, prevent.
             QMessageBox.warning(self, "内部状态错误", "无法验证删除操作，因为缺少重复组信息。请重新运行分析。")
             return

        # Rebuild hash_to_all_keys_in_group mapping based on current duplicate_blocks
        for hash_val, block_list in self.duplicate_blocks.items():
             for block_info in block_list:
                 decision_key = self._create_decision_key(block_info[0], block_info[1], block_info[2])
                 hash_to_all_keys_in_group[hash_val].append(decision_key)

        # Iterate through selected blocks to determine deletability
        for full_block_info_to_delete in selected_full_block_infos:
            decision_key_to_delete = self._create_decision_key(full_block_info_to_delete[0], full_block_info_to_delete[1], full_block_info_to_delete[2])
            block_text_to_delete = full_block_info_to_delete[3]
            block_hash = hashlib.md5(block_text_to_delete.encode('utf-8')).hexdigest()

            if block_hash in hash_to_all_keys_in_group:
                group_member_keys = hash_to_all_keys_in_group[block_hash]
                if not group_member_keys:
                    print(f"[内部错误] 找不到哈希 {block_hash} 对应的键列表。跳过 {decision_key_to_delete}")
                    continue

                # Count how many in the group are NOT currently marked for deletion
                non_delete_count = sum(1 for member_key in group_member_keys if self.block_decisions.get(member_key, 'undecided') != 'delete')

                # Check if the current item is one of those non-delete items
                current_decision = self.block_decisions.get(decision_key_to_delete, 'undecided')

                # Is this the last one standing (or one of the last ones)?
                is_last_one_or_already_deleted = (non_delete_count <= 1 and current_decision != 'delete')

                if is_last_one_or_already_deleted:
                    skipped_count += 1
                    skipped_keys.append(decision_key_to_delete)
                    print(f"[跳过] '{full_block_info_to_delete[0].name}' ({full_block_info_to_delete[2]} #{full_block_info_to_delete[1]}) 是其重复组中最后一个非删除项或已被标记删除。")
                else:
                    # Safe to mark for deletion
                    if decision_key_to_delete in self.block_decisions:
                        if self.block_decisions[decision_key_to_delete] != 'delete':
                             self.block_decisions[decision_key_to_delete] = 'delete'
                             marked_keys.append(decision_key_to_delete)
                             actually_marked_count += 1
                        # else: already marked, do nothing
                    else:
                        print(f"[警告] 尝试更新未初始化的决策键: {decision_key_to_delete}") # Should not happen

            else:
                # This block wasn't found in the duplicate groups - might be a single instance mistakenly shown?
                # Or maybe it was selected but isn't actually a duplicate shown in the table?
                # For safety, treat it as if it cannot be deleted unless we are sure.
                # Alternative: Allow deletion if it's not part of a duplicate group?
                # Current logic: Only delete if part of a group and not the last one.
                print(f"[警告] 选中的块未在重复组中找到 (哈希不匹配?): {decision_key_to_delete}. 无法标记为删除。")
                skipped_count += 1 # Treat as skipped
                skipped_keys.append(decision_key_to_delete)


        # Update visuals based on marked and skipped keys
        affected_rows_delete = []
        affected_rows_skipped = [] # Rows that were selected but skipped

        for row, info in self.row_to_block_info.items():
            key = self._create_decision_key(info[0], info[1], info[2])
            if key in marked_keys:
                affected_rows_delete.append(row)
            # Check if this row corresponds to a key that was selected but skipped
            elif key in skipped_keys and self.md5_results_table.item(row, 0).checkState() == Qt.CheckState.Checked:
                 affected_rows_skipped.append(row)


        for row in affected_rows_delete:
            self._update_md5_row_visuals(row, 'delete')
        # For skipped rows, we just leave their visual state as is, but uncheck them.
        # for row in affected_rows_skipped:
        #     pass # Visuals don't change, maybe uncheck? Done by clear_md5_selection

        print(f"操作完成：成功将 {actually_marked_count} 项新标记为 '删除'。")
        if skipped_count > 0:
            print(f"有 {skipped_count} 项因需保留至少一个实例或未在重复组中找到而被跳过。")
            QMessageBox.warning(self, "部分跳过", f"成功标记 {actually_marked_count} 项为删除。\n有 {skipped_count} 项被跳过，因为它们是其重复组中最后一个未标记为删除的项，或者存在内部错误。")

        # Update semantic table visuals if any of these blocks are part of a pair
        self._sync_semantic_visuals_from_block_decisions()

        self.clear_md5_selection() # Uncheck boxes after processing
        QApplication.processEvents() # Ensure UI updates

    # --- MD5 Table Interaction Helpers ---

    def clear_md5_selection(self):
        """Unchecks all checkboxes in the MD5 results table."""
        for row in range(self.md5_results_table.rowCount()):
            item = self.md5_results_table.item(row, 0) # Checkbox is in column 0
            if item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(Qt.CheckState.Unchecked)

    def _update_md5_row_visuals(self, row, decision):
        """Sets the background color of a row in the MD5 results table based on the decision."""
        if row < 0 or row >= self.md5_results_table.rowCount():
            print(f"[警告] _update_md5_row_visuals: 无效行索引 {row}")
            return

        # Determine the base color (considering alternating rows)
        default_bg_color = self.md5_results_table.palette().color(QPalette.ColorRole.Base)
        # Use a slightly darker color for alternating rows for better contrast
        alt_bg_color = self.md5_results_table.palette().color(QPalette.ColorRole.AlternateBase)
        base_color = alt_bg_color if row % 2 == 1 else default_bg_color

        color_to_set = base_color # Default to the base alternating color

        if decision == 'keep':
            color_to_set = self.color_keep
        elif decision == 'delete':
            color_to_set = self.color_delete
        # 'undecided' uses the base_color

        # Apply the color to all cells in the row
        for col in range(self.md5_results_table.columnCount()):
            item = self.md5_results_table.item(row, col)
            if item:
                item.setBackground(color_to_set)
            else:
                # If an item doesn't exist (shouldn't happen with current population), create a placeholder?
                # Or just skip. Skipping is safer.
                pass

    def _get_selected_md5_block_info(self):
        """Gets block_info tuples for rows currently checked in the MD5 results table."""
        selected = []
        for row in range(self.md5_results_table.rowCount()):
            checkbox_item = self.md5_results_table.item(row, 0) # Checkbox is in column 0
            # Check if item exists, is checkable, and is checked
            if checkbox_item and \
               checkbox_item.flags() & Qt.ItemFlag.ItemIsUserCheckable and \
               checkbox_item.checkState() == Qt.CheckState.Checked:
                # Retrieve the block_info associated with this row
                block_info = self.row_to_block_info.get(row)
                if block_info:
                    selected.append(block_info)
                else:
                    # This indicates a mismatch between the table and the mapping dict
                    print(f"[警告] 在 MD5 表格的第 {row} 行找到了选中的复选框，但在 row_to_block_info 映射中找不到对应的块信息。")
        return selected

    # --- Slot Methods for Semantic Action Buttons ---

    def mark_keep1_delete2(self):
        """Marks block 1 as keep, block 2 as delete for the SINGLE selected semantic pair."""
        print("\n--- '保留块1, 删除块2' 按钮点击 ---")
        selected_pairs_with_rows = self._get_selected_semantic_pairs_with_rows() # Get pairs and their rows

        if len(selected_pairs_with_rows) != 1:
            QMessageBox.warning(self, "选择错误", "请在下方表格中刚好选择一项（一对相似块）以执行此操作。")
            print(f"需要刚好选择 1 对，当前选择了 {len(selected_pairs_with_rows)} 对。")
            return

        # Unpack the selected pair and its row index
        info1, info2, score, row = selected_pairs_with_rows[0]
        key1 = self._create_decision_key(info1[0], info1[1], info1[2])
        key2 = self._create_decision_key(info2[0], info2[1], info2[2])

        print(f"将标记 '{info1[0].name}' ({info1[2]} #{info1[1]}) 为 'keep', '{info2[0].name}' ({info2[2]} #{info2[1]}) 为 'delete'")

        # --- Update Master Decisions ---
        # Check if keys exist before updating (they should)
        if key1 in self.block_decisions:
            self.block_decisions[key1] = 'keep'
        else: print(f"[警告] 尝试更新未初始化的决策键: {key1}")
        if key2 in self.block_decisions:
             # Check if deleting key2 would violate the "keep at least one" rule for its MD5 group (if any)
            can_delete_key2 = self._can_delete_block(key2)
            if can_delete_key2:
                self.block_decisions[key2] = 'delete'
            else:
                print(f"[信息] 无法将块2 ({key2}) 标记为删除，因为它可能是其精确重复组中的最后一个。块2将保持 '{self.block_decisions.get(key2)}' 状态。")
                QMessageBox.warning(self, "操作受限", f"无法将块 '{info2[0].name}' ({info2[2]} #{info2[1]}) 标记为删除，因为它可能是其精确重复组中的最后一个。\n\n块1已标记为 'keep'。请手动检查块2的决策。")
                # Even if key2 cannot be deleted, key1 is marked keep, so the pair is processed.
        else: print(f"[警告] 尝试更新未初始化的决策键: {key2}")

        # --- Update Semantic table row visual ---
        # Mark the row in the semantic table as processed (e.g., grey background)
        self._update_semantic_row_visuals(row, 'processed')

        # --- Update corresponding MD5 visuals ---
        # Find rows in MD5 table for key1 and key2 and update their colors
        self._update_md5_visuals_for_key(key1, self.block_decisions.get(key1)) # Use the actual decision made
        self._update_md5_visuals_for_key(key2, self.block_decisions.get(key2)) # Use the actual decision made

        self.clear_semantic_selection() # Uncheck the box
        QApplication.processEvents()

    def mark_keep2_delete1(self):
        """Marks block 2 as keep, block 1 as delete for the SINGLE selected semantic pair."""
        print("\n--- '保留块2, 删除块1' 按钮点击 ---")
        selected_pairs_with_rows = self._get_selected_semantic_pairs_with_rows()

        if len(selected_pairs_with_rows) != 1:
            QMessageBox.warning(self, "选择错误", "请在下方表格中刚好选择一项（一对相似块）以执行此操作。")
            print(f"需要刚好选择 1 对，当前选择了 {len(selected_pairs_with_rows)} 对。")
            return

        info1, info2, score, row = selected_pairs_with_rows[0]
        key1 = self._create_decision_key(info1[0], info1[1], info1[2])
        key2 = self._create_decision_key(info2[0], info2[1], info2[2])

        print(f"将标记 '{info2[0].name}' ({info2[2]} #{info2[1]}) 为 'keep', '{info1[0].name}' ({info1[2]} #{info1[1]}) 为 'delete'")

        # --- Update Master Decisions ---
        if key2 in self.block_decisions:
            self.block_decisions[key2] = 'keep'
        else: print(f"[警告] 尝试更新未初始化的决策键: {key2}")

        if key1 in self.block_decisions:
            can_delete_key1 = self._can_delete_block(key1)
            if can_delete_key1:
                self.block_decisions[key1] = 'delete'
            else:
                 print(f"[信息] 无法将块1 ({key1}) 标记为删除，因为它可能是其精确重复组中的最后一个。块1将保持 '{self.block_decisions.get(key1)}' 状态。")
                 QMessageBox.warning(self, "操作受限", f"无法将块 '{info1[0].name}' ({info1[2]} #{info1[1]}) 标记为删除，因为它可能是其精确重复组中的最后一个。\n\n块2已标记为 'keep'。请手动检查块1的决策。")
        else: print(f"[警告] 尝试更新未初始化的决策键: {key1}")


        # --- Update Visuals ---
        self._update_semantic_row_visuals(row, 'processed')
        self._update_md5_visuals_for_key(key1, self.block_decisions.get(key1))
        self._update_md5_visuals_for_key(key2, self.block_decisions.get(key2))

        self.clear_semantic_selection()
        QApplication.processEvents()

    def mark_both_keep(self):
        """Marks both blocks in the SINGLE selected semantic pair as keep (ignore similarity)."""
        print("\n--- '全部保留 (忽略此对)' 按钮点击 ---")
        selected_pairs_with_rows = self._get_selected_semantic_pairs_with_rows()

        # Allow processing multiple selections for "keep both"
        if not selected_pairs_with_rows:
             QMessageBox.information(self, "未选择", "请在下方表格中勾选要全部保留的相似对。")
             print("没有选中任何语义相似对。")
             return

        print(f"共选中 {len(selected_pairs_with_rows)} 对。将更新这些对中所有块的决策为 '保留':")
        count = 0
        affected_keys = [] # Track keys whose decision changed to 'keep'
        affected_semantic_rows = [] # Track semantic rows to update visuals

        for info1, info2, score, row in selected_pairs_with_rows:
            key1 = self._create_decision_key(info1[0], info1[1], info1[2])
            key2 = self._create_decision_key(info2[0], info2[1], info2[2])
            affected_semantic_rows.append(row) # Mark this semantic row for visual update

            # Update decision for key1
            if key1 in self.block_decisions:
                if self.block_decisions[key1] != 'keep':
                    self.block_decisions[key1] = 'keep'
                    affected_keys.append(key1)
                    count += 1
            else: print(f"[警告] 尝试更新未初始化的决策键: {key1}")

            # Update decision for key2
            if key2 in self.block_decisions:
                 if self.block_decisions[key2] != 'keep':
                    self.block_decisions[key2] = 'keep'
                    affected_keys.append(key2)
                    count += 1
            else: print(f"[警告] 尝试更新未初始化的决策键: {key2}")


        # --- Update MD5 visuals (if applicable) ---
        # Find all MD5 rows corresponding to the affected keys
        affected_rows_md5 = []
        for row_md5, info_md5 in self.row_to_block_info.items():
            key = self._create_decision_key(info_md5[0], info_md5[1], info_md5[2])
            if key in affected_keys: # Check if this key was one that changed to 'keep'
                affected_rows_md5.append(row_md5)

        # Update visuals for unique MD5 rows
        unique_affected_rows_md5 = sorted(list(set(affected_rows_md5)))
        if unique_affected_rows_md5:
             print(f"正在更新 {len(unique_affected_rows_md5)} 个 MD5 表格行的视觉效果...")
        for row_md5 in unique_affected_rows_md5:
            self._update_md5_row_visuals(row_md5, 'keep') # Set to green

        # --- Update Semantic visuals ---
        # Mark the selected semantic rows as ignored/processed
        for row in affected_semantic_rows:
            self._update_semantic_row_visuals(row, 'ignored') # Use 'ignored' or 'processed'

        print(f"已将 {count} 个块新标记为 '保留' (状态已在内存中更新，对应表格行背景已更改)。")
        self.clear_semantic_selection() # Uncheck boxes
        QApplication.processEvents()

    # --- Helper to check if a block can be deleted (MD5 group check) ---
    def _can_delete_block(self, decision_key_to_delete):
        """Checks if deleting this block would leave at least one non-delete block in its MD5 group."""
        # Find the block info corresponding to the key
        block_info_to_delete = None
        for info in self.blocks_data:
            key = self._create_decision_key(info[0], info[1], info[2])
            if key == decision_key_to_delete:
                block_info_to_delete = info
                break

        if not block_info_to_delete:
            print(f"[警告] _can_delete_block: 无法找到键 {decision_key_to_delete} 对应的块信息。假定可以删除。")
            return True # Or False for safety? Let's assume true if info not found.

        # Calculate its hash
        block_text_to_delete = block_info_to_delete[3]
        block_hash = hashlib.md5(block_text_to_delete.encode('utf-8')).hexdigest()

        # Find its MD5 group members
        group_members = self.duplicate_blocks.get(block_hash)

        # If it's not in a duplicate group, it can always be deleted
        if not group_members or len(group_members) < 2:
            return True

        # Check the decisions of all members in the group
        group_member_keys = [self._create_decision_key(info[0], info[1], info[2]) for info in group_members]
        non_delete_count = sum(1 for member_key in group_member_keys if self.block_decisions.get(member_key, 'undecided') != 'delete')

        # Can we delete this specific one? Only if doing so leaves at least one non-delete block remaining.
        current_decision = self.block_decisions.get(decision_key_to_delete, 'undecided')
        if current_decision != 'delete':
            # If we delete this one, will the non_delete_count drop to 0?
            return non_delete_count > 1
        else:
            # It's already marked for delete, so deleting it doesn't change the count
            # But the action might be redundant. The check is more about *preventing* deletion.
            # Let's rephrase: can this block transition *to* delete? Yes, if non_delete_count > 1.
             return non_delete_count > 1


    # --- Semantic Table Interaction Helpers ---

    def clear_semantic_selection(self):
        """Unchecks all checkboxes in the semantic results table."""
        for row in range(self.semantic_results_table.rowCount()):
            item = self.semantic_results_table.item(row, 0) # Checkbox is in column 0
            if item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(Qt.CheckState.Unchecked)

    def _get_selected_semantic_pairs_with_rows(self):
        """Gets pair_info tuples AND their row index for checked rows in the semantic table."""
        selected_pairs = []
        for row in range(self.semantic_results_table.rowCount()):
            checkbox_item = self.semantic_results_table.item(row, 0) # Checkbox is in column 0
            if checkbox_item and \
               checkbox_item.flags() & Qt.ItemFlag.ItemIsUserCheckable and \
               checkbox_item.checkState() == Qt.CheckState.Checked:
                # Retrieve the pair_info associated with this row
                pair_info = self.semantic_row_to_pair_info.get(row)
                if pair_info:
                    # Append the row index to the tuple: (info1, info2, score, row)
                    selected_pairs.append(pair_info + (row,))
                else:
                    print(f"[警告] 在语义表格的第 {row} 行找到了选中的复选框，但在 semantic_row_to_pair_info 映射中找不到对应的对信息。")
        return selected_pairs


    def _update_semantic_row_visuals(self, row, decision_type):
        """Sets the background color of a row in the semantic results table."""
        if row < 0 or row >= self.semantic_results_table.rowCount():
            print(f"[警告] _update_semantic_row_visuals: 无效行索引 {row}")
            return

        # Determine the base color (considering alternating rows)
        default_bg_color = self.semantic_results_table.palette().color(QPalette.ColorRole.Base)
        alt_bg_color = self.semantic_results_table.palette().color(QPalette.ColorRole.AlternateBase)
        base_color = alt_bg_color if row % 2 == 1 else default_bg_color

        color_to_set = base_color # Default to the base alternating color

        # Set color based on the action taken on the pair
        if decision_type in ['processed', 'ignored']:
            # Use a distinct color (e.g., light grey) to show the pair has been handled
            color_to_set = self.color_semantic_processed
        # 'undecided' or other states use the base_color

        # Apply the color to all cells in the row
        for col in range(self.semantic_results_table.columnCount()):
            item = self.semantic_results_table.item(row, col)
            if item:
                item.setBackground(color_to_set)
                # --- Optionally disable checkbox after processing ---
                # Keep it checkable for now, user might want to re-process?
                # If you want to disable after processing:
                # if col == 0 and decision_type in ['processed', 'ignored']:
                #     item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable) # Remove checkable flag
                # Ensure it's unchecked after processing action
                if col == 0 and decision_type in ['processed', 'ignored']:
                     item.setCheckState(Qt.CheckState.Unchecked)


    def _sync_semantic_visuals_from_block_decisions(self):
        """Updates semantic table row visuals based on the current state in block_decisions."""
        print("同步语义表格视觉效果...")
        processed_count = 0
        ignored_count = 0 # Could represent 'keep both'
        undecided_count = 0

        # Iterate through the semantic pairs currently displayed
        # Check if semantic_row_to_pair_info is populated
        if not self.semantic_row_to_pair_info:
            print("  (无语义对信息可同步)")
            return

        for row, pair_data in self.semantic_row_to_pair_info.items():
            # Ensure pair_data is unpacked correctly
            if len(pair_data) == 3:
                info1, info2, score = pair_data
            else:
                print(f"  [警告] 语义行 {row} 的数据格式不正确: {pair_data}")
                continue # Skip this row

            key1 = self._create_decision_key(info1[0], info1[1], info1[2])
            key2 = self._create_decision_key(info2[0], info2[1], info2[2])
            decision1 = self.block_decisions.get(key1, 'undecided') # Gets current decision for block 1
            decision2 = self.block_decisions.get(key2, 'undecided') # Gets current decision for block 2

            # Determine the state of the *pair* based on individual block decisions
            pair_state = 'undecided' # Default
            if (decision1 == 'keep' and decision2 == 'delete') or \
               (decision1 == 'delete' and decision2 == 'keep'):
                pair_state = 'processed' # One kept, one deleted
                processed_count += 1
            elif decision1 == 'keep' and decision2 == 'keep':
                 pair_state = 'ignored' # Both kept (e.g., "keep both")
                 ignored_count +=1
            # If both are 'delete', that shouldn't happen due to MD5 rules, treat as undecided?
            # If one or both are 'undecided', the pair is effectively undecided.
            elif decision1 == 'undecided' or decision2 == 'undecided':
                 # Check if it was previously processed/ignored, if so, keep that state?
                 # No, if one becomes undecided, the pair should reflect that.
                 undecided_count += 1
                 pair_state = 'undecided'


            self._update_semantic_row_visuals(row, pair_state) # Calls the visual update for the semantic row

        print(f"语义表格同步完成：Processed={processed_count}, Ignored/KeepBoth={ignored_count}, Undecided={undecided_count}")


    # --- Helper to find and update MD5 row visuals given a key ---
    def _update_md5_visuals_for_key(self, decision_key, decision):
        """Finds the row(s) in the MD5 table corresponding to the key and updates its visuals."""
        rows_to_update = []
        # Iterate through the MD5 table mapping to find matching rows
        for row, block_info in self.row_to_block_info.items():
            key = self._create_decision_key(block_info[0], block_info[1], block_info[2])
            if key == decision_key:
                rows_to_update.append(row)

        if not rows_to_update:
            # This block might not be part of an *MD5* duplicate group, so it won't be in the MD5 table.
            # print(f"Debug: Key {decision_key} not found in MD5 table mapping (row_to_block_info).")
            pass # It's okay if it's not in the MD5 table
        else:
            # print(f"Debug: Updating MD5 table rows {rows_to_update} for key {decision_key} to {decision}")
            for row in rows_to_update:
                self._update_md5_row_visuals(row, decision) # Use the specific MD5 visual update method


    # --- Save/Load/Apply Decision Logic ---

    def save_decisions(self):
        """Saves the current block decisions ('keep'/'delete') to a JSON file."""
        print("\n--- '保存决策'按钮点击 ---")

        if not self.block_decisions:
            QMessageBox.information(self, "无决策", "没有可保存的决策（分析可能未运行或未做任何标记）。")
            return

        # Filter decisions: only save those marked 'keep' or 'delete'
        data_to_save = []
        saved_count = 0
        for decision_key, decision in self.block_decisions.items():
            if decision in ['keep', 'delete']:
                try:
                    # Parse the key back into components
                    str_path, index_str, type_str = decision_key.split(self.DECISION_KEY_SEPARATOR)
                    # Store relative path if possible, or absolute path as string
                    data_to_save.append({
                        "file": str_path, # Store the path as a string
                        "index": int(index_str),
                        "type": type_str,
                        "decision": decision
                    })
                    saved_count += 1
                except ValueError:
                    print(f"[错误] 无法解析决策键进行保存: {decision_key}")
                    continue
                except Exception as e:
                     print(f"[错误] 保存决策条目时发生意外错误 for key {decision_key}: {e}")
                     continue


        if not data_to_save:
            QMessageBox.information(self, "无已标记决策", "没有已标记为 '保留' 或 '删除' 的决策可供保存。")
            return

        # Suggest a filename in the original folder or user's home
        default_folder = self.folder_path_edit.text() or os.path.expanduser("~")
        suggested_filename = os.path.join(default_folder, "kd_decisions.json")

        # Open save file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存决策文件",
            suggested_filename,
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            print("用户取消保存。")
            return

        # Write the filtered decisions to the JSON file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            print(f"已将 {saved_count} 条决策成功保存到: {file_path}")
            QMessageBox.information(self, "保存成功", f"已将 {saved_count} 条决策成功保存到:\n{file_path}")
        except Exception as e:
            print(f"[错误] 保存文件时出错: {e}")
            QMessageBox.critical(self, "保存失败", f"保存文件时出错:\n{e}")

    def load_decisions(self):
        """Loads block decisions from JSON, applies them to the current state, and updates visuals."""
        print("\n--- '加载决策'按钮点击 ---")

        # Check if analysis has been run (block_decisions should be initialized)
        if not self.block_decisions:
             QMessageBox.warning(self, "无分析结果", "请先运行分析，然后再加载决策。\n(需要先初始化内部决策状态)")
             return

        # Suggest starting directory
        default_folder = self.folder_path_edit.text() or os.path.expanduser("~")

        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "加载决策文件",
            default_folder,
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            print("用户取消加载。")
            return

        # Load data from JSON
        try:
            print(f"尝试从以下文件加载决策: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            print(f"成功读取 {len(loaded_data)} 条决策记录。")
        except FileNotFoundError:
             print(f"[错误] 文件未找到: {file_path}")
             QMessageBox.critical(self, "加载失败", f"文件未找到:\n{file_path}")
             return
        except json.JSONDecodeError as e:
            print(f"[错误] 解析JSON时出错: {e}")
            QMessageBox.critical(self, "加载失败", f"解析JSON文件时出错 (文件格式可能不正确):\n{e}")
            return
        except Exception as e:
            print(f"[错误] 读取文件时发生意外错误: {e}")
            QMessageBox.critical(self, "加载失败", f"读取文件时发生意外错误:\n{e}")
            return

        # --- Apply loaded decisions ---
        applied_count = 0
        mismatched_count = 0
        invalid_decision_count = 0
        keys_updated = [] # Track which keys were actually updated

        # Store loaded decisions in a temporary map for quick lookup
        loaded_decision_map = {}
        for item in loaded_data:
            if isinstance(item, dict) and all(k in item for k in ["file", "index", "type", "decision"]):
                 # Reconstruct the key using the loaded data
                 key = self._create_decision_key(item['file'], item['index'], item['type'])
                 loaded_decision_map[key] = item['decision']
            else:
                print(f"[警告] 加载的决策文件中发现格式不正确的条目，已忽略: {item}")

        print(f"将 {len(loaded_decision_map)} 条有效决策应用到当前分析结果...")

        # --- FIX: Reset existing decisions to 'undecided' BEFORE applying loaded ones ---
        print("  重置当前内存中的决策为 'undecided'...")
        reset_count = 0
        for key in self.block_decisions:
            self.block_decisions[key] = 'undecided'
            reset_count += 1
        print(f"  已重置 {reset_count} 个块的决策。")
        # -----------------------------------------------------------------------------

        # Iterate through the *loaded* decisions and apply them
        # (No need to iterate through current_keys_set anymore)
        current_keys_set = set(self.block_decisions.keys()) # Get current keys for mismatch check
        for loaded_key, loaded_decision in loaded_decision_map.items():
            if loaded_key in current_keys_set: # Check if the block from the file still exists in current analysis
                if loaded_decision in ['keep', 'delete']:
                    # Apply the loaded decision, overwriting the 'undecided' state
                    self.block_decisions[loaded_key] = loaded_decision
                    applied_count += 1 # Count every successful application from the loaded file
                    keys_updated.append(loaded_key)
                else:
                    print(f"[警告] 加载的决策值无效 '{loaded_decision}' for key {loaded_key}, 保持 'undecided' 状态。")
                    invalid_decision_count += 1
            else:
                # This key from loaded file doesn't match anything in the current analysis
                print(f"[警告] 加载的决策与当前分析结果不匹配 (文件/块可能已更改或不存在)，已忽略: {loaded_key}")
                mismatched_count += 1


        print(f"成功应用/覆盖 {applied_count} 条加载的决策。")
        if mismatched_count > 0: print(f"{mismatched_count} 条加载的决策因与当前分析不匹配而被忽略。")
        if invalid_decision_count > 0: print(f"{invalid_decision_count} 条加载的决策值无效。")

        # --- Refresh Table Visuals ---
        print("正在刷新 MD5 表格视觉效果以匹配加载的决策...")
        self.md5_results_table.setSortingEnabled(False) # Disable sorting for update
        for row, block_info in self.row_to_block_info.items():
             key = self._create_decision_key(block_info[0], block_info[1], block_info[2])
             # Get the potentially updated decision (will be 'undecided' if not in loaded file)
             decision = self.block_decisions.get(key, 'undecided')
             self._update_md5_row_visuals(row, decision)
             # Ensure checkbox is unchecked after load, user needs to re-select if needed
             item = self.md5_results_table.item(row, 0)
             if item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                 item.setCheckState(Qt.CheckState.Unchecked)
        self.md5_results_table.setSortingEnabled(True) # Re-enable sorting
        print("MD5 表格视觉效果刷新完成。")

        # Also refresh semantic table visuals based on the updated block_decisions
        print("准备同步语义表格视觉效果...") # Add log
        self.semantic_results_table.setSortingEnabled(False) # <<< Disable sorting for semantic update
        self._sync_semantic_visuals_from_block_decisions() # Calls the sync function for semantic table
        self.semantic_results_table.setSortingEnabled(True)  # <<< Re-enable sorting for semantic table
        print("语义表格视觉效果同步调用完成。") # Add log

        # --- MODIFICATION: Replace QMessageBox with print statements ---
        print("--- 加载决策完成 ---")
        print(f"  已成功加载并应用 {applied_count} 条决策。")
        print(f"  {mismatched_count} 条不匹配，{invalid_decision_count} 条无效值。")
        print(f"  表格视觉效果已更新。")


    def apply_decisions(self):
        """Applies decisions by generating new files containing only 'kept' or 'undecided' blocks."""
        print("\n--- '应用决策'按钮点击 ---")

        if not self.blocks_data:
            QMessageBox.warning(self, "无分析结果", "请先运行分析，再应用决策。")
            return
        if not self.block_decisions:
             QMessageBox.warning(self, "无决策信息", "决策信息丢失，请重新运行分析或加载决策。")
             return

        # Ask user for the output directory
        output_dir = QFileDialog.getExistingDirectory(self, "选择保存去重后文件的目标文件夹")
        if not output_dir:
            print("用户取消选择目标文件夹。")
            return

        output_dir_path = Path(output_dir)
        print(f"将把去重后的文件保存到: {output_dir_path}")

        # Ensure we know which files were processed in the run
        if not hasattr(self, 'processed_files_in_run') or not self.processed_files_in_run:
             # Fallback: Infer from block_data if the set wasn't populated correctly
             processed_files = {info[0] for info in self.blocks_data}
             if not processed_files:
                 QMessageBox.critical(self, "错误", "无法确定处理了哪些文件，请重新运行分析。")
                 return
             self.processed_files_in_run = processed_files
             print("[警告] 未找到运行时的文件列表，从当前块数据推断。这可能不准确，如果文件在分析后被删除。")

        processed_files_count = 0
        error_files = []
        QApplication.setOverrideCursor(Qt.WaitCursor) # Show busy cursor

        # Process each file that was part of the analysis run
        for original_file_path in self.processed_files_in_run:
            print(f"处理文件: {original_file_path.name}")
            # Get all blocks belonging to this file, sorted by original index
            # Filter blocks_data for the current file and sort by index (item 1)
            blocks_for_this_file = sorted(
                [b for b in self.blocks_data if b[0] == original_file_path],
                key=lambda x: x[1] # Sort by block_index
            )

            if not blocks_for_this_file:
                print(f"文件 {original_file_path.name} 没有提取到内容块，跳过生成。")
                continue

            new_content_parts = []
            kept_block_count = 0
            deleted_block_count = 0

            # Iterate through the blocks of this file in their original order
            for block_info in blocks_for_this_file:
                # Get the decision for this block
                decision_key = self._create_decision_key(block_info[0], block_info[1], block_info[2])
                # Default to 'keep' if undecided, as user hasn't explicitly marked for deletion
                decision = self.block_decisions.get(decision_key, 'keep')

                if decision != 'delete':
                    kept_block_count += 1
                    # Reconstruct the block's original Markdown-like format (simplified)
                    file_path, b_index, b_type, b_text = block_info
                    formatted_text = b_text # Start with the raw text

                    # Add back some basic Markdown syntax based on type
                    # This is a simplification and might not perfectly match original formatting
                    if b_type == 'heading':
                        # Headings were excluded, but if they were included, format like this:
                        # level = token.get('level', 2) # Assuming level was stored
                        # formatted_text = f"{'#' * level} {b_text}"
                        pass # Currently excluded
                    elif b_type == 'block_code':
                        # Ensure code blocks have fences, handle potential language specifier if stored
                        # Assuming b_text contains only the code, not the fences/lang
                        lang = "" # Extract lang if stored during parsing
                        if not b_text.startswith("```"):
                             formatted_text = f"```{lang}\n{b_text}\n```"
                        # else: assume b_text already includes fences
                    elif b_type == 'list_item':
                        # Assume simple unordered list items
                        formatted_text = f"- {b_text}" # Add back the dash
                    elif b_type == 'block_quote':
                        # Add '>' prefix to each line
                        lines = b_text.splitlines()
                        formatted_text = "\n".join([f"> {line}" for line in lines])
                    # Add rules for other types (paragraph, etc.) if needed
                    # Paragraphs often don't need extra formatting unless joining affects them

                    new_content_parts.append(formatted_text)
                else:
                    deleted_block_count +=1

            # Join the kept parts with appropriate spacing (e.g., double newline)
            new_content = "\n\n".join(new_content_parts)

            # Construct the output filename
            output_filename = f"{original_file_path.stem}_deduped{original_file_path.suffix}"
            output_filepath = output_dir_path / output_filename

            # Write the new content to the output file
            try:
                output_filepath.write_text(new_content, encoding='utf-8')
                print(f"  -> 已生成新文件: {output_filepath.name} (保留 {kept_block_count} 块, 删除 {deleted_block_count} 块)")
                processed_files_count += 1
            except Exception as e:
                print(f"[错误] 写入文件 '{output_filepath.name}' 时出错: {e}")
                error_files.append(original_file_path.name)

        QApplication.restoreOverrideCursor() # Restore normal cursor

        # --- Report results ---
        if not error_files:
            QMessageBox.information(self, "应用完成", f"已成功为 {processed_files_count} 个文件生成去重后的版本，保存在:\n{output_dir_path}")
        else:
            QMessageBox.warning(self, "部分完成",
                                f"为 {processed_files_count} 个文件生成了新版本。\n"
                                f"但以下 {len(error_files)} 个文件的处理失败:\n" + "\n".join(error_files) +
                                f"\n\n请检查目标文件夹权限或错误日志。\n输出目录: {output_dir_path}")

        print("--- 应用决策完成 ---")


    # --- Text Extraction Helper (Used by Parsing) ---
    def _extract_text_from_children(self, children):
        """Helper function to recursively extract raw text from mistune token children."""
        text = ""
        if children is None:
            return ""
        for child in children:
            child_type = child.get('type')
            if child_type == 'text':
                text += child.get('raw', '') # Use 'raw' for the exact text
            elif child_type == 'codespan':
                text += child.get('raw', '') # Include backticks if needed? Raw usually has them.
            elif child_type in ['link', 'image', 'emphasis', 'strong', 'strikethrough']:
                # Recursively extract text from children of inline formatting elements
                text += self._extract_text_from_children(child.get('children', []))
            elif child_type == 'softbreak' or child_type == 'linebreak':
                # Replace line breaks with a space for continuous text within a block
                text += ' '
            elif child_type == 'inline_html':
                # Optionally include or exclude inline HTML content
                # text += child.get('raw', '')
                pass # Exclude inline HTML for now
            # Add other inline types if necessary (e.g., footnote_ref)
        # Use strip() judiciously, maybe only at the end of the whole block extraction
        # return text.strip()
        return text # Return potentially leading/trailing spaces for joining


    # --- Semantic Detail Dialog Slot ---
    def show_semantic_detail(self, item):
        """Shows a dialog with the full text of the double-clicked semantic pair, highlighting the blocks."""
        if item is None: return # Clicked on empty area
        row = item.row()
        pair_info = self.semantic_row_to_pair_info.get(row)

        if pair_info:
            # Ensure pair_info has the correct number of elements before unpacking
            if len(pair_info) == 3:
                info1, info2, score = pair_info
            else:
                print(f"[错误] 语义行 {row} 的数据格式不正确: {pair_info}")
                QMessageBox.warning(self, "内部错误", f"无法显示行 {row} 的详情，数据格式错误。")
                return

            file_path1 = info1[0] # Path object
            file_path2 = info2[0] # Path object

            # Retrieve the full original content from memory
            full_text1 = self.markdown_files_content.get(file_path1)
            full_text2 = self.markdown_files_content.get(file_path2)

            # Check if content was loaded correctly
            if full_text1 is None:
                 print(f"[错误] 无法获取文件 '{file_path1.name}' 的原始内容。")
                 QMessageBox.warning(self, "错误", f"无法获取文件 '{file_path1.name}' 的原始内容以显示详情。")
                 return
            if full_text2 is None:
                 print(f"[错误] 无法获取文件 '{file_path2.name}' 的原始内容。")
                 QMessageBox.warning(self, "错误", f"无法获取文件 '{file_path2.name}' 的原始内容以显示详情。")
                 return

            # Create and show the dialog
            dialog = SemanticDetailDialog(score, info1, info2, full_text1, full_text2, self) # Pass self as parent
            dialog.highlight_and_scroll() # Call highlight method after dialog is created
            dialog.exec() # Show dialog modally
        else:
            print(f"[警告] 无法找到语义表格第 {row} 行对应的详细信息。")


# --- 程序入口 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Apply a basic style (optional)
    # app.setStyle("Fusion")

    # You can also set a dark mode palette (example)
    # app.setStyle("Fusion")
    # palette = QPalette()
    # palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    # palette.setColor(QPalette.ColorRole.WindowText, Qt.white)
    # # ... set other colors ...
    # app.setPalette(palette)

    window = KDToolWindow()
    window.show()
    sys.exit(app.exec())
