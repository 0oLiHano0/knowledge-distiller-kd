import sys
import os
from pathlib import Path
import hashlib
import collections
import mistune
import json
from sentence_transformers import SentenceTransformer, util # Ensure this is installed
import time

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QTextEdit, QLabel, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QSizePolicy, QDialog, QDialogButtonBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette, QTextCharFormat, QTextCursor, QFont

# --- Dialog Class (No changes needed) ---
class SemanticDetailDialog(QDialog):
    # ... (Implementation remains the same) ...
    def __init__(self, score, info1, info2, full_text1, full_text2, parent=None):
        super().__init__(parent)
        self.setWindowTitle("语义相似块详情 (含上下文)")
        self.setMinimumSize(800, 600)
        self.info1 = info1; self.info2 = info2
        self.full_text1 = full_text1 if full_text1 is not None else "[错误] 无法加载文件内容"
        self.full_text2 = full_text2 if full_text2 is not None else "[错误] 无法加载文件内容"
        file_path1, self.b_index1, self.b_type1, self.b_text1 = info1
        file_path2, self.b_index2, self.b_type2, self.b_text2 = info2
        info_label1 = QLabel(f"<b>文件 1:</b> {file_path1.name} ({self.b_type1} #{self.b_index1})")
        info_label2 = QLabel(f"<b>文件 2:</b> {file_path2.name} ({self.b_type2} #{self.b_index2})")
        score_label = QLabel(f"<b>相似度:</b> {score:.4f}")
        self.text_edit1 = QTextEdit(); self.text_edit1.setPlainText(self.full_text1); self.text_edit1.setReadOnly(True)
        self.text_edit2 = QTextEdit(); self.text_edit2.setPlainText(self.full_text2); self.text_edit2.setReadOnly(True)
        font = QFont("Courier New", 11); self.text_edit1.setFont(font); self.text_edit2.setFont(font)
        main_layout = QVBoxLayout(self); main_layout.addWidget(score_label)
        content_layout = QHBoxLayout(); layout1 = QVBoxLayout(); layout1.addWidget(info_label1); layout1.addWidget(self.text_edit1); layout2 = QVBoxLayout(); layout2.addWidget(info_label2); layout2.addWidget(self.text_edit2)
        content_layout.addLayout(layout1); content_layout.addLayout(layout2); main_layout.addLayout(content_layout)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok); button_box.accepted.connect(self.accept); main_layout.addWidget(button_box)

    def highlight_and_scroll(self):
        highlight_color = QColor("yellow"); fmt = QTextCharFormat(); fmt.setBackground(highlight_color)
        cursor1 = self.text_edit1.document().find(self.b_text1);
        if not cursor1.isNull(): cursor1.mergeCharFormat(fmt); self.text_edit1.setTextCursor(cursor1); self.text_edit1.ensureCursorVisible()
        else: print(f"[警告] 无法在文件1的完整内容中定位块文本: {self.b_text1[:50]}...")
        cursor2 = self.text_edit2.document().find(self.b_text2);
        if not cursor2.isNull(): cursor2.mergeCharFormat(fmt); self.text_edit2.setTextCursor(cursor2); self.text_edit2.ensureCursorVisible()
        else: print(f"[警告] 无法在文件2的完整内容中定位块文本: {self.b_text2[:50]}...")


# --- Main Window Class ---
class KDToolWindow(QWidget):
    SEMANTIC_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
    SIMILARITY_THRESHOLD = 0.85
    DECISION_KEY_SEPARATOR = "::"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("知识蒸馏 KD Tool v1.21 - Correct Semantic Logic") # Update title
        self.setGeometry(200, 100, 800, 800)

        # --- 定义颜色常量 ---
        self.color_keep = QColor("lightgreen")
        self.color_delete = QColor(255, 200, 200) # Light red
        self.color_semantic_processed = QColor(220, 220, 220) # Light grey for processed semantic pairs
        self.color_undecided = None

        # --- 创建界面控件 ---
        # (No changes here)
        self.folder_label = QLabel("目标文件夹:")
        self.folder_path_edit = QLineEdit(); self.folder_path_edit.setReadOnly(True)
        self.browse_button = QPushButton("选择文件夹...")
        self.start_button = QPushButton("开始分析")
        self.md5_status_label = QLabel("精确重复项列表:")
        self.md5_results_table = QTableWidget(); self.setup_md5_results_table()
        self.keep_button = QPushButton("标记选中项为 '保留'")
        self.delete_button = QPushButton("标记选中项为 '删除'")
        self.semantic_status_label = QLabel("语义相似项列表 (双击查看详情):")
        self.semantic_results_table = QTableWidget(); self.setup_semantic_table()
        self.semantic_keep1_del2_button = QPushButton("保留块1, 删除块2")
        self.semantic_keep2_del1_button = QPushButton("保留块2, 删除块1")
        self.semantic_keep_both_button = QPushButton("全部保留 (忽略此对)")
        self.save_button = QPushButton("保存决策...")
        self.load_button = QPushButton("加载决策...")
        self.apply_button = QPushButton("应用决策 (生成新文件)...")

        # --- 设置布局 ---
        # (No changes here)
        main_layout = QVBoxLayout(self)
        folder_layout = QHBoxLayout(); folder_layout.addWidget(self.folder_label); folder_layout.addWidget(self.folder_path_edit); folder_layout.addWidget(self.browse_button)
        main_layout.addLayout(folder_layout)
        main_layout.addWidget(self.start_button)
        main_layout.addWidget(self.md5_status_label); main_layout.addWidget(self.md5_results_table); self.md5_results_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        md5_action_layout = QHBoxLayout(); md5_action_layout.addWidget(self.keep_button); md5_action_layout.addWidget(self.delete_button); md5_action_layout.addStretch()
        main_layout.addLayout(md5_action_layout)
        main_layout.addWidget(self.semantic_status_label); main_layout.addWidget(self.semantic_results_table); self.semantic_results_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        semantic_action_layout = QHBoxLayout(); semantic_action_layout.addWidget(self.semantic_keep1_del2_button); semantic_action_layout.addWidget(self.semantic_keep2_del1_button); semantic_action_layout.addWidget(self.semantic_keep_both_button); semantic_action_layout.addStretch()
        main_layout.addLayout(semantic_action_layout)
        global_action_layout = QHBoxLayout(); global_action_layout.addStretch(); global_action_layout.addWidget(self.load_button); global_action_layout.addWidget(self.save_button); global_action_layout.addWidget(self.apply_button)
        main_layout.addLayout(global_action_layout)
        main_layout.setStretchFactor(self.md5_results_table, 1); main_layout.setStretchFactor(self.semantic_results_table, 1)

        # --- 连接信号与槽 ---
        # (No changes here)
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

        # --- 类成员变量 ---
        # (No changes here)
        self.markdown_files_content = {}
        self.blocks_data = []
        self.duplicate_blocks = {}
        self.row_to_block_info = {}
        self.block_decisions = {}
        self.semantic_duplicates = []
        self.semantic_row_to_pair_info = {}
        self.semantic_pair_decisions = {}
        self.semantic_model = None
        self.processed_files_in_run = set()

        self._load_semantic_model()

    # --- Helper method to create the decision key ---
    def _create_decision_key(self, file_path, block_index, block_type):
        """Creates a unique string key for a block."""
        return f"{str(file_path)}{self.DECISION_KEY_SEPARATOR}{int(block_index)}{self.DECISION_KEY_SEPARATOR}{str(block_type)}"

    def _load_semantic_model(self):
        """Loads the sentence transformer model."""
        # (No changes needed here)
        try:
            print(f"正在加载语义模型: {self.SEMANTIC_MODEL_NAME} ... (首次加载可能需要下载)")
            start_time = time.time(); self.semantic_model = SentenceTransformer(self.SEMANTIC_MODEL_NAME); end_time = time.time()
            print(f"语义模型加载完成，耗时: {end_time - start_time:.2f} 秒")
        except Exception as e:
            print(f"[严重错误] 无法加载语义模型: {e}"); QMessageBox.critical(self, "模型加载失败", f"无法加载语义模型 '{self.SEMANTIC_MODEL_NAME}'.\n错误: {e}\n\n语义去重功能将不可用。"); self.semantic_model = None

    # --- Renamed setup method for MD5 table ---
    def setup_md5_results_table(self):
        """Configures the QTableWidget for MD5 duplicates."""
        # (No changes needed here)
        self.md5_results_table.setColumnCount(6); self.md5_results_table.setHorizontalHeaderLabels(["选择", "重复组", "文件名", "类型", "块索引", "内容预览"]); self.md5_results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); self.md5_results_table.setSortingEnabled(True); header = self.md5_results_table.horizontalHeader(); header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents); header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch); self.md5_results_table.setAlternatingRowColors(True)

    # --- Setup method for Semantic table ---
    def setup_semantic_table(self):
        """Configures the QTableWidget for semantic duplicates."""
        # (No changes needed here)
        self.semantic_results_table.setColumnCount(6); self.semantic_results_table.setHorizontalHeaderLabels(["选择", "相似度", "块 1 (文件/类型/索引)", "块 1 预览", "块 2 (文件/类型/索引)", "块 2 预览"]); self.semantic_results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); self.semantic_results_table.setSortingEnabled(True); header = self.semantic_results_table.horizontalHeader(); header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents); header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive); header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive); header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch); header.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive); header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch); self.semantic_results_table.setAlternatingRowColors(True)

    def browse_folder(self):
        """Opens a directory selection dialog."""
        # (No changes needed here)
        folder_path = QFileDialog.getExistingDirectory(self, "选择包含 Markdown 文件的文件夹")
        if folder_path:
            self.folder_path_edit.setText(folder_path); self.md5_results_table.setRowCount(0); self.semantic_results_table.setRowCount(0); self.row_to_block_info = {}; self.semantic_row_to_pair_info = {}; self.block_decisions = {}; self.semantic_duplicates = []; self.processed_files_in_run = set(); self.semantic_pair_decisions = {}
            print(f"已选择文件夹: {folder_path}")
        else: print("用户取消选择文件夹。")

    def start_analysis(self):
        """Full analysis pipeline: Read -> Parse (Exclude Headings) -> Init Decisions -> MD5 Dedup -> Semantic Dedup -> Populate BOTH tables."""
        selected_folder_str = self.folder_path_edit.text()
        # --- Reset state ---
        self.markdown_files_content = {}; self.blocks_data = []; self.duplicate_blocks = {}
        self.md5_results_table.setRowCount(0); self.semantic_results_table.setRowCount(0)
        self.row_to_block_info = {}; self.semantic_row_to_pair_info = {}
        self.block_decisions = {}; self.semantic_duplicates = []; self.processed_files_in_run = set(); self.semantic_pair_decisions = {}
        if not selected_folder_str: print("[错误] 请先选择一个文件夹！"); return
        selected_folder = Path(selected_folder_str)
        if not selected_folder.is_dir(): print(f"[错误] 选择的路径不是一个有效的文件夹: {selected_folder_str}"); return
        print(f"开始分析文件夹: {selected_folder}\n---")

        # --- 1. Find and Read Files ---
        # (Same as before)
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

        # --- 2. Parse Markdown and Extract Blocks (Exclude Headings) ---
        # (Same as before)
        print("---")
        print("开始解析 Markdown 并提取内容块 (排除标题)...")
        self.blocks_data = []
        markdown_parser = mistune.create_markdown(renderer=None)
        try:
            for md_file_path, content in self.markdown_files_content.items():
                self.processed_files_in_run.add(md_file_path)
                block_tokens = markdown_parser(content)
                for index, token in enumerate(block_tokens):
                    block_type = token['type']; block_text = ""; blocks_to_add = []
                    if block_type == 'heading': continue # Skip headings
                    elif block_type == 'paragraph': block_text = self._extract_text_from_children(token.get('children', []));
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
            print(f"Markdown 解析完成：共获得 {total_blocks} 个有效内容块 (已排除标题)。")
        except Exception as e: print(f"解析 Markdown 时出错: {e}"); return

        # --- 3. Initialize Decisions for ALL Parsed Blocks ---
        # ** CRITICAL FIX: Initialize decisions for ALL non-heading blocks **
        print("---")
        print("初始化所有有效块的决策状态...")
        self.block_decisions = {} # Ensure it's clear
        try:
            for block_info in self.blocks_data: # Iterate through ALL blocks
                key = self._create_decision_key(block_info[0], block_info[1], block_info[2])
                self.block_decisions[key] = 'undecided'
            print(f"已为 {len(self.block_decisions)} 个有效块初始化决策。")
        except Exception as e:
            print(f"[错误] 初始化决策时出错: {e}")
            return

        # --- 4. Calculate MD5 Hashes and Find Duplicates ---
        # (Same as before)
        print("---")
        print("开始计算 MD5 并查找重复内容块...")
        hash_map = collections.defaultdict(list)
        try:
            for block_info in self.blocks_data: hash_object = hashlib.md5(block_info[3].encode('utf-8')); hex_dig = hash_object.hexdigest(); hash_map[hex_dig].append(block_info)
            self.duplicate_blocks = {h: b for h, b in hash_map.items() if len(b) > 1}; print(f"找到 {len(self.duplicate_blocks)} 组完全重复的内容块 (不含标题)。")
            # --- 5. Populate MD5 Table ---
            self.md5_results_table.setSortingEnabled(False); self.md5_results_table.setRowCount(0); current_row = 0; group_id_counter = 1; self.row_to_block_info = {}
            if not self.duplicate_blocks: print("未找到完全重复的内容块。")
            else:
                print("开始填充 MD5 重复项表格...")
                for hash_val, b_list in self.duplicate_blocks.items():
                    group_id_str = f"组 {group_id_counter}"
                    for block_info in b_list: self._populate_md5_table_row(current_row, group_id_str, block_info); self.row_to_block_info[current_row] = block_info; current_row += 1
                    group_id_counter += 1
            self.md5_results_table.setSortingEnabled(True); print("MD5 表格填充完成。"); print("---"); print("MD5 去重分析完成。")
        except Exception as e: print(f"计算 MD5、查找或填充表格时出错: {e}")

        # --- 6. Semantic Deduplication (Uses filtered blocks_data) ---
        # (No changes needed here)
        print("---")
        print("开始进行语义相似度分析 (不含标题)...")
        if self.semantic_model is None: print("[错误] 语义模型未加载，跳过语义分析。"); return
        if len(self.blocks_data) < 2: print("内容块数量不足 (<2)，无法进行语义比较。"); return
        try:
            start_time = time.time(); block_texts = [info[3] for info in self.blocks_data]
            print(f"正在为 {len(block_texts)} 个有效内容块计算向量嵌入..."); embeddings = self._calculate_embeddings(block_texts); embed_time = time.time()
            print(f"向量嵌入计算完成，耗时: {embed_time - start_time:.2f} 秒")
            if embeddings is None: raise Exception("Embedding calculation failed.")
            print(f"正在查找相似度 > {self.SIMILARITY_THRESHOLD} 的内容块对..."); self.semantic_duplicates = self._find_similar_pairs(embeddings, self.blocks_data, self.SIMILARITY_THRESHOLD); find_time = time.time()
            print(f"语义相似对查找完成，耗时: {find_time - embed_time:.2f} 秒")
            # --- 7. Populate Semantic Table ---
            self.semantic_results_table.setSortingEnabled(False); self.semantic_results_table.setRowCount(0); current_semantic_row = 0; self.semantic_row_to_pair_info = {}
            if not self.semantic_duplicates: print("未找到语义相似的内容块对。")
            else:
                print(f"开始填充语义相似项表格 ({len(self.semantic_duplicates)} 对)...")
                for info1, info2, score in self.semantic_duplicates:
                    self._populate_semantic_table_row(current_semantic_row, score, info1, info2)
                    self.semantic_row_to_pair_info[current_semantic_row] = (info1, info2, score)
                    current_semantic_row += 1
            self.semantic_results_table.setSortingEnabled(True); print("语义表格填充完成。"); print("--- 语义分析完成 ---")
        except Exception as e: print(f"[错误] 语义分析或填充表格时出错: {e}"); QMessageBox.critical(self, "语义分析失败", f"语义分析或填充表格时出错:\n{e}")


    # --- Renamed populator for MD5 table ---
    def _populate_md5_table_row(self, row, group_id_str, block_info):
        """Helper method to populate a single row in the MD5 results table."""
        # (No changes needed here)
        file_path, b_index, b_type, b_text = block_info; preview_text = (b_text[:100] + '...') if len(b_text) > 100 else b_text; preview_text = preview_text.replace('\n', ' '); checkbox_item = QTableWidgetItem(); checkbox_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled); checkbox_item.setCheckState(Qt.CheckState.Unchecked); item_group = QTableWidgetItem(group_id_str); item_file = QTableWidgetItem(file_path.name); item_type = QTableWidgetItem(b_type); item_index = QTableWidgetItem(str(b_index)); item_preview = QTableWidgetItem(preview_text); self.md5_results_table.insertRow(row); self.md5_results_table.setItem(row, 0, checkbox_item); self.md5_results_table.setItem(row, 1, item_group); self.md5_results_table.setItem(row, 2, item_file); self.md5_results_table.setItem(row, 3, item_type); self.md5_results_table.setItem(row, 4, item_index); self.md5_results_table.setItem(row, 5, item_preview)

    # --- UPDATED populator for Semantic table (Make checkbox checkable) ---
    def _populate_semantic_table_row(self, row, score, block_info1, block_info2):
        """Helper method to populate a single row in the semantic results table."""
        # (No changes needed here)
        file_path1, b_index1, b_type1, b_text1 = block_info1; file_path2, b_index2, b_type2, b_text2 = block_info2; score_str = f"{score:.4f}"; preview1 = (b_text1[:70] + '...') if len(b_text1) > 70 else b_text1; preview1 = preview1.replace('\n', ' '); preview2 = (b_text2[:70] + '...') if len(b_text2) > 70 else b_text2; preview2 = preview2.replace('\n', ' '); info1_str = f"{file_path1.name} ({b_type1} #{b_index1})"; info2_str = f"{file_path2.name} ({b_type2} #{b_index2})";
        checkbox_item = QTableWidgetItem();
        checkbox_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled); # Make checkable
        checkbox_item.setCheckState(Qt.CheckState.Unchecked);
        item_score = QTableWidgetItem(score_str); item_info1 = QTableWidgetItem(info1_str); item_preview1 = QTableWidgetItem(preview1); item_info2 = QTableWidgetItem(info2_str); item_preview2 = QTableWidgetItem(preview2); item_score.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter); self.semantic_results_table.insertRow(row); self.semantic_results_table.setItem(row, 0, checkbox_item); self.semantic_results_table.setItem(row, 1, item_score); self.semantic_results_table.setItem(row, 2, item_info1); self.semantic_results_table.setItem(row, 3, item_preview1); self.semantic_results_table.setItem(row, 4, item_info2); self.semantic_results_table.setItem(row, 5, item_preview2)

    # --- Slot Methods for MD5 Action Buttons ---
    def mark_selected_keep(self):
        """Handles 'Keep' button click FOR MD5 DUPLICATES."""
        # (No changes needed here)
        print("\n--- '标记为保留'按钮点击 (精确重复) ---"); selected_full_block_infos = self._get_selected_md5_block_info();
        if not selected_full_block_infos: print("没有选中任何精确重复项。"); return
        print(f"共选中 {len(selected_full_block_infos)} 项。将更新这些项的决策为 '保留':"); count = 0; affected_rows = []
        for full_block_info in selected_full_block_infos:
            decision_key = self._create_decision_key(full_block_info[0], full_block_info[1], full_block_info[2])
            if decision_key in self.block_decisions:
                self.block_decisions[decision_key] = 'keep'; count += 1
                for row, info in self.row_to_block_info.items():
                    current_key = self._create_decision_key(info[0], info[1], info[2])
                    if current_key == decision_key: affected_rows.append(row)
            else: print(f"[警告] 无法在决策字典中找到键: {decision_key}") # Should not happen now
        unique_affected_rows = sorted(list(set(affected_rows)));
        for row in unique_affected_rows: self._update_md5_row_visuals(row, 'keep')
        print(f"已将 {count} 项标记为 '保留' (状态已在内存中更新，行背景已更改)。");
        self.clear_md5_selection()
        QApplication.processEvents()

    def mark_selected_delete(self):
        """Handles 'Delete' button click FOR MD5 DUPLICATES."""
        # (No changes needed here)
        print("\n--- '标记为删除'按钮点击 (精确重复) ---"); selected_full_block_infos = self._get_selected_md5_block_info();
        if not selected_full_block_infos: print("没有选中任何项。"); return
        print(f"共选中 {len(selected_full_block_infos)} 项尝试标记为 '删除'..."); actually_marked_count = 0; skipped_count = 0; marked_keys = []; skipped_keys = []
        hash_to_all_keys_in_group = collections.defaultdict(list);
        for hash_val, block_list in self.duplicate_blocks.items():
             for block_info in block_list: decision_key = self._create_decision_key(block_info[0], block_info[1], block_info[2]); hash_to_all_keys_in_group[hash_val].append(decision_key)
        for full_block_info_to_delete in selected_full_block_infos:
            decision_key_to_delete = self._create_decision_key(full_block_info_to_delete[0], full_block_info_to_delete[1], full_block_info_to_delete[2]); block_hash = hashlib.md5(full_block_info_to_delete[3].encode('utf-8')).hexdigest()
            if block_hash in hash_to_all_keys_in_group:
                group_member_keys = hash_to_all_keys_in_group[block_hash]
                if not group_member_keys: print(f"[内部错误] 找不到哈希 {block_hash} 对应的键列表。"); continue
                non_delete_count = sum(1 for member_key in group_member_keys if self.block_decisions.get(member_key, 'undecided') != 'delete')
                current_decision = self.block_decisions.get(decision_key_to_delete, 'undecided'); is_last_one = (non_delete_count <= 1 and current_decision != 'delete')
                if is_last_one: skipped_count += 1; skipped_keys.append(decision_key_to_delete); print(f"[跳过] '{full_block_info_to_delete[0].name}' ({full_block_info_to_delete[2]} #{full_block_info_to_delete[1]}) 是其重复组中最后一个非删除项。")
                else:
                    if decision_key_to_delete in self.block_decisions: self.block_decisions[decision_key_to_delete] = 'delete'; marked_keys.append(decision_key_to_delete); actually_marked_count += 1
                    else: print(f"[警告] 无法在决策字典中找到键: {decision_key_to_delete}") # Should not happen now
            else: print(f"[警告] 选中的块未在重复组中找到 (哈希不匹配?): {decision_key_to_delete}")
        affected_rows_delete = []; affected_rows_skipped = []
        for row, info in self.row_to_block_info.items():
            key = self._create_decision_key(info[0], info[1], info[2])
            if key in marked_keys: affected_rows_delete.append(row)
            elif key in skipped_keys: affected_rows_skipped.append(row)
        for row in affected_rows_delete: self._update_md5_row_visuals(row, 'delete')
        print(f"操作完成：成功将 {actually_marked_count} 项标记为 '删除'。")
        if skipped_count > 0: print(f"有 {skipped_count} 项因需保留至少一个实例而被跳过。");
        self.clear_md5_selection()
        QApplication.processEvents()

    # --- Renamed/Specific methods for MD5 Table Interaction ---
    def clear_md5_selection(self):
        """Unchecks all checkboxes in the MD5 results table."""
        # (No changes needed here)
        for row in range(self.md5_results_table.rowCount()):
            item = self.md5_results_table.item(row, 0)
            if item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable: item.setCheckState(Qt.CheckState.Unchecked)

    def _update_md5_row_visuals(self, row, decision):
        """Sets the background color of a row in the MD5 results table."""
        # (No changes needed here)
        default_bg_color = self.md5_results_table.palette().color(QPalette.ColorRole.Base); alt_bg_color = default_bg_color.darker(105) if row % 2 == 1 else default_bg_color; color_to_set = alt_bg_color
        if decision == 'keep': color_to_set = self.color_keep
        elif decision == 'delete': color_to_set = self.color_delete
        for col in range(self.md5_results_table.columnCount()):
            item = self.md5_results_table.item(row, col)
            if item: item.setBackground(color_to_set)

    def _get_selected_md5_block_info(self):
        """Gets block_info for checked rows in the MD5 results table."""
        # (No changes needed here)
        selected = []
        for row in range(self.md5_results_table.rowCount()):
            checkbox_item = self.md5_results_table.item(row, 0)
            if checkbox_item and checkbox_item.checkState() == Qt.CheckState.Checked:
                block_info = self.row_to_block_info.get(row);
                if block_info: selected.append(block_info)
        return selected

    # --- Slot Methods for Semantic Action Buttons (UPDATED) ---
    def mark_keep1_delete2(self):
        """Marks block 1 as keep, block 2 as delete for the SINGLE selected semantic pair."""
        print("\n--- '保留块1, 删除块2' 按钮点击 ---")
        selected_pairs = self._get_selected_semantic_pairs()
        if len(selected_pairs) != 1: QMessageBox.warning(self, "选择错误", "请在下方表格中刚好选择一项（一对相似块）以执行此操作。"); print(f"需要刚好选择 1 对，当前选择了 {len(selected_pairs)} 对。"); return

        info1, info2, score, row = selected_pairs[0]
        key1 = self._create_decision_key(info1[0], info1[1], info1[2])
        key2 = self._create_decision_key(info2[0], info2[1], info2[2])

        print(f"将标记 '{info1[0].name}' #{info1[1]} 为 'keep', '{info2[0].name}' #{info2[1]} 为 'delete'")
        if key1 in self.block_decisions: self.block_decisions[key1] = 'keep'
        else: print(f"[警告] 尝试更新未初始化的决策键: {key1}") # Should not happen now
        if key2 in self.block_decisions: self.block_decisions[key2] = 'delete'
        else: print(f"[警告] 尝试更新未初始化的决策键: {key2}") # Should not happen now

        # --- Update Semantic table row visual ---
        self._update_semantic_row_visuals(row, 'processed') # Mark row as processed
        # --- Update MD5 visuals if blocks exist there ---
        self._update_md5_visuals_for_key(key1, 'keep')
        self._update_md5_visuals_for_key(key2, 'delete')

        self.clear_semantic_selection()
        QApplication.processEvents()

    def mark_keep2_delete1(self):
        """Marks block 2 as keep, block 1 as delete for the SINGLE selected semantic pair."""
        print("\n--- '保留块2, 删除块1' 按钮点击 ---")
        selected_pairs = self._get_selected_semantic_pairs()
        if len(selected_pairs) != 1: QMessageBox.warning(self, "选择错误", "请在下方表格中刚好选择一项（一对相似块）以执行此操作。"); print(f"需要刚好选择 1 对，当前选择了 {len(selected_pairs)} 对。"); return

        info1, info2, score, row = selected_pairs[0]
        key1 = self._create_decision_key(info1[0], info1[1], info1[2])
        key2 = self._create_decision_key(info2[0], info2[1], info2[2])

        print(f"将标记 '{info1[0].name}' #{info1[1]} 为 'delete', '{info2[0].name}' #{info2[1]} 为 'keep'")
        if key1 in self.block_decisions: self.block_decisions[key1] = 'delete'
        else: print(f"[警告] 尝试更新未初始化的决策键: {key1}")
        if key2 in self.block_decisions: self.block_decisions[key2] = 'keep'
        else: print(f"[警告] 尝试更新未初始化的决策键: {key2}")

        # --- Update Semantic table row visual ---
        self._update_semantic_row_visuals(row, 'processed') # Mark row as processed
        # --- Update MD5 visuals if blocks exist there ---
        self._update_md5_visuals_for_key(key1, 'delete')
        self._update_md5_visuals_for_key(key2, 'keep')

        self.clear_semantic_selection()
        QApplication.processEvents()

    def mark_both_keep(self):
        """Marks both blocks in selected semantic pair(s) as keep (ignore similarity)."""
        print("\n--- '全部保留 (忽略此对)' 按钮点击 ---")
        selected_pairs = self._get_selected_semantic_pairs()
        if not selected_pairs: print("没有选中任何语义相似对。"); return

        print(f"共选中 {len(selected_pairs)} 对。将更新这些对中所有块的决策为 '保留':")
        count = 0; affected_keys = []; affected_semantic_rows = []
        for info1, info2, score, row in selected_pairs:
            key1 = self._create_decision_key(info1[0], info1[1], info1[2])
            key2 = self._create_decision_key(info2[0], info2[1], info2[2])
            if key1 in self.block_decisions: self.block_decisions[key1] = 'keep'; affected_keys.append(key1); count+=1
            else: print(f"[警告] 尝试更新未初始化的决策键: {key1}")
            if key2 in self.block_decisions: self.block_decisions[key2] = 'keep'; affected_keys.append(key2); count+=1
            else: print(f"[警告] 尝试更新未初始化的决策键: {key2}")
            affected_semantic_rows.append(row) # Track semantic row

        # --- Update MD5 visuals (if applicable) ---
        affected_rows_md5 = []
        for row_md5, info_md5 in self.row_to_block_info.items():
            key = self._create_decision_key(info_md5[0], info_md5[1], info_md5[2])
            if key in affected_keys: affected_rows_md5.append(row_md5)
        unique_affected_rows_md5 = sorted(list(set(affected_rows_md5)));
        for row_md5 in unique_affected_rows_md5: self._update_md5_row_visuals(row_md5, 'keep')
        # --------------------------------------------

        # --- Update Semantic visuals ---
        for row in affected_semantic_rows:
            self._update_semantic_row_visuals(row, 'ignored') # Mark as ignored/processed
        # -----------------------------

        print(f"已将 {count} 个块标记为 '保留' (状态已在内存中更新，对应精确重复项背景已更改)。")
        self.clear_semantic_selection()
        QApplication.processEvents()


    # --- Helper to find and update MD5 row visuals given a key ---
    def _update_md5_visuals_for_key(self, decision_key, decision):
        """Finds the row(s) in the MD5 table corresponding to the key and updates its visuals."""
        # (No changes needed here)
        rows_to_update = []
        for row, block_info in self.row_to_block_info.items():
            key = self._create_decision_key(block_info[0], block_info[1], block_info[2])
            if key == decision_key: rows_to_update.append(row)
        if not rows_to_update: pass
        else:
            # print(f"Debug: Updating MD5 table rows {rows_to_update} for key {decision_key} to {decision}")
            for row in rows_to_update: self._update_md5_row_visuals(row, decision)


    # --- Methods for Semantic Table Interaction ---
    def clear_semantic_selection(self):
        """Unchecks all checkboxes in the semantic results table."""
        # (No changes needed here)
        for row in range(self.semantic_results_table.rowCount()):
            item = self.semantic_results_table.item(row, 0)
            if item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable: item.setCheckState(Qt.CheckState.Unchecked)

    def _get_selected_semantic_pairs(self):
        """Gets pair_info for checked rows in the semantic results table."""
        # (No changes needed here)
        selected_pairs = []
        for row in range(self.semantic_results_table.rowCount()):
            checkbox_item = self.semantic_results_table.item(row, 0)
            if checkbox_item and (checkbox_item.flags() & Qt.ItemFlag.ItemIsUserCheckable) and checkbox_item.checkState() == Qt.CheckState.Checked:
                pair_info = self.semantic_row_to_pair_info.get(row)
                if pair_info: selected_pairs.append(pair_info + (row,)) # Add row index
        return selected_pairs

    # --- UPDATED Helper Method for Semantic Visuals ---
    def _update_semantic_row_visuals(self, row, decision_type):
        """Sets the background color of a row in the semantic results table."""
        default_bg_color = self.semantic_results_table.palette().color(QPalette.ColorRole.Base)
        alt_bg_color = default_bg_color.darker(105) if row % 2 == 1 else default_bg_color
        color_to_set = alt_bg_color # Default is alternating color

        # Set color based on the action taken on the pair
        if decision_type == 'processed' or decision_type == 'ignored':
            color_to_set = self.color_semantic_processed # Use light grey for now
        # Add other conditions if needed later

        for col in range(self.semantic_results_table.columnCount()):
            item = self.semantic_results_table.item(row, col)
            if item:
                item.setBackground(color_to_set)
                # --- Optionally disable checkbox after processing ---
                if col == 0 and decision_type in ['processed', 'ignored']:
                    # Keep it checkable for now, user might want to re-process?
                    # item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Unchecked) # Ensure unchecked after processing


    # --- Methods for Save/Load/Apply (Currently only handle MD5 decisions) ---
    # (No changes needed here for now)
    def save_decisions(self):
        """Saves the current MD5 block decisions to a JSON file."""
        print("\n--- '保存决策'按钮点击 ---");
        # ... (rest of save logic for self.block_decisions) ...
        if not self.block_decisions: QMessageBox.information(self, "无决策", "没有可保存的决策。"); return
        data_to_save = []
        for decision_key, decision in self.block_decisions.items():
            if decision != 'undecided':
                try: str_path, index_str, type_str = decision_key.split(self.DECISION_KEY_SEPARATOR); data_to_save.append({"file": str_path, "index": int(index_str), "type": type_str, "decision": decision})
                except ValueError: print(f"[错误] 无法解析决策键: {decision_key}"); continue
        if not data_to_save: QMessageBox.information(self, "无已标记决策", "没有已标记为 '保留' 或 '删除' 的决策可供保存。"); return
        default_folder = self.folder_path_edit.text() or os.path.expanduser("~"); suggested_filename = os.path.join(default_folder, "kd_decisions.json"); file_path, _ = QFileDialog.getSaveFileName(self, "保存决策文件", suggested_filename, "JSON Files (*.json);;All Files (*)")
        if not file_path: print("用户取消保存。"); return
        try:
            with open(file_path, 'w', encoding='utf-8') as f: json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            print(f"决策已成功保存到: {file_path}"); QMessageBox.information(self, "保存成功", f"决策已成功保存到:\n{file_path}")
        except Exception as e: print(f"[错误] 保存文件时出错: {e}"); QMessageBox.critical(self, "保存失败", f"保存文件时出错:\n{e}")

    def load_decisions(self):
        """Loads MD5 block decisions from JSON and updates the MD5 table visuals."""
        # (No changes needed here for now)
        print("\n--- '加载决策'按钮点击 ---");
        # ... (rest of load logic for self.block_decisions and updating MD5 table) ...
        if not self.row_to_block_info: QMessageBox.warning(self, "无分析结果", "请先运行分析并填充 MD5 表格后，再加载决策。"); return
        default_folder = self.folder_path_edit.text() or os.path.expanduser("~"); file_path, _ = QFileDialog.getOpenFileName(self, "加载决策文件", default_folder, "JSON Files (*.json);;All Files (*)")
        if not file_path: print("用户取消加载。"); return
        try:
            print(f"尝试从以下文件加载决策: {file_path}");
            with open(file_path, 'r', encoding='utf-8') as f: loaded_data = json.load(f)
            print(f"成功读取 {len(loaded_data)} 条决策记录。")
        except Exception as e: print(f"[错误] 读取或解析JSON时出错: {e}"); QMessageBox.critical(self, "加载失败", f"读取或解析JSON文件时出错:\n{e}"); return
        loaded_decision_map = {}
        for item in loaded_data:
            if isinstance(item, dict) and all(k in item for k in ["file", "index", "type", "decision"]):
                 key = self._create_decision_key(item['file'], item['index'], item['type']); loaded_decision_map[key] = item['decision']
            else: print(f"[警告] 加载的决策文件中发现格式不正确的条目: {item}")
        print(f"将 {len(loaded_decision_map)} 条有效决策应用到当前分析结果..."); applied_count = 0
        current_keys_in_memory = list(self.block_decisions.keys())
        for key in current_keys_in_memory: self.block_decisions[key] = 'undecided'
        for key, loaded_decision in loaded_decision_map.items():
             if key in self.block_decisions:
                 if loaded_decision in ['keep', 'delete']: self.block_decisions[key] = loaded_decision; applied_count += 1
                 else: print(f"[警告] 加载的决策值无效 '{loaded_decision}' for {key}, 设为 'undecided'.")
             else: print(f"[警告] 加载的决策与当前分析结果不匹配，已忽略: {key}")
        print(f"成功应用 {applied_count} 条加载的决策。"); print("正在刷新 MD5 表格视觉效果以匹配加载的决策...")
        self.md5_results_table.setSortingEnabled(False)
        for row, block_info in self.row_to_block_info.items():
             key = self._create_decision_key(block_info[0], block_info[1], block_info[2]); decision = self.block_decisions.get(key, 'undecided'); self._update_md5_row_visuals(row, decision)
             item = self.md5_results_table.item(row, 0);
             if item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable: item.setCheckState(Qt.CheckState.Unchecked)
        self.md5_results_table.setSortingEnabled(True); print("MD5 表格视觉效果刷新完成。")
        QMessageBox.information(self, "加载完成", f"已成功加载并应用 {applied_count} 条决策。\nMD5 表格视觉效果已更新。")

    def apply_decisions(self):
        """Applies MD5 decisions by generating new files containing 'kept' blocks."""
        # (No changes needed here for now)
        print("\n--- '应用决策'按钮点击 ---");
        # ... (rest of apply logic using self.block_decisions) ...
        if not self.blocks_data: QMessageBox.warning(self, "无分析结果", "请先运行分析，再应用决策。"); return
        output_dir = QFileDialog.getExistingDirectory(self, "选择保存去重后文件的目标文件夹")
        if not output_dir: print("用户取消选择目标文件夹。"); return
        output_dir_path = Path(output_dir); print(f"将把去重后的文件保存到: {output_dir_path}")
        if not hasattr(self, 'processed_files_in_run') or not self.processed_files_in_run:
             processed_files = {info[0] for info in self.blocks_data}
             if not processed_files: QMessageBox.critical(self, "错误", "无法确定处理了哪些文件，请重新运行分析。"); return
             self.processed_files_in_run = processed_files; print("[警告] 未找到运行时的文件列表，从当前块数据推断。")
        processed_files_count = 0; error_files = []
        for original_file_path in self.processed_files_in_run:
            blocks_for_this_file = sorted([b for b in self.blocks_data if b[0] == original_file_path], key=lambda x: x[1])
            if not blocks_for_this_file: print(f"文件 {original_file_path.name} 没有提取到内容块，跳过生成。"); continue
            new_content_parts = []
            for block_info in blocks_for_this_file:
                decision_key = self._create_decision_key(block_info[0], block_info[1], block_info[2]); decision = self.block_decisions.get(decision_key, 'keep') # Default to keep undecided
                if decision != 'delete':
                    file_path, b_index, b_type, b_text = block_info; formatted_text = b_text
                    if b_type == 'heading': formatted_text = f"## {b_text}" # Excluded, but keep logic
                    elif b_type == 'block_code': formatted_text = f"```\n{b_text}\n```" if not (b_text.startswith("```") and b_text.endswith("```")) else b_text
                    elif b_type == 'list_item': formatted_text = f"- {b_text}"
                    elif b_type == 'block_quote': lines = b_text.splitlines(); formatted_text = "\n".join([f"> {line}" for line in lines])
                    new_content_parts.append(formatted_text)
            new_content = "\n\n".join(new_content_parts); output_filename = f"{original_file_path.stem}_deduped{original_file_path.suffix}"; output_filepath = output_dir_path / output_filename
            try: output_filepath.write_text(new_content, encoding='utf-8'); print(f"已生成新文件: {output_filepath}"); processed_files_count += 1
            except Exception as e: print(f"[错误] 写入文件 '{output_filepath.name}' 时出错: {e}"); error_files.append(original_file_path.name)
        if not error_files: QMessageBox.information(self, "应用完成", f"已成功为 {processed_files_count} 个文件生成去重后的版本（包含基本格式），保存在:\n{output_dir_path}")
        else: QMessageBox.warning(self, "部分完成", f"为 {processed_files_count} 个文件生成了新版本，但以下文件处理失败:\n" + "\n".join(error_files) + f"\n\n请检查目标文件夹权限或错误日志。\n输出目录: {output_dir_path}")
        print("--- 应用决策完成 ---")


    # --- RESTORED HELPER METHODS for Semantic Analysis ---
    # (No changes needed here)
    def _calculate_embeddings(self, texts):
        if self.semantic_model and texts:
            try: embeddings = self.semantic_model.encode(texts, convert_to_tensor=True, show_progress_bar=False); return embeddings
            except Exception as e: print(f"[错误] 计算向量嵌入时出错: {e}"); return None
        return None
    def _find_similar_pairs(self, embeddings, blocks_info, threshold):
        # --- UPDATED to log skipped pairs ---
        similar_pairs = [];
        if embeddings is None or len(embeddings) < 2: return similar_pairs
        try:
            hits = util.semantic_search(embeddings, embeddings, query_chunk_size=100, corpus_chunk_size=500, top_k=len(embeddings), score_function=util.cos_sim); processed_pairs = set()
            block_hashes = {info: hashlib.md5(info[3].encode('utf-8')).hexdigest() for info in blocks_info}
            md5_skipped_count = 0 # Counter for skipped pairs
            for query_idx in range(len(hits)):
                query_block_info = blocks_info[query_idx]; query_hash = block_hashes.get(query_block_info)
                for hit in hits[query_idx]:
                    corpus_idx = hit['corpus_id']; score = hit['score']
                    if query_idx == corpus_idx or score < threshold: continue
                    corpus_block_info = blocks_info[corpus_idx]; corpus_hash = block_hashes.get(corpus_block_info)
                    # --- Skip if MD5 hashes match (exact duplicate) ---
                    if query_hash is not None and corpus_hash is not None and query_hash == corpus_hash:
                        pair_check = tuple(sorted((query_idx, corpus_idx)))
                        if pair_check not in processed_pairs:
                            md5_skipped_count += 1
                            processed_pairs.add(pair_check)
                        continue
                    # --------------------------------------------------
                    pair = tuple(sorted((query_idx, corpus_idx)))
                    if pair not in processed_pairs: similar_pairs.append((query_block_info, corpus_block_info, score)); processed_pairs.add(pair)
            if md5_skipped_count > 0:
                 print(f"在语义分析中跳过了 {md5_skipped_count} 对 MD5 完全相同的块。") # Log skipped count
        except Exception as e: print(f"[错误] 计算或筛选相似对时出错: {e}")
        return similar_pairs
    # ----------------------------------------------------

    def _extract_text_from_children(self, children):
        """Helper function to recursively extract text from mistune token children."""
        # (No changes needed here)
        text = "";
        if children is None: return ""
        for child in children:
            child_type = child.get('type')
            if child_type == 'text': text += child.get('raw', '')
            elif child_type == 'codespan': text += child.get('raw', '')
            elif child_type in ['link', 'image', 'emphasis', 'strong', 'strikethrough']: text += self._extract_text_from_children(child.get('children', []))
            elif child_type == 'softbreak' or child_type == 'linebreak': text += ' '
            elif child_type == 'inline_html': pass
        return text.strip()

    # --- UPDATED Slot for Semantic Table Double Click ---
    def show_semantic_detail(self, item):
        """Shows a dialog with the full text of the double-clicked semantic pair, highlighting the blocks."""
        # (No changes needed here)
        if item is None: return
        row = item.row()
        pair_info = self.semantic_row_to_pair_info.get(row)
        if pair_info:
            info1, info2, score = pair_info; file_path1 = info1[0]; file_path2 = info2[0]
            full_text1 = self.markdown_files_content.get(file_path1); full_text2 = self.markdown_files_content.get(file_path2)
            if full_text1 is None or full_text2 is None: QMessageBox.warning(self, "错误", "无法获取原始文件内容以显示详情。"); return
            dialog = SemanticDetailDialog(score, info1, info2, full_text1, full_text2, self)
            dialog.highlight_and_scroll(); dialog.exec()
        else: print(f"[警告] 无法找到语义表格第 {row} 行对应的详细信息。")


# --- 程序入口 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KDToolWindow()
    window.show()
    sys.exit(app.exec())

