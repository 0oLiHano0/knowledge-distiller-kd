# KD_Tool_CLI/knowledge_distiller_kd/core/document_processor.py
"""
文档预处理模块。

此模块负责将不同格式的文档转换为标准化的内容块(ContentBlock)。
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Union
import logging
import traceback # 确保导入 traceback

# 确保导入了 partition 和 partition_md
from unstructured.partition.auto import partition
from unstructured.partition.md import partition_md
from unstructured.documents.elements import (
    Element, Title, NarrativeText, ListItem, CodeSnippet, Table,
    Text
)

from knowledge_distiller_kd.core.error_handler import KDError
from knowledge_distiller_kd.core import constants # 导入常量

# 配置日志
logger = logging.getLogger(constants.LOGGER_NAME) # 使用常量中定义的名字

class DocumentProcessingError(KDError):
    """文档处理相关错误"""
    pass

class ContentBlock:
    """内容块类，封装Unstructured的Element对象，并提供标准化文本"""

    # 定义元素类型映射
    ELEMENT_TYPE_MAP = {
        Title: "Title",
        NarrativeText: "NarrativeText",
        ListItem: "ListItem",
        CodeSnippet: "CodeSnippet",
        Table: "Table",
        Text: "Text"
    }

    def __init__(self, element: Element, file_path: str):
        """初始化ContentBlock"""
        if not isinstance(element, Element):
            raise TypeError("element must be an instance of unstructured.documents.elements.Element")

        self.element: Element = element
        self.file_path: str = file_path
        self.block_id: str = str(element.id)
        self.original_text: str = element.text # 存储原始文本
        self._infer_block_type()
        self.analysis_text: str = self._normalize_text()

    def _infer_block_type(self) -> None:
        """推断内容块的类型，并在必要时修改元素类型。"""
        # 使用 original_text 进行推断
        text = self.original_text.strip()
        current_type = type(self.element)

        if current_type == CodeSnippet: return # unstructured 识别优先

        # 优先检查代码块碎片
        if re.fullmatch(r'```\s*', text):
            logger.debug(f"Inferring block type for {self.block_id} as CodeSnippet (end fence).")
            self.element.__class__ = CodeSnippet; return
        if re.match(r'^```(\s*\w*)?', text):
             logger.debug(f"Inferring block type for {self.block_id} as CodeSnippet (start fence).")
             self.element.__class__ = CodeSnippet; return

        # 再检查其他类型
        if re.match(r'^#{1,6}\s+', text):
            logger.debug(f"Inferring block type for {self.block_id} as Title based on content.")
            self.element.__class__ = Title; return
        if re.match(r'^\s*([-*+]|\d+\.)\s+', text):
            logger.debug(f"Inferring block type for {self.block_id} as ListItem based on content.")
            self.element.__class__ = ListItem; return

        # 最后处理 Text/NarrativeText
        if current_type in [Text, NarrativeText]:
            if self._looks_like_narrative():
                 logger.debug(f"Inferring block type for {self.block_id} as NarrativeText based on _looks_like_narrative.")
                 self.element.__class__ = NarrativeText
            else:
                 logger.debug(f"Inferring block type for {self.block_id} remains Text as it doesn't look narrative.")
                 self.element.__class__ = Text

    def _normalize_text(self) -> str:
        """生成用于分析的标准化文本。"""
        # 使用 self.original_text 作为标准化的输入源
        text = self.original_text

        # 特殊类型处理 (基于 self.element 的当前类型，可能已被 _infer_block_type 修改)
        if isinstance(self.element, CodeSnippet):
            logger.debug(f"Normalizing CodeSnippet {self.block_id}: Keeping original text, stripping outer whitespace.")
            return text.strip()
        elif isinstance(self.element, Table):
            logger.debug(f"Normalizing Table {self.block_id}: Keeping original text.")
            return text.strip()
        elif isinstance(self.element, Title):
            # ==================== 添加日志：检查输入 ====================
            logger.debug(f"Normalizing Title {self.block_id} - Input text: '{text[:50]}...'")
            # ==========================================================
            # 移除开头的 # 标记并 strip
            normalized = re.sub(r'^#{1,6}\s+', '', text).strip()
            logger.debug(f"Normalizing Title {self.block_id}: Result -> '{normalized[:50]}...'")
            return normalized

        # 通用处理 (适用于 NarrativeText, ListItem, Text 等)
        logger.debug(f"Normalizing generic block {self.block_id} (type: {type(self.element).__name__}). Original: '{text[:50]}...'")
        # 移除Markdown格式标记
        text = re.sub(r'`', '', text)
        text = re.sub(r'(?<!\\)[*_~]{1,2}(.+?)(?<!\\)[*_~]{1,2}', r'\1', text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)
        text = re.sub(r'^\s*([-*+]|\d+\.)\s+', '', text)
        text = re.sub(r'^\s*>\s?', '', text, flags=re.MULTILINE)

        # 规范化空白字符
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            line = ' '.join(line.split()) # 行内多空格变单空格
            if line:
                normalized_lines.append(line)
        text = ' '.join(normalized_lines) # 用空格连接处理过的行

        final_text = text.strip() # 移除首尾空白
        logger.debug(f"Normalization result for {self.block_id}: '{final_text[:50]}...'")
        return final_text

    @property
    def block_type(self) -> str:
        """获取块类型"""
        element_type = type(self.element)
        return self.ELEMENT_TYPE_MAP.get(element_type, type(self.element).__name__)

    def _looks_like_narrative(self) -> bool:
        """判断文本是否看起来像普通叙述文本"""
        text = self.original_text.strip()
        if not text: return False
        has_content = bool(re.search(r'[a-zA-Z\u4e00-\u9fff]', text))
        not_code_fence = not re.fullmatch(r'```\s*.*', text)
        not_simple_markup = not re.match(r'^(#+\s*|[-*+]\s*|>\s*)$', text)
        return has_content and not_simple_markup and not_code_fence

    @property
    def metadata(self) -> Dict:
        """获取元数据"""
        return {
            'file_path': self.file_path,
            'block_id': self.block_id,
            'block_type': self.block_type,
            'original_text_preview': self.original_text[:100] + ('...' if len(self.original_text) > 100 else ''),
            'analysis_text_preview': self.analysis_text[:100] + ('...' if len(self.analysis_text) > 100 else ''),
            'element_metadata': self.element.metadata.to_dict() if hasattr(self.element, 'metadata') else {}
        }

    def __repr__(self) -> str:
        """提供字符串表示"""
        return f"ContentBlock(id={self.block_id}, type={self.block_type}, file='{Path(self.file_path).name}')"

# --- process_file 和 process_directory 函数 ---
def process_file(file_path: Union[str, Path]) -> List[ContentBlock]:
    """处理单个文件，将其转换为ContentBlock列表。"""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        logger.info(f"Processing file: {file_path}")
        elements = []
        if file_path.suffix.lower() in ['.md', '.markdown']:
            # 使用 unstructured 解析 Markdown 文件
            elements = partition_md(filename=str(file_path))
        else:
            logger.warning(f"Skipping non-markdown file: {file_path}")
            return []
        if not elements:
            logger.warning(f"No elements extracted from file: {file_path}")
            return []
        blocks = []
        for element in elements:
            try:
                # 将解析出的每个 element 包装成 ContentBlock
                block = ContentBlock(element, str(file_path))
                blocks.append(block)
            except TypeError as e:
                # 处理无效 element 的情况
                logger.warning(f"Skipping invalid element in {file_path}: {e}")
                continue
            except Exception as e_init:
                # 处理 ContentBlock 初始化时的其他错误
                logger.error(f"Error initializing ContentBlock for element {getattr(element, 'id', 'N/A')} in {file_path}: {e_init}", exc_info=True)
                continue
        logger.info(f"Successfully processed {len(blocks)} blocks from {file_path}")
        return blocks
    except FileNotFoundError as e:
        # 文件未找到错误
        logger.error(f"File not found during processing: {file_path}")
        raise DocumentProcessingError(f"File not found: {file_path}", error_code="FILE_NOT_FOUND") from e
    except Exception as e:
        # 其他处理错误
        logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
        raise DocumentProcessingError(
            f"Error processing file {file_path}: {str(e)}",
            error_code="FILE_PROCESSING_ERROR",
            details={"error": str(e), "traceback": traceback.format_exc()}
        )

def process_directory(dir_path: Union[str, Path], recursive: bool = True) -> Dict[Path, List[ContentBlock]]:
    """处理目录中的所有Markdown文件。"""
    try:
        dir_path = Path(dir_path)
        if not dir_path.exists(): raise FileNotFoundError(f"Directory not found: {dir_path}")
        if not dir_path.is_dir(): raise NotADirectoryError(f"Not a directory: {dir_path}")
        logger.info(f"Processing directory: {dir_path}")
        results = {}
        markdown_extensions = constants.SUPPORTED_EXTENSIONS # 使用常量
        pattern = "**/*" if recursive else "*"
        # 查找所有匹配扩展名的文件
        files_to_process = [fp for fp in dir_path.glob(pattern) if fp.is_file() and fp.suffix.lower() in markdown_extensions]
        if not files_to_process:
            logger.warning(f"No Markdown files found in directory: {dir_path}"); return {}
        processed_count = 0
        # 遍历并处理每个文件
        for file_path in files_to_process:
            try:
                blocks = process_file(file_path)
                if blocks: # 只添加处理后包含块的文件
                    results[file_path] = blocks; processed_count += 1
            except DocumentProcessingError as e:
                # 记录单个文件处理错误，但继续处理其他文件
                logger.error(f"Error processing file {file_path} within directory: {e}"); continue
            except Exception as e_single:
                # 捕获未预期的错误
                logger.error(f"Unexpected error processing file {file_path} within directory: {e_single}", exc_info=True); continue
        logger.info(f"Successfully processed {processed_count} Markdown files with content in {dir_path}")
        return results
    except (FileNotFoundError, NotADirectoryError) as e:
        # 目录路径无效错误
        logger.error(f"Invalid directory path: {dir_path} - {e}")
        raise DocumentProcessingError(str(e), error_code="INVALID_DIRECTORY") from e
    except Exception as e:
        # 其他目录处理错误
        logger.error(f"Error processing directory {dir_path}: {str(e)}", exc_info=True)
        raise DocumentProcessingError(
            f"Error processing directory {dir_path}: {str(e)}",
            error_code="DIRECTORY_PROCESSING_ERROR",
            details={"error": str(e), "traceback": traceback.format_exc()}
        )
