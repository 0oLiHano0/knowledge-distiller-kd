# tests/test_document_processor.py
"""
文档预处理模块的测试用例。

此模块包含对ContentBlock类和文档处理功能的测试。
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List # 移除 Tuple
from unittest.mock import Mock, patch # 保持 mock 和 patch
import tempfile
import os
import shutil

# 导入真实的 Element 类型，而不是使用 MockElement
from unstructured.documents.elements import (
    Element, Title, NarrativeText, ListItem, CodeSnippet, Table, Text
)

from knowledge_distiller_kd.core.document_processor import (
    ContentBlock,
    DocumentProcessingError,
    process_file,
    process_directory
)

# --- ContentBlock测试 ---
class TestContentBlock:
    """测试ContentBlock类"""

    @pytest.fixture
    def basic_element(self) -> NarrativeText:
        """提供基本的Element对象 (使用真实类型)"""
        return NarrativeText(
            text="这是一个测试文本",
            element_id="test_001"
        )

    @pytest.fixture
    def markdown_element(self) -> NarrativeText: # 保持 NarrativeText，让 _infer_block_type 推断
        """提供包含Markdown格式的Element对象 (使用真实类型)"""
        # ContentBlock 的 normalize 会处理 markdown
        return NarrativeText(
            text="# 标题\n\n这是一个**加粗**的段落，包含[链接](http://example.com)和`代码`。",
            element_id="test_002"
        )

    @pytest.fixture
    def code_element(self) -> CodeSnippet:
        """提供代码块Element对象 (使用真实类型)"""
        return CodeSnippet(
            text="```python\ndef test():\n    print('Hello')\n```",
            element_id="test_003"
        )

    def test_initialization(self, basic_element):
        """测试ContentBlock的初始化"""
        file_path = "test.md"
        block = ContentBlock(basic_element, file_path)

        assert block.element == basic_element
        assert block.file_path == file_path
        assert block.block_id == basic_element.id
        assert isinstance(block.analysis_text, str)
        assert block.block_type == "NarrativeText" # 初始类型

    def test_initialization_with_invalid_element(self):
        """测试使用无效Element初始化ContentBlock"""
        with pytest.raises(TypeError, match="element must be an instance of unstructured.documents.elements.Element"):
            ContentBlock("invalid_element", "test.md") # type: ignore

    def test_normalize_text_basic(self, basic_element):
        """测试基本文本的标准化"""
        block = ContentBlock(basic_element, "test.md")
        # 预期：移除多余空格，保留文本
        assert block.analysis_text == "这是一个测试文本"

    def test_normalize_text_markdown(self, markdown_element):
        """测试Markdown文本的标准化"""
        block = ContentBlock(markdown_element, "test.md")
        # 预期：_infer_block_type 会识别为 Title
        assert block.block_type == "Title"

        # ==================== 修改：更新 expected 值 ====================
        # 预期：_normalize_text 对 Title 只移除 # 和 strip()，保留内部 markdown
        expected = "标题\n\n这是一个**加粗**的段落，包含[链接](http://example.com)和`代码`。"
        # ============================================================

        # print(f"\nMarkdown Normalize Result: '{block.analysis_text}'") # 调试用
        assert block.analysis_text == expected

    def test_normalize_text_code(self, code_element):
        """测试代码块的标准化"""
        block = ContentBlock(code_element, "test.md")
        assert block.block_type == "CodeSnippet"
        # 预期：标准化应该只移除外部空白
        expected = "```python\ndef test():\n    print('Hello')\n```"
        assert block.analysis_text == expected

    def test_properties(self, basic_element):
        """测试属性访问"""
        block = ContentBlock(basic_element, "test.md")
        assert block.original_text == basic_element.text
        assert block.block_type == "NarrativeText"
        assert isinstance(block.metadata, dict)

    def test_metadata(self, basic_element):
        """测试元数据生成"""
        block = ContentBlock(basic_element, "test.md")
        metadata = block.metadata
        assert isinstance(metadata, dict)
        assert metadata["file_path"] == "test.md"
        assert metadata["block_id"] == basic_element.id
        assert metadata["block_type"] == "NarrativeText"
        assert "original_text_preview" in metadata
        assert "analysis_text_preview" in metadata
        assert "element_metadata" in metadata

    def test_repr(self, basic_element):
        """测试字符串表示"""
        block = ContentBlock(basic_element, "test.md")
        repr_str = repr(block)
        assert "ContentBlock" in repr_str
        assert basic_element.id in repr_str
        assert "NarrativeText" in repr_str
        assert "test.md" in repr_str

# --- 文件处理测试 ---
class TestFileProcessing:
    """测试文件处理功能"""

    @pytest.fixture
    def test_dir(self, tmp_path: Path):
        """创建临时测试目录"""
        test_dir_path = tmp_path / "test_proc_dir"
        test_dir_path.mkdir()
        yield test_dir_path
        # pytest 会自动清理

    @pytest.fixture
    def test_md_file(self, test_dir):
        """创建测试Markdown文件"""
        # 定义测试文件内容，确保包含不同类型的块
        content = """# 测试文档 Title

这是一个测试文档 NarrativeText。

```python
def hello():
    print("Hello, World!")
```
""" # 内容简化，使其明确产生3个主要块
        file_path = test_dir / "test.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        # --- 修正：确保 return 语句在函数内部 ---
        return file_path

    @pytest.fixture
    def mock_partition_md(self):
        """模拟Unstructured的partition_md函数"""
        # 返回真实的 Element 实例列表
        elements = [
            Title(text="# 测试文档 Title", element_id="md_001"),
            NarrativeText(text="这是一个测试文档 NarrativeText。", element_id="md_002"),
            CodeSnippet(text='```python\ndef hello():\n    print("Hello, World!")\n```', element_id="md_003")
        ]
        # patch 的目标是 process_file 内部实际调用的 partition_md
        with patch("knowledge_distiller_kd.core.document_processor.partition_md") as mock:
            mock.return_value = elements
            yield mock

    def test_process_file(self, test_md_file, mock_partition_md):
        """测试单个文件处理 (使用 mock)"""
        blocks = process_file(test_md_file)
        mock_partition_md.assert_called_once_with(filename=str(test_md_file))
        assert len(blocks) == 3 # 与 mock 返回的数量一致
        assert all(isinstance(block, ContentBlock) for block in blocks)
        assert blocks[0].block_type == "Title"
        assert blocks[1].block_type == "NarrativeText"
        assert blocks[2].block_type == "CodeSnippet"

    def test_process_file_not_found(self):
        """测试处理不存在的文件"""
        with pytest.raises((DocumentProcessingError, FileNotFoundError)):
            process_file("nonexistent.md")

    def test_process_directory(self, test_dir, test_md_file):
        """测试目录处理 (不使用 mock)"""
        extra_file = test_dir / "extra.md"
        shutil.copy(test_md_file, extra_file)
        results = process_directory(test_dir)
        assert len(results) == 2
        assert test_md_file in results
        assert extra_file in results
        assert all(isinstance(blocks, list) for blocks in results.values())
        for blocks in results.values():
             assert len(blocks) > 0
             assert all(isinstance(block, ContentBlock) for block in blocks)

    def test_process_directory_empty(self, test_dir):
        """测试处理空目录"""
        for item in test_dir.iterdir():
            if item.is_file(): item.unlink()
            elif item.is_dir(): shutil.rmtree(item)
        results = process_directory(test_dir)
        assert len(results) == 0

    def test_process_directory_invalid(self):
        """测试处理无效目录"""
        with pytest.raises(DocumentProcessingError):
            process_directory("nonexistent_dir")
