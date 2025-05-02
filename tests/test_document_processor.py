"""
文档预处理模块的测试用例。

此模块包含对ContentBlock类和文档处理功能的测试。
"""

import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch
import tempfile
import os
import shutil

from knowledge_distiller_kd.core.document_processor import (
    ContentBlock, 
    DocumentProcessingError,
    process_file,
    process_directory
)

# --- 测试数据 ---
class MockElement:
    """模拟Unstructured的Element类"""
    def __init__(self, text: str, element_type: str, element_id: str = "test_id"):
        self.text = text
        self.type = element_type
        self.id = element_id

# --- ContentBlock测试 ---
class TestContentBlock:
    """测试ContentBlock类"""

    @pytest.fixture
    def basic_element(self) -> MockElement:
        """提供基本的Element对象"""
        return MockElement(
            text="这是一个测试文本",
            element_type="NarrativeText",
            element_id="test_001"
        )

    @pytest.fixture
    def markdown_element(self) -> MockElement:
        """提供包含Markdown格式的Element对象"""
        return MockElement(
            text="# 标题\n\n这是一个**加粗**的段落，包含[链接](http://example.com)和`代码`。",
            element_type="NarrativeText",
            element_id="test_002"
        )

    @pytest.fixture
    def code_element(self) -> MockElement:
        """提供代码块Element对象"""
        return MockElement(
            text="```python\ndef test():\n    print('Hello')\n```",
            element_type="CodeSnippet",
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

    def test_initialization_with_invalid_element(self):
        """测试使用无效Element初始化ContentBlock"""
        with pytest.raises(TypeError):
            ContentBlock("invalid_element", "test.md")

    def test_normalize_text_basic(self, basic_element):
        """测试基本文本的标准化"""
        block = ContentBlock(basic_element, "test.md")
        assert block.analysis_text == "这是一个测试文本"

    def test_normalize_text_markdown(self, markdown_element):
        """测试Markdown文本的标准化"""
        block = ContentBlock(markdown_element, "test.md")
        expected = "标题 这是一个加粗的段落，包含链接和代码。"
        assert block.analysis_text == expected

    def test_normalize_text_code(self, code_element):
        """测试代码块的标准化"""
        block = ContentBlock(code_element, "test.md")
        expected = "```python\ndef test():\n    print('Hello')\n```"
        assert block.analysis_text == expected

    def test_properties(self, basic_element):
        """测试属性访问"""
        block = ContentBlock(basic_element, "test.md")
        
        assert block.original_text == basic_element.text
        assert block.block_type == basic_element.type
        assert isinstance(block.metadata, dict)

    def test_metadata(self, basic_element):
        """测试元数据生成"""
        block = ContentBlock(basic_element, "test.md")
        metadata = block.metadata

        assert isinstance(metadata, dict)
        assert metadata["file_path"] == "test.md"
        assert metadata["block_id"] == basic_element.id
        assert metadata["block_type"] == basic_element.type
        assert metadata["original_text"] == basic_element.text
        assert metadata["analysis_text"] == block.analysis_text

    def test_repr(self, basic_element):
        """测试字符串表示"""
        block = ContentBlock(basic_element, "test.md")
        repr_str = repr(block)
        
        assert "ContentBlock" in repr_str
        assert basic_element.id in repr_str
        assert basic_element.type in repr_str
        assert "test.md" in repr_str

# --- 文件处理测试 ---
class TestFileProcessing:
    """测试文件处理功能"""

    @pytest.fixture
    def test_dir(self):
        """创建临时测试目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_md_file(self, test_dir):
        """创建测试Markdown文件"""
        content = """# 测试文档

这是一个测试文档。

## 第二部分

这是第二部分的内容。

```python
def hello():
    print("Hello, World!")
```
"""
        file_path = Path(test_dir) / "test.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path

    @pytest.fixture
    def mock_partition(self):
        """模拟Unstructured的partition函数"""
        elements = [
            MockElement("这是一个测试文档。", "NarrativeText", "001"),
            MockElement("这是第二部分的内容。", "NarrativeText", "002"),
            MockElement("```python\ndef hello():\n    print('Hello, World!')\n```", "CodeSnippet", "003")
        ]
        with patch("knowledge_distiller_kd.core.document_processor.partition") as mock:
            mock.return_value = elements
            yield mock

    def test_process_file(self, test_md_file, mock_partition):
        """测试单个文件处理"""
        blocks = process_file(test_md_file)
        
        assert len(blocks) == 3
        assert all(isinstance(block, ContentBlock) for block in blocks)
        assert blocks[0].block_type == "NarrativeText"
        assert blocks[2].block_type == "CodeSnippet"

    def test_process_file_not_found(self):
        """测试处理不存在的文件"""
        with pytest.raises(DocumentProcessingError):
            process_file("nonexistent.md")

    def test_process_directory(self, test_dir, test_md_file, mock_partition):
        """测试目录处理"""
        # 创建额外的测试文件
        extra_file = Path(test_dir) / "extra.md"
        shutil.copy(test_md_file, extra_file)

        results = process_directory(test_dir)
        
        assert len(results) == 2
        assert test_md_file in results
        assert extra_file in results
        assert all(isinstance(blocks, list) for blocks in results.values())
        assert all(isinstance(block, ContentBlock) for blocks in results.values() for block in blocks)

    def test_process_directory_empty(self, test_dir):
        """测试处理空目录"""
        results = process_directory(test_dir)
        assert len(results) == 0

    def test_process_directory_invalid(self):
        """测试处理无效目录"""
        with pytest.raises(DocumentProcessingError):
            process_directory("nonexistent_dir") 