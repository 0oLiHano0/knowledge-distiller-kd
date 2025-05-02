"""
文档预处理模块与KDToolCLI的集成测试。

此模块包含对文档预处理模块与KDToolCLI集成的测试用例。
"""

import pytest
import json
import os
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any

from knowledge_distiller_kd.core.kd_tool_CLI import KDToolCLI
from knowledge_distiller_kd.core.document_processor import ContentBlock, process_file, process_directory
from knowledge_distiller_kd.core import constants
from tests.test_data_generator import DataGenerator
from tests.test_utils import (
    verify_file_content,
    verify_decision_file,
    cleanup_test_environment
)

# 测试数据生成器
test_data_generator = DataGenerator()

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """创建临时输出目录"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir

@pytest.fixture
def temp_decision_file(tmp_path: Path) -> Path:
    """创建临时决策文件"""
    decision_dir = tmp_path / "decisions"
    decision_dir.mkdir()
    return decision_dir / "decisions.json"

@pytest.fixture
def kd_tool(temp_output_dir: Path, temp_decision_file: Path) -> KDToolCLI:
    """创建KDToolCLI实例"""
    return KDToolCLI(
        output_dir=str(temp_output_dir),
        decision_file=str(temp_decision_file)
    )

@pytest.fixture
def test_md_file(tmp_path: Path) -> Path:
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
    file_path = tmp_path / "test.md"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path

@pytest.fixture
def test_dir(tmp_path: Path, test_md_file: Path) -> Path:
    """创建测试目录，包含多个Markdown文件"""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    
    # 复制测试文件到目录中
    shutil.copy(test_md_file, test_dir / "test1.md")
    shutil.copy(test_md_file, test_dir / "test2.md")
    
    return test_dir

class TestDocumentProcessorIntegration:
    """测试文档预处理模块与KDToolCLI的集成"""

    def test_integration_with_single_file(self, kd_tool: KDToolCLI, test_md_file: Path):
        """测试单个文件处理的集成"""
        # 设置输入目录
        kd_tool.set_input_dir(str(test_md_file.parent))
        
        # 运行分析
        kd_tool.run_analysis()
        
        # 验证结果
        assert len(kd_tool.blocks_data) > 0
        assert all(isinstance(block, ContentBlock) for block in kd_tool.blocks_data)
        
        # 验证内容块属性
        first_block = kd_tool.blocks_data[0]
        assert first_block.file_path == str(test_md_file)
        assert "测试文档" in first_block.analysis_text
        assert first_block.block_type in ["NarrativeText", "Title"]

    def test_integration_with_directory(self, kd_tool: KDToolCLI, test_dir: Path):
        """测试目录处理的集成"""
        # 设置输入目录
        kd_tool.set_input_dir(str(test_dir))
        
        # 运行分析
        kd_tool.run_analysis()
        
        # 验证结果
        assert len(kd_tool.blocks_data) > 0
        assert all(isinstance(block, ContentBlock) for block in kd_tool.blocks_data)
        
        # 验证文件数量
        processed_files = {block.file_path for block in kd_tool.blocks_data}
        assert len(processed_files) == 2  # test1.md 和 test2.md
        
        # 验证每个文件的内容块
        for file_path in processed_files:
            file_blocks = [block for block in kd_tool.blocks_data if block.file_path == file_path]
            assert len(file_blocks) > 0
            assert all("测试文档" in block.analysis_text for block in file_blocks)

    def test_integration_with_empty_directory(self, kd_tool: KDToolCLI, tmp_path: Path):
        """测试空目录处理的集成"""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        
        # 设置输入目录
        kd_tool.set_input_dir(str(empty_dir))
        
        # 运行分析
        kd_tool.run_analysis()
        
        # 验证结果
        assert len(kd_tool.blocks_data) == 0

    def test_integration_with_invalid_file(self, kd_tool: KDToolCLI, tmp_path: Path):
        """测试无效文件处理的集成"""
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("This is not a markdown file")
        
        # 设置输入目录
        kd_tool.set_input_dir(str(tmp_path))
        
        # 运行分析
        kd_tool.run_analysis()
        
        # 验证结果
        assert len(kd_tool.blocks_data) == 0

    def test_integration_with_mixed_content(self, kd_tool: KDToolCLI, tmp_path: Path):
        """测试混合内容处理的集成"""
        # 创建包含不同类型内容的文件
        mixed_file = tmp_path / "mixed.md"
        content = """# 标题

这是普通文本。

```python
def test():
    print("Hello")
```

## 子标题

这是另一个段落。

- 列表项1
- 列表项2
"""
        mixed_file.write_text(content)
        
        # 设置输入目录
        kd_tool.set_input_dir(str(tmp_path))
        
        # 运行分析
        kd_tool.run_analysis()
        
        # 验证结果
        assert len(kd_tool.blocks_data) > 0
        assert all(isinstance(block, ContentBlock) for block in kd_tool.blocks_data)
        
        # 验证不同类型的块
        block_types = {block.block_type for block in kd_tool.blocks_data}
        assert "Title" in block_types
        assert "NarrativeText" in block_types
        assert "CodeSnippet" in block_types
        assert "ListItem" in block_types 