"""
KDToolCLI测试模块。

此模块包含对KDToolCLI类的单元测试。
"""

import pytest
import json
import os
import shutil
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Dict, List, Tuple, Any

from knowledge_distiller_kd.core.kd_tool_CLI import KDToolCLI
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
    """
    创建一个临时输出目录。
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir

@pytest.fixture
def temp_decision_file(tmp_path: Path) -> Path:
    """
    创建一个临时决策文件。
    """
    decision_dir = tmp_path / "decisions"
    decision_dir.mkdir()
    return decision_dir / "decisions.json"

@pytest.fixture
def kd_tool(temp_output_dir: Path, temp_decision_file: Path) -> KDToolCLI:
    """
    创建一个 KDToolCLI 实例。
    """
    return KDToolCLI(
        output_dir=temp_output_dir,
        decision_file=temp_decision_file,
        skip_semantic=True
    )

def test_initialization(kd_tool: KDToolCLI, temp_output_dir: Path, temp_decision_file: Path) -> None:
    """
    测试 KDToolCLI 的初始化。
    """
    assert kd_tool.output_dir == temp_output_dir.resolve()
    assert kd_tool.decision_file == temp_decision_file.resolve()
    assert kd_tool.skip_semantic is True
    assert kd_tool.input_dir is None
    assert kd_tool.markdown_files_content == {}
    assert kd_tool.blocks_data == []
    assert kd_tool.block_decisions == {}

def test_set_input_dir(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试设置输入目录的功能。
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    result = kd_tool.set_input_dir(input_dir)
    assert result is True
    assert kd_tool.input_dir == input_dir.resolve()

def test_set_input_dir_relative_path(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试使用相对路径设置输入目录。
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    # 使用相对路径
    relative_path = Path("input")
    # 切换到临时目录
    os.chdir(tmp_path)
    result = kd_tool.set_input_dir(relative_path)
    assert result is True
    assert kd_tool.input_dir == (tmp_path / relative_path).resolve()

def test_set_input_dir_invalid(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试设置无效输入目录的情况。
    """
    invalid_dir = tmp_path / "nonexistent"
    result = kd_tool.set_input_dir(invalid_dir)
    assert result is False
    assert kd_tool.input_dir is None

@patch('kd_tool_CLI.KDToolCLI._read_files')
@patch('kd_tool_CLI.KDToolCLI._parse_markdown')
@patch('kd_tool_CLI.KDToolCLI._initialize_decisions')
def test_run_analysis(mock_init_decisions: MagicMock, mock_parse_markdown: MagicMock, 
                     mock_read_files: MagicMock, kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试运行分析的功能。
    """
    # 设置输入目录
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    kd_tool.set_input_dir(input_dir)
    
    # 设置模拟返回值
    mock_read_files.return_value = True
    mock_parse_markdown.return_value = True
    mock_init_decisions.return_value = None

    # 运行分析
    result = kd_tool.run_analysis()
    
    assert result is True
    mock_read_files.assert_called_once()
    mock_parse_markdown.assert_called_once()
    mock_init_decisions.assert_called_once()

def test_run_analysis_no_input_dir(kd_tool: KDToolCLI) -> None:
    """
    测试在没有设置输入目录的情况下运行分析。
    """
    result = kd_tool.run_analysis()
    assert result is False

@patch('kd_tool_CLI.KDToolCLI._read_files')
def test_run_analysis_read_files_failed(mock_read_files: MagicMock, kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试在读取文件失败的情况下运行分析。
    """
    # 设置输入目录
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    kd_tool.set_input_dir(input_dir)
    
    # 设置模拟返回值
    mock_read_files.return_value = False
    
    # 运行分析
    result = kd_tool.run_analysis()
    
    assert result is False
    mock_read_files.assert_called_once()

@patch('kd_tool_CLI.KDToolCLI._read_files')
@patch('kd_tool_CLI.KDToolCLI._parse_markdown')
def test_run_analysis_parse_markdown_failed(mock_parse_markdown: MagicMock, mock_read_files: MagicMock, 
                                          kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试在解析 Markdown 失败的情况下运行分析。
    """
    # 设置输入目录
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    kd_tool.set_input_dir(input_dir)
    
    # 设置模拟返回值
    mock_read_files.return_value = True
    mock_parse_markdown.return_value = False
    
    # 运行分析
    result = kd_tool.run_analysis()
    
    assert result is False
    mock_read_files.assert_called_once()
    mock_parse_markdown.assert_called_once()

def test_load_decisions_file_not_exists(kd_tool: KDToolCLI) -> None:
    """
    测试从不存在的决策文件加载决策。
    """
    kd_tool.decision_file = Path("nonexistent.json")
    result = kd_tool.load_decisions()
    assert result is False

def test_load_decisions_invalid_json(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试从无效的 JSON 文件加载决策。
    """
    # 创建一个包含无效 JSON 的文件
    decision_file = tmp_path / "invalid.json"
    decision_file.write_text("invalid json")
    kd_tool.decision_file = decision_file
    
    result = kd_tool.load_decisions()
    assert result is False

def test_load_decisions_success(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试成功加载决策。
    """
    # 设置输入目录
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    kd_tool.set_input_dir(input_dir)
    
    # 创建测试文件
    test_file = input_dir / "test.md"
    test_file.write_text("Test content")
    test_file_abs = test_file.resolve()
    
    # 创建一个包含有效决策的文件
    decision_file = tmp_path / "valid.json"
    decisions = [
        {
            "file": str(test_file_abs),
            "index": 1,
            "type": "paragraph",
            "decision": "keep"
        },
        {
            "file": str(test_file_abs),
            "index": 2,
            "type": "paragraph",
            "decision": "delete"
        }
    ]
    decision_file.write_text(json.dumps(decisions))
    kd_tool.decision_file = decision_file
    
    # 设置一些块数据
    kd_tool.blocks_data = [
        (test_file_abs, 1, "paragraph", "Test content 1"),
        (test_file_abs, 2, "paragraph", "Test content 2")
    ]
    
    result = kd_tool.load_decisions()
    assert result is True
    expected_decisions = {
        str(test_file_abs) + "::1::paragraph": "keep",
        str(test_file_abs) + "::2::paragraph": "delete"
    }
    assert kd_tool.block_decisions == expected_decisions

def test_load_decisions_relative_path(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试加载包含相对路径的决策文件。
    """
    # 创建测试文件
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    test_file = input_dir / "test.md"
    test_file.write_text("Test content")
    
    # 设置输入目录
    kd_tool.set_input_dir(input_dir)
    
    # 创建测试决策文件
    decisions = [{
        "file": "test.md",  # 相对路径
        "index": 0,
        "type": "paragraph",
        "decision": "keep"
    }]
    kd_tool.decision_file.write_text(json.dumps(decisions), encoding='utf-8')
    
    # 添加一些块数据
    kd_tool.blocks_data = [(test_file.resolve(), 0, "paragraph", "Test content")]
    
    # 运行测试
    result = kd_tool.load_decisions()
    assert result is True
    assert len(kd_tool.block_decisions) == 1

def test_save_decisions_no_decisions(kd_tool: KDToolCLI) -> None:
    """
    测试在没有决策的情况下保存决策。
    """
    result = kd_tool.save_decisions()
    assert result is False

def test_save_decisions_success(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试成功保存决策。
    """
    # 设置输入目录
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    kd_tool.set_input_dir(input_dir)
    
    # 创建测试文件
    test_file = input_dir / "test.md"
    test_file.write_text("Test content")
    test_file_abs = test_file.resolve()
    
    # 设置一些块数据和决策
    kd_tool.blocks_data = [
        (test_file_abs, 1, "paragraph", "Test content 1"),
        (test_file_abs, 2, "paragraph", "Test content 2")
    ]
    kd_tool.block_decisions = {
        str(test_file_abs) + "::1::paragraph": "keep",
        str(test_file_abs) + "::2::paragraph": "delete"
    }
    
    # 保存决策
    result = kd_tool.save_decisions()
    assert result is True
    
    # 验证保存的文件内容
    saved_data = json.loads(kd_tool.decision_file.read_text())
    assert len(saved_data) == 2
    assert all(isinstance(item, dict) for item in saved_data)
    assert all(key in item for item in saved_data for key in ["file", "index", "type", "decision"])
    
    # 验证相对路径
    for item in saved_data:
        assert item["file"] == "test.md"
        assert item["type"] == "paragraph"
        assert item["decision"] in ["keep", "delete"]
        assert item["index"] in [1, 2]

def test_save_decisions_no_input_dir(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试在没有设置输入目录的情况下保存决策。
    """
    # 创建测试文件
    test_file = tmp_path / "test.md"
    test_file.write_text("Test content")
    test_file_abs = test_file.resolve()
    
    # 设置一些块数据和决策
    kd_tool.blocks_data = [
        (test_file_abs, 1, "paragraph", "Test content 1"),
        (test_file_abs, 2, "paragraph", "Test content 2")
    ]
    kd_tool.block_decisions = {
        str(test_file_abs) + "::1::paragraph": "keep",
        str(test_file_abs) + "::2::paragraph": "delete"
    }
    
    # 保存决策
    result = kd_tool.save_decisions()
    assert result is True
    
    # 验证保存的文件内容
    saved_data = json.loads(kd_tool.decision_file.read_text())
    assert len(saved_data) == 2
    assert all(isinstance(item, dict) for item in saved_data)
    assert all(key in item for item in saved_data for key in ["file", "index", "type", "decision"])
    
    # 验证绝对路径
    for item in saved_data:
        assert Path(item["file"]).is_absolute()
        assert item["type"] == "paragraph"
        assert item["decision"] in ["keep", "delete"]
        assert item["index"] in [1, 2]

def test_save_decisions_relative_path(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试保存决策时使用相对路径。
    """
    # 创建测试文件
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    test_file = input_dir / "test.md"
    test_file.write_text("Test content")
    
    # 设置输入目录
    kd_tool.set_input_dir(input_dir)
    
    # 添加测试决策
    key = f"{test_file.resolve()}::0::paragraph"
    kd_tool.block_decisions[key] = "keep"
    
    # 运行测试
    result = kd_tool.save_decisions()
    assert result is True
    
    # 验证保存的内容
    saved_data = json.loads(kd_tool.decision_file.read_text(encoding='utf-8'))
    assert len(saved_data) == 1
    assert saved_data[0]["file"] == "test.md"  # 应该是相对路径

def test_apply_decisions_no_blocks(kd_tool: KDToolCLI) -> None:
    """
    测试在没有块数据的情况下应用决策。
    """
    result = kd_tool.apply_decisions()
    assert result is True  # 没有块也算成功
    assert len(kd_tool.blocks_data) == 0

def test_apply_decisions_success(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试成功应用决策。
    """
    # 设置输入目录
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    kd_tool.set_input_dir(input_dir)
    
    # 创建测试文件
    test_file = input_dir / "test.md"
    test_file.write_text("Test content")
    test_file_abs = test_file.resolve()
    
    # 设置一些块数据和决策
    kd_tool.blocks_data = [
        (test_file_abs, 1, "paragraph", "Test content 1"),
        (test_file_abs, 2, "paragraph", "Test content 2"),
        (test_file_abs, 3, "paragraph", "Test content 3")
    ]
    kd_tool.block_decisions = {
        str(test_file_abs) + "::1::paragraph": "keep",
        str(test_file_abs) + "::2::paragraph": "delete",
        str(test_file_abs) + "::3::paragraph": "undecided"
    }
    
    # 应用决策
    result = kd_tool.apply_decisions()
    assert result is True
    
    # 验证输出文件
    output_file = kd_tool.output_dir / f"test{constants.DEFAULT_OUTPUT_SUFFIX}.md"
    assert output_file.exists()
    content = output_file.read_text()
    assert "Test content 1" in content
    assert "Test content 2" not in content
    assert "Test content 3" in content  # undecided 应该被保留

def test_apply_decisions_no_decisions(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试在没有决策的情况下应用决策。
    """
    # 设置输入目录
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    kd_tool.set_input_dir(input_dir)
    
    # 创建测试文件
    test_file = input_dir / "test.md"
    test_file.write_text("Test content")
    test_file_abs = test_file.resolve()
    
    # 设置一些块数据但没有决策
    kd_tool.blocks_data = [
        (test_file_abs, 1, "paragraph", "Test content 1"),
        (test_file_abs, 2, "paragraph", "Test content 2")
    ]
    
    # 应用决策
    result = kd_tool.apply_decisions()
    assert result is True
    
    # 验证输出文件（所有块都应该被保留）
    output_file = kd_tool.output_dir / f"test{constants.DEFAULT_OUTPUT_SUFFIX}.md"
    assert output_file.exists()
    content = output_file.read_text()
    assert "Test content 1" in content
    assert "Test content 2" in content

def test_read_files_success(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试成功读取文件的情况。
    """
    # 设置输入目录
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    kd_tool.set_input_dir(input_dir)
    
    # 创建测试文件
    test_file = input_dir / "test.md"
    test_file.write_text("# Test\n\nContent")
    
    result = kd_tool._read_files()
    assert result is True
    assert test_file in kd_tool.markdown_files_content
    assert kd_tool.markdown_files_content[test_file] == "# Test\n\nContent"

def test_read_files_no_markdown(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试没有 Markdown 文件的情况。
    """
    # 设置输入目录
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    kd_tool.set_input_dir(input_dir)
    
    # 创建非 Markdown 文件
    test_file = input_dir / "test.txt"
    test_file.write_text("Content")
    
    result = kd_tool._read_files()
    assert result is False  # 应该返回 False，因为没有找到 Markdown 文件
    assert len(kd_tool.markdown_files_content) == 0

def test_read_files_empty_directory(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试空目录的情况。
    """
    # 设置输入目录
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    kd_tool.set_input_dir(input_dir)
    
    result = kd_tool._read_files()
    assert result is False  # 应该返回 False，因为目录为空
    assert len(kd_tool.markdown_files_content) == 0

def test_parse_markdown_success(kd_tool: KDToolCLI) -> None:
    """
    测试成功解析 Markdown 的情况。
    """
    # 设置测试数据
    test_file = Path("test.md")
    kd_tool.markdown_files_content = {
        test_file: """# Title

Paragraph 1

- List item 1
- List item 2

```python
print('Hello')
```"""
    }
    
    result = kd_tool._parse_markdown()
    assert result is True
    assert len(kd_tool.blocks_data) > 0
    
    # 验证块类型和内容
    block_types = [(block[2], block[3].strip()) for block in kd_tool.blocks_data]
    
    # 验证标题
    assert any(t == "heading" and "Title" in c for t, c in block_types), "标题块未找到"
    
    # 验证段落
    assert any(t == "paragraph" and "Paragraph 1" in c for t, c in block_types), "段落块未找到"
    
    # 验证列表项
    assert any(t == "list" and "List item 1" in c for t, c in block_types), "列表项1未找到"
    assert any(t == "list" and "List item 2" in c for t, c in block_types), "列表项2未找到"
    
    # 验证代码块
    assert any(t == "code" and "print('Hello')" in c for t, c in block_types), "代码块未找到"

def test_parse_markdown_empty_content(kd_tool: KDToolCLI) -> None:
    """
    测试解析空内容的情况。
    """
    # 设置测试数据
    test_file = Path("test.md")
    kd_tool.markdown_files_content = {test_file: ""}
    
    result = kd_tool._parse_markdown()
    assert result is False  # 应该返回 False，因为内容为空
    assert len(kd_tool.blocks_data) == 0

def test_parse_markdown_invalid_content(kd_tool: KDToolCLI) -> None:
    """
    测试解析无效内容的情况。
    """
    # 设置测试数据
    test_file = Path("test.md")
    kd_tool.markdown_files_content = {test_file: None}
    
    result = kd_tool._parse_markdown()
    assert result is False

def test_initialize_decisions(kd_tool: KDToolCLI) -> None:
    """
    测试初始化决策的情况。
    """
    # 设置测试数据
    test_file = Path("test.md")
    kd_tool.blocks_data = [
        (test_file, 1, "paragraph", "Content 1"),
        (test_file, 2, "paragraph", "Content 2")
    ]
    
    kd_tool._initialize_decisions()
    assert len(kd_tool.block_decisions) == 2
    for key in kd_tool.block_decisions:
        assert kd_tool.block_decisions[key] == "undecided"

def test_ensure_dirs_exist(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试确保目录存在的情况。
    """
    # 设置测试目录
    output_dir = tmp_path / "output"
    decision_dir = tmp_path / "decisions"
    kd_tool.output_dir = output_dir
    kd_tool.decision_file = decision_dir / "decisions.json"
    
    kd_tool._ensure_dirs_exist()
    assert output_dir.exists()
    assert decision_dir.exists()

def test_ensure_dirs_exist_error(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试确保目录存在时发生错误的情况。
    """
    # 设置测试目录（使用只读目录）
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    # 修改目录权限为只读
    output_dir.chmod(0o444)
    
    kd_tool.output_dir = output_dir
    # 测试应该不会抛出异常
    kd_tool._ensure_dirs_exist()

def test_path_resolution_with_cwd(kd_tool: KDToolCLI, tmp_path: Path) -> None:
    """
    测试相对于当前工作目录的路径解析。
    """
    # 创建测试文件
    test_file = tmp_path / "test.md"
    test_file.write_text("Test content")
    
    # 切换到临时目录
    os.chdir(tmp_path)
    
    # 创建测试决策文件
    decisions = [{
        "file": "test.md",  # 相对于当前工作目录的路径
        "index": 0,
        "type": "paragraph",
        "decision": "keep"
    }]
    kd_tool.decision_file.write_text(json.dumps(decisions), encoding='utf-8')
    
    # 添加一些块数据
    kd_tool.blocks_data = [(test_file.resolve(), 0, "paragraph", "Test content")]
    
    # 运行测试
    result = kd_tool.load_decisions()
    assert result is True
    assert len(kd_tool.block_decisions) == 1 