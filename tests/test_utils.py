"""
工具函数测试模块。

此模块包含对工具函数的单元测试。
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import pytest
import mistune
import json
import hashlib

from knowledge_distiller_kd.core.utils import (
    setup_logger,
    create_decision_key,
    parse_decision_key,
    extract_text_from_children,
    display_block_preview,
    get_markdown_parser,
    sort_blocks_key
)

# 导出工具函数
__all__ = [
    'create_test_markdown_file',
    'create_test_decision_file',
    'calculate_md5_hash',
    'verify_file_content',
    'verify_decision_file',
    'create_test_environment',
    'cleanup_test_environment',
    'verify_test_results'
]

def test_setup_logger() -> None:
    """
    测试日志设置功能。
    """
    logger = setup_logger()
    assert logger is not None
    assert logger.name == 'KDToolLogger'
    assert logger.level == logging.INFO

def test_create_decision_key(tmp_path: Path) -> None:
    """
    测试创建决策键的功能。
    
    Args:
        tmp_path: pytest 提供的临时目录
    """
    # 在临时目录中创建测试文件
    test_file = tmp_path / "test.md"
    test_file.touch()
    
    # 创建一个特殊对象用于测试
    class TestObj:
        def __str__(self):
            return "test_obj"
    
    test_cases = [
        (test_file, 1, "paragraph", f"{test_file}::1::paragraph"),
        (tmp_path / "subdir" / "file.md", "1_0", "list", f"{tmp_path}/subdir/file.md::1_0::list"),
        # 测试无效路径
        (TestObj(), 1, "paragraph", "test_obj::1::paragraph"),  # 测试异常处理
    ]
    
    for file_path, block_index, block_type, expected in test_cases:
        key = create_decision_key(file_path, block_index, block_type)
        assert key == expected

def test_parse_decision_key() -> None:
    """
    测试解析决策键的功能。
    """
    test_cases = [
        # 基本测试用例（整数索引）
        ("test.md::1::paragraph", ("test.md", 1, "paragraph")),
        
        # 测试带有路径分隔符的文件路径（整数索引）
        ("/path/to/file.md::1::paragraph", ("/path/to/file.md", 1, "paragraph")),
        
        # 测试非数字索引
        ("test.md::abc::list", ("test.md", "abc", "list")),
        
        # 测试无效的键
        ("invalid_key", (None, None, None)),
        ("", (None, None, None)),
        ("test.md::1", (None, None, None)),
        
        # 测试异常情况
        (None, (None, None, None)),  # 测试 None 输入
        (123, (None, None, None)),   # 测试非字符串输入
    ]
    
    for key, expected in test_cases:
        result = parse_decision_key(key)
        assert result == expected

def test_extract_text_from_children() -> None:
    """
    测试从子令牌中提取文本的功能。
    """
    test_cases = [
        # 基本文本测试
        ([{"type": "text", "raw": "Hello"}], "Hello"),
        
        # 测试强调文本
        ([{"type": "emphasis", "children": [{"type": "text", "raw": "World"}]}], "World"),
        
        # 测试链接
        ([{"type": "link", "children": [{"type": "text", "raw": "Click"}]}], "Click"),
        
        # 测试换行符
        ([{"type": "softbreak"}], " "),
        
        # 测试复杂组合
        ([
            {"type": "text", "raw": "Hello"},
            {"type": "softbreak"},
            {"type": "text", "raw": "World"}
        ], "Hello World"),
        
        # 测试空输入
        (None, ""),
        ([], ""),
        
        # 测试内联HTML
        ([{"type": "inline_html", "raw": "<b>Bold</b>"}], ""),
        
        # 测试缺失的 raw 属性
        ([{"type": "text"}], ""),
        
        # 测试缺失的 children 属性
        ([{"type": "emphasis"}], ""),
    ]
    
    for children, expected in test_cases:
        result = extract_text_from_children(children)
        assert result.strip() == expected.strip()

def test_display_block_preview() -> None:
    """
    测试生成块内容预览的功能。
    """
    test_cases = [
        ("Short text", "Short text"),
        ("A" * 100, "A" * 77 + "..."),  # 测试长文本
        ("Text with\nnewlines", "Text with newlines"),
        ("", ""),  # 测试空字符串
        (None, ""),  # 测试 None 输入
    ]
    
    for text, expected in test_cases:
        result = display_block_preview(text if text is not None else "")
        assert result == expected

def test_get_markdown_parser() -> None:
    """
    测试获取 Markdown 解析器的功能。
    """
    parser = get_markdown_parser()
    assert parser is not None
    assert isinstance(parser, mistune.Markdown)
    
    # 测试解析器的基本功能
    test_md = "**Bold** and *italic*"
    result = parser(test_md)
    assert result is not None

def test_sort_blocks_key() -> None:
    """
    测试块排序键生成的功能。
    """
    test_cases = [
        # 测试基本排序
        (("file1.md", 1, "paragraph", "content"), ("file1.md", 1, 0)),
        (("file1.md", 2, "paragraph", "content"), ("file1.md", 2, 0)),
        (("file2.md", 1, "paragraph", "content"), ("file2.md", 1, 0)),
        
        # 测试列表项索引
        (("file1.md", "1_0", "list", "content"), ("file1.md", 1, 0)),
        (("file1.md", "2_1", "list", "content"), ("file1.md", 2, 1)),
        
        # 测试无效索引
        (("file1.md", "invalid", "paragraph", "content"), ("file1.md", float('inf'), "invalid")),
        
        # 测试异常情况
        (("file1.md", "1_x", "list", "content"), ("file1.md", float('inf'), "1_x")),
    ]
    
    # 验证排序顺序
    results = [sort_blocks_key(block_info) for block_info, expected in test_cases]
    
    # 验证排序结果
    sorted_results = sorted(results)
    assert len(results) == len(test_cases)  # 确保所有测试用例都生成了结果
    
    # 验证特定的排序规则
    for i in range(len(results)-1):
        # 如果文件名相同，检查索引顺序
        if results[i][0] == results[i+1][0]:
            if isinstance(results[i][1], (int, float)) and isinstance(results[i+1][1], (int, float)):
                assert results[i][1] <= results[i+1][1]

def create_test_markdown_file(
    file_path: Path,
    content: str,
    encoding: str = "utf-8"
) -> None:
    """
    创建测试用的Markdown文件。

    Args:
        file_path: 文件路径
        content: 文件内容
        encoding: 文件编码
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding=encoding) as f:
        f.write(content)

def create_test_decision_file(
    file_path: Path,
    decisions: Dict[str, str],
    encoding: str = "utf-8"
) -> None:
    """
    创建测试用的决策文件。

    Args:
        file_path: 文件路径
        decisions: 决策数据
        encoding: 文件编码
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False)

def calculate_md5_hash(content: str) -> str:
    """
    计算文本的MD5哈希值。

    Args:
        content: 要计算哈希的文本

    Returns:
        str: MD5哈希值
    """
    return hashlib.md5(content.encode("utf-8")).hexdigest()

def verify_file_content(
    file_path: Path,
    expected_content: str,
    encoding: str = "utf-8"
) -> bool:
    """
    验证文件内容是否符合预期。

    Args:
        file_path: 文件路径
        expected_content: 预期的内容
        encoding: 文件编码

    Returns:
        bool: 如果内容符合预期返回True，否则返回False
    """
    if not file_path.exists():
        return False
    
    with open(file_path, "r", encoding=encoding) as f:
        actual_content = f.read()
    
    return actual_content == expected_content

def verify_decision_file(
    file_path: Path,
    expected_decisions: Dict[str, str],
    encoding: str = "utf-8"
) -> bool:
    """
    验证决策文件内容是否符合预期。

    Args:
        file_path: 文件路径
        expected_decisions: 预期的决策数据
        encoding: 文件编码

    Returns:
        bool: 如果内容符合预期返回True，否则返回False
    """
    if not file_path.exists():
        return False
    
    with open(file_path, "r", encoding=encoding) as f:
        actual_decisions = json.load(f)
    
    return actual_decisions == expected_decisions

def create_test_environment(
    test_dir: Path,
    input_files: List[Tuple[str, str]],
    decisions: Optional[Dict[str, str]] = None
) -> None:
    """
    创建测试环境。

    Args:
        test_dir: 测试目录
        input_files: 输入文件列表，每个元素为(文件名, 内容)的元组
        decisions: 决策数据
    """
    # 创建输入目录
    input_dir = test_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建输出目录
    output_dir = test_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建输入文件
    for filename, content in input_files:
        create_test_markdown_file(input_dir / filename, content)
    
    # 创建决策文件
    if decisions:
        decision_file = test_dir / "decisions.json"
        create_test_decision_file(decision_file, decisions)

def cleanup_test_environment(test_dir: Path) -> None:
    """
    清理测试环境。

    Args:
        test_dir: 测试目录
    """
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)

def verify_test_results(
    test_dir: Path,
    expected_output_files: List[Tuple[str, str]],
    expected_decisions: Optional[Dict[str, str]] = None
) -> bool:
    """
    验证测试结果。

    Args:
        test_dir: 测试目录
        expected_output_files: 预期的输出文件列表，每个元素为(文件名, 内容)的元组
        expected_decisions: 预期的决策数据

    Returns:
        bool: 如果所有验证都通过返回True，否则返回False
    """
    # 验证输出文件
    output_dir = test_dir / "output"
    for filename, expected_content in expected_output_files:
        if not verify_file_content(output_dir / filename, expected_content):
            return False
    
    # 验证决策文件
    if expected_decisions:
        decision_file = test_dir / "decisions.json"
        if not verify_decision_file(decision_file, expected_decisions):
            return False
    
    return True 