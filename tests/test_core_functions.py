"""
核心功能测试模块。

此模块包含对知识蒸馏工具核心功能的测试。
"""

import pytest
from pathlib import Path
from typing import List, Tuple, Dict

from knowledge_distiller_kd.core.kd_tool_CLI import KDToolCLI
from knowledge_distiller_kd.core.md5_analyzer import MD5Analyzer
from knowledge_distiller_kd.core.semantic_analyzer import SemanticAnalyzer
from tests.test_data_generator import DataGenerator
from tests.test_utils import (
    verify_file_content,
    verify_decision_file,
    cleanup_test_environment
)

# 测试数据生成器
test_data_generator = DataGenerator()

def test_md5_duplicate_detection(tmp_path: Path) -> None:
    """
    测试MD5重复检测功能。
    """
    # 创建测试目录
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    decision_file = tmp_path / "decisions.json"
    input_dir.mkdir()
    output_dir.mkdir()

    # 创建测试文件
    test_data_generator.generate_md5_duplicates(input_dir)

    # 初始化工具
    tool = KDToolCLI(
        input_dir=input_dir,
        output_dir=output_dir,
        decision_file=decision_file,
        skip_semantic=True
    )

    # 运行分析
    assert tool.run_analysis() is True

    # 验证结果
    assert len(tool.blocks_data) > 0
    assert tool.md5_analyzer is not None
    assert len(tool.md5_analyzer.duplicate_blocks) > 0

def test_semantic_duplicate_detection(tmp_path: Path) -> None:
    """
    测试语义重复检测功能。
    """
    # 创建测试目录
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    decision_file = tmp_path / "decisions.json"
    input_dir.mkdir()
    output_dir.mkdir()

    # 创建测试文件
    test_data_generator.generate_semantic_duplicates(input_dir)

    # 初始化工具
    tool = KDToolCLI(
        input_dir=input_dir,
        output_dir=output_dir,
        decision_file=decision_file,
        skip_semantic=False
    )

    # 运行分析
    assert tool.run_analysis() is True

    # 验证结果
    assert len(tool.blocks_data) > 0
    assert tool.semantic_analyzer is not None
    assert len(tool.semantic_analyzer.duplicate_blocks) > 0

def test_file_operations(tmp_path: Path) -> None:
    """
    测试文件操作功能。
    """
    # 创建测试目录
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    decision_file = tmp_path / "decisions.json"
    input_dir.mkdir()
    output_dir.mkdir()

    # 创建测试文件
    test_data_generator.generate_test_files(input_dir)

    # 初始化工具
    tool = KDToolCLI(
        input_dir=input_dir,
        output_dir=output_dir,
        decision_file=decision_file,
        skip_semantic=True
    )

    # 运行分析
    assert tool.run_analysis() is True

    # 验证结果
    assert len(tool.blocks_data) > 0
    assert len(tool.markdown_files_content) > 0

def test_decision_handling(tmp_path: Path) -> None:
    """
    测试决策处理功能。
    """
    # 创建测试目录
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    decision_file = tmp_path / "decisions.json"
    input_dir.mkdir()
    output_dir.mkdir()

    # 创建测试文件
    test_data_generator.generate_test_files(input_dir)

    # 初始化工具
    tool = KDToolCLI(
        input_dir=input_dir,
        output_dir=output_dir,
        decision_file=decision_file,
        skip_semantic=True
    )

    # 运行分析
    assert tool.run_analysis() is True

    # 添加一些测试决策
    for block_info in tool.blocks_data[:2]:
        file_path, block_index, block_type, _ = block_info
        key = f"{file_path}::{block_index}::{block_type}"
        tool.block_decisions[key] = "keep"

    # 保存决策
    assert tool.save_decisions() is True

    # 验证决策文件
    assert verify_decision_file(decision_file)

def test_integration(tmp_path: Path) -> None:
    """
    测试整体功能集成。
    """
    # 创建测试目录
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    decision_file = tmp_path / "decisions.json"
    input_dir.mkdir()
    output_dir.mkdir()

    # 创建测试文件
    test_data_generator.generate_test_files(input_dir)

    # 初始化工具
    tool = KDToolCLI(
        input_dir=input_dir,
        output_dir=output_dir,
        decision_file=decision_file,
        skip_semantic=True
    )

    # 运行完整流程
    assert tool.run_analysis() is True
    assert tool.save_decisions() is True
    assert tool.apply_decisions() is True

    # 验证输出
    assert verify_file_content(output_dir) 