"""
测试 ContentBlock 类的功能，特别是 file_path 和 analysis_text 属性
"""
import pytest
from pathlib import Path
import uuid

from knowledge_distiller_kd.core.models import ContentBlock, BlockType


def test_content_block_attributes():
    """测试 ContentBlock 初始化和属性设置"""
    # 创建 ContentBlock，显式设置所有属性
    cb = ContentBlock(
        file_id="test_file_id",
        text="这是测试文本",
        block_type=BlockType.TEXT,
        file_path="/path/to/test.md",
        analysis_text="处理后的文本用于分析"
    )
    
    # 验证属性赋值是否正确
    assert cb.file_id == "test_file_id"
    assert cb.text == "这是测试文本"
    assert cb.block_type == BlockType.TEXT
    assert cb.file_path == "/path/to/test.md"
    assert cb.analysis_text == "处理后的文本用于分析"


def test_content_block_default_values():
    """测试 ContentBlock 默认值处理"""
    # 仅设置必需字段，测试默认值
    cb = ContentBlock(
        file_id="test_file_id",
        text="这是测试文本",
        block_type=BlockType.TEXT
    )
    
    # 验证 file_path 和 analysis_text 默认值
    assert cb.file_path == ""
    assert cb.analysis_text == "这是测试文本"  # 默认与 text 相同


def test_content_block_metadata_original_path():
    """测试 metadata 中 original_path 属性映射到 file_path"""
    # 在 metadata 中包含 original_path
    cb = ContentBlock(
        file_id="test_file_id",
        text="这是测试文本",
        block_type=BlockType.TEXT,
        metadata={"original_path": "/original/path/to/file.md"}
    )
    
    # 验证 file_path 是否从 metadata 中提取
    assert cb.file_path == "/original/path/to/file.md"


def test_content_block_serialization():
    """测试 ContentBlock 的序列化和反序列化"""
    original_cb = ContentBlock(
        file_id="test_file_id",
        text="这是测试文本",
        block_type=BlockType.TEXT,
        file_path="/path/to/test.md",
        analysis_text="处理后的文本用于分析"
    )
    
    # 序列化为字典
    cb_dict = original_cb.to_dict()
    
    # 验证序列化是否包含所有属性
    assert cb_dict["file_id"] == "test_file_id"
    assert cb_dict["text"] == "这是测试文本"
    assert cb_dict["block_type"] == BlockType.TEXT.value
    assert cb_dict["file_path"] == "/path/to/test.md"
    assert cb_dict["analysis_text"] == "处理后的文本用于分析"
    
    # 从字典反序列化
    deserialized_cb = ContentBlock.from_dict(cb_dict)
    
    # 验证反序列化是否正确恢复所有值
    assert deserialized_cb.file_id == original_cb.file_id
    assert deserialized_cb.text == original_cb.text
    assert deserialized_cb.block_type == original_cb.block_type
    assert deserialized_cb.file_path == original_cb.file_path
    assert deserialized_cb.analysis_text == original_cb.analysis_text


def test_content_block_from_dict_defaults():
    """测试从缺少某些字段的字典创建 ContentBlock"""
    # 最小化字典，只包含必需字段
    cb_dict = {
        "file_id": "test_file_id",
        "text": "这是测试文本",
    }
    
    # 从最小字典创建
    cb = ContentBlock.from_dict(cb_dict)
    
    # 验证默认值处理
    assert cb.file_id == "test_file_id"
    assert cb.text == "这是测试文本"
    assert cb.block_type == BlockType.UNKNOWN  # 应该默认为 UNKNOWN
    assert cb.file_path == ""
    assert cb.analysis_text == "这是测试文本"  # 应该默认与 text 相同 