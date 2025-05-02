# tests/test_integration.py
"""
集成测试，测试不同模块之间的交互。
"""

import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock # 添加 MagicMock

# 导入核心类和函数
from knowledge_distiller_kd.core.kd_tool_CLI import KDToolCLI
from knowledge_distiller_kd.core.document_processor import ContentBlock
from knowledge_distiller_kd.core import constants
from knowledge_distiller_kd.core.utils import create_decision_key # 导入


# --- Fixtures ---

@pytest.fixture
def sample_input_dir(tmp_path: Path) -> Path:
    """创建一个包含测试 Markdown 文件的临时输入目录"""
    input_dir = tmp_path / "integration_input"
    input_dir.mkdir()
    # 文件1: 包含重复内容和代码块
    file1_content = """# 文件1

这是文件1的普通段落。

```python
def func1():
    print("Hello from file 1")
```

这是另一段普通文本。
"""
    # 文件2: 包含与文件1部分重复的内容和不同的代码块
    file2_content = """# 文件2

这是文件1的普通段落。

```javascript
function func2() {
    console.log("Hello from file 2");
}
```

这是文件2独有的文本。
"""
    # 文件3: 内容完全不同
    file3_content = """# 文件3

完全不同的内容。
"""
    (input_dir / "file1.md").write_text(file1_content, encoding="utf-8")
    (input_dir / "file2.md").write_text(file2_content, encoding="utf-8")
    (input_dir / "file3.md").write_text(file3_content, encoding="utf-8")
    return input_dir

@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """创建一个临时输出目录"""
    out_dir = tmp_path / "integration_output"
    out_dir.mkdir()
    return out_dir

@pytest.fixture
def decision_file(tmp_path: Path) -> Path:
    """提供一个临时的决策文件路径"""
    return tmp_path / "integration_decisions.json"

@pytest.fixture
def kd_tool_instance(sample_input_dir: Path, output_dir: Path, decision_file: Path) -> KDToolCLI:
    """创建一个配置好的 KDToolCLI 实例用于集成测试"""
    # 使用较高的语义阈值，避免不必要的匹配干扰测试
    return KDToolCLI(
        input_dir=sample_input_dir,
        output_dir=output_dir,
        decision_file=decision_file,
        similarity_threshold=0.95, # 设置较高阈值
        skip_semantic=False # 默认不跳过
    )

# --- 测试用例 ---

def test_full_analysis_and_apply(kd_tool_instance: KDToolCLI, output_dir: Path, sample_input_dir: Path) -> None:
    """
    测试完整的分析流程（MD5+语义）和决策应用。
    注意：这个测试会实际加载模型，可能较慢。如果需要更快，需要 mock 模型。
    """
    # 运行分析
    analysis_success = kd_tool_instance.run_analysis()
    assert analysis_success is True
    assert kd_tool_instance._analysis_completed is True
    assert len(kd_tool_instance.blocks_data) > 0 # 确保处理了块

    # 检查 MD5 是否找到了重复段落 "这是文件1的普通段落。"
    md5_duplicates = kd_tool_instance.md5_analyzer.md5_duplicates if kd_tool_instance.md5_analyzer else []
    found_md5_para = False
    deleted_key_md5 = None
    kept_key_md5 = None
    for group in md5_duplicates:
        texts = {block.analysis_text for block in group}
        target_text = "这是文件1的普通段落。"
        if target_text in texts:
            found_md5_para = True
            for block in group:
                 key = create_decision_key(block.file_path, block.block_id, block.block_type)
                 if Path(block.file_path).name == "file1.md": kept_key_md5 = key
                 elif Path(block.file_path).name == "file2.md": deleted_key_md5 = key
            break
    assert found_md5_para, "MD5 未找到预期的重复段落"
    assert kept_key_md5 is not None and kd_tool_instance.block_decisions.get(kept_key_md5) == constants.DECISION_KEEP
    assert deleted_key_md5 is not None and kd_tool_instance.block_decisions.get(deleted_key_md5) == constants.DECISION_DELETE

    # 应用决策
    apply_success = kd_tool_instance.apply_decisions()
    assert apply_success is True

    # 验证输出文件
    output_file1 = output_dir / f"file1{constants.DEFAULT_OUTPUT_SUFFIX}.md"
    output_file2 = output_dir / f"file2{constants.DEFAULT_OUTPUT_SUFFIX}.md"
    output_file3 = output_dir / f"file3{constants.DEFAULT_OUTPUT_SUFFIX}.md"

    assert output_file1.exists()
    assert output_file2.exists()
    assert output_file3.exists()

    # ==================== 修改：调整输出内容断言 ====================
    # 检查 file1 输出：应该包含所有内容，但标题不带 #
    content1 = output_file1.read_text(encoding="utf-8")
    assert "文件1" in content1 # 检查不带 # 的标题
    assert "这是文件1的普通段落。" in content1
    assert "def func1():" in content1 # 检查代码块内容
    assert "这是另一段普通文本。" in content1

    # 检查 file2 输出：应该不包含重复的段落，标题不带 #
    content2 = output_file2.read_text(encoding="utf-8")
    assert "文件2" in content2 # 检查不带 # 的标题
    assert "这是文件1的普通段落。" not in content2 # 验证被删除
    assert "function func2()" in content2 # 检查代码块内容
    assert "这是文件2独有的文本。" in content2

    # 检查 file3 输出：应该包含所有内容，标题不带 #
    content3 = output_file3.read_text(encoding="utf-8")
    assert "文件3" in content3 # 检查不带 # 的标题
    assert "完全不同的内容。" in content3
    # ============================================================

def test_save_and_load_decisions(kd_tool_instance: KDToolCLI, decision_file: Path) -> None:
    """测试决策的保存和加载功能"""
    # 运行分析以生成一些块和默认决策
    kd_tool_instance.run_analysis()
    assert len(kd_tool_instance.block_decisions) > 0

    # 手动修改一些决策
    keys_to_modify = list(kd_tool_instance.block_decisions.keys())[:2]
    original_decisions = {}
    if len(keys_to_modify) >= 1:
        original_decisions[keys_to_modify[0]] = constants.DECISION_KEEP
        kd_tool_instance.block_decisions[keys_to_modify[0]] = constants.DECISION_KEEP
    if len(keys_to_modify) >= 2:
        original_decisions[keys_to_modify[1]] = constants.DECISION_DELETE
        kd_tool_instance.block_decisions[keys_to_modify[1]] = constants.DECISION_DELETE

    # 保存决策
    save_success = kd_tool_instance.save_decisions()
    assert save_success is True
    assert decision_file.exists()

    # 创建一个新的实例来加载决策
    new_tool = KDToolCLI(
        input_dir=kd_tool_instance.input_dir,
        output_dir=kd_tool_instance.output_dir,
        decision_file=decision_file
    )
    # 需要先处理文档
    new_tool._process_documents()
    # 加载决策
    load_success = new_tool.load_decisions()
    assert load_success is True

    # 比较加载后的决策与原始手动修改的决策
    for key, decision in original_decisions.items():
        assert new_tool.block_decisions.get(key) == decision

