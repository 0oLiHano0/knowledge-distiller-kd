"""
测试配置文件，用于设置测试环境。

此模块包含：
1. 测试环境配置
2. 测试日志配置
3. 测试数据目录配置
"""

import os
import sys
import logging
from pathlib import Path
from typing import Generator

import pytest

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# 测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_INPUT_DIR = TEST_DATA_DIR / "input"
TEST_OUTPUT_DIR = TEST_DATA_DIR / "output"
TEST_DECISION_FILE = TEST_DATA_DIR / "decisions.json"

# 确保测试目录存在
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_INPUT_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

def pytest_sessionfinish(session, exitstatus):
    """
    测试会话结束时清理测试数据目录。
    """
    import shutil
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)

@pytest.fixture
def temp_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """
    创建一个临时目录用于测试。
    
    Args:
        tmp_path: pytest 提供的临时路径夹具
        
    Yields:
        Path: 临时目录的路径
    """
    yield tmp_path

@pytest.fixture
def sample_markdown_content() -> str:
    """
    提供示例 Markdown 内容用于测试。
    
    Returns:
        str: 示例 Markdown 内容
    """
    return """# 测试标题

这是一个测试段落。

## 二级标题

这是另一个段落。

```python
def test_function():
    print("Hello, World!")
```

- 列表项 1
- 列表项 2
"""

@pytest.fixture
def sample_blocks_data() -> list[dict]:
    """
    提供示例块数据用于测试。
    
    Returns:
        list[dict]: 示例块数据列表
    """
    return [
        {
            "file_name": "test1.md",
            "block_type": "paragraph",
            "content": "这是第一个测试段落。",
            "block_id": "test1_1"
        },
        {
            "file_name": "test1.md",
            "block_type": "paragraph",
            "content": "这是第二个测试段落。",
            "block_id": "test1_2"
        },
        {
            "file_name": "test2.md",
            "block_type": "paragraph",
            "content": "这是第一个测试段落。",  # 重复内容
            "block_id": "test2_1"
        }
    ] 