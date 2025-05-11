#!/usr/bin/env python
"""
生成存储使用情况报告的独立脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from tests.storage.test_detect_storage_usage import generate_storage_usage_report

if __name__ == "__main__":
    report = generate_storage_usage_report()
    print(report) 