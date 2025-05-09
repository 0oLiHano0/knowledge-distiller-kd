"""
测试运行脚本，用于运行所有测试。

此脚本包含：
1. 测试运行配置
2. 测试结果报告
3. 测试覆盖率报告
"""

import os
import sys
import pytest
import coverage
from pathlib import Path

def main():
    """
    运行所有测试并生成报告。
    """
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    
    # 配置coverage
    cov = coverage.Coverage(
        branch=True,
        source=[str(project_root)],
        omit=[
            "*/tests/*",
            "*/venv/*",
            "*/__pycache__/*"
        ]
    )
    
    try:
        # 开始收集覆盖率数据
        cov.start()
        
        # 运行测试
        test_dir = project_root / "tests"
        pytest_args = [
            str(test_dir),
            "-v",
            "--cov=knowledge_distiller_kd",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_report"
        ]
        exit_code = pytest.main(pytest_args)
        
        # 停止收集覆盖率数据
        cov.stop()
        cov.save()
        
        # 生成覆盖率报告
        cov.report()
        cov.html_report(directory="coverage_report")
        
        return exit_code
    except Exception as e:
        print(f"测试运行出错: {e}")
        return 1
    finally:
        # 确保覆盖率收集器被正确关闭
        if 'cov' in locals():
            cov.stop()
            cov.save()

if __name__ == "__main__":
    sys.exit(main()) 