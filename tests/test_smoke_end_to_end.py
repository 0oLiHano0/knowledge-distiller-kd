"""
端到端测试（Smoke Test）：验证工具在不同模式下的基本功能。

此模块包含：
1. 预过滤模式测试 - 仅执行预过滤并验证输出统计
2. 默认模式测试 - 执行完整流程，验证预过滤和MD5分析
"""

import pytest
import subprocess
import sys
import re
from pathlib import Path


@pytest.mark.smoke
def test_prefilter_mode():
    """
    测试 --pre-filter 模式：执行预过滤并验证输出统计
    """
    cmd = [sys.executable, "-m", "knowledge_distiller_kd.cli", "--pre-filter", "--input-dir", "input/"]
    
    # 运行命令并捕获输出
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False  # 不自动检查返回码
    )
    
    # 检查返回码是否为0（成功）
    assert result.returncode == 0, f"命令执行失败，错误信息：{result.stderr}"
    
    output = result.stdout + result.stderr
    
    # 验证预过滤输出
    prefilter_pattern = r"\[Prefilter\] Scanned (\d+) files, filtered (\d+) duplicates → (\d+) .*remain"
    match = re.search(prefilter_pattern, output)
    assert match, f"未找到预过滤统计输出，实际输出: {output}"
    
    # 提取数字并验证
    total_files = int(match.group(1))
    filtered_dupes = int(match.group(2))
    remain_files = int(match.group(3))
    
    assert total_files >= 4, f"预期至少扫描4个文件，实际扫描了{total_files}个"
    assert filtered_dupes >= 2, f"预期至少过滤2个重复文件，实际过滤了{filtered_dupes}个"
    assert remain_files >= 2, f"预期至少保留2个文件，实际保留了{remain_files}个"
    
    # 验证没有MD5分析日志
    assert "MD5 duplicates found:" not in output, "预过滤模式不应该运行MD5分析"


@pytest.mark.smoke
def test_default_mode():
    """
    测试默认模式：执行完整流程，包括预过滤和MD5分析
    """
    # 添加 --non-interactive 参数，避免启动交互式 UI
    cmd = [sys.executable, "-m", "knowledge_distiller_kd.cli", "--input-dir", "input/", "--non-interactive"]
    
    # 运行命令并捕获输出
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False  # 不自动检查返回码
    )
    
    # 检查返回码是否为0（成功）
    assert result.returncode == 0, f"命令执行失败，错误信息：{result.stderr}"
    
    output = result.stdout + result.stderr
    
    # 验证预过滤输出
    prefilter_pattern = r"\[\*\] 预过滤完成: 扫描了 (\d+) 个文件, 过滤了 (\d+) 个重复文件, 剩余 (\d+) 个唯一文件"
    match = re.search(prefilter_pattern, output)
    assert match, f"未找到预过滤统计输出，实际输出: {output}"
    
    # 提取数字并验证
    total_files = int(match.group(1))
    filtered_dupes = int(match.group(2))
    remain_files = int(match.group(3))
    
    assert total_files >= 4, f"预期至少扫描4个文件，实际扫描了{total_files}个"
    assert filtered_dupes >= 2, f"预期至少过滤2个重复文件，实际过滤了{filtered_dupes}个"
    assert remain_files >= 2, f"预期至少保留2个文件，实际保留了{remain_files}个"
    
    # 验证MD5分析日志
    md5_pattern = r"MD5 duplicates found: (\d+) pairs"
    md5_match = re.search(md5_pattern, output)
    assert md5_match, f"未找到MD5分析输出，实际输出: {output}"
    
    md5_dupes = int(md5_match.group(1))
    assert md5_dupes >= 1, f"预期至少发现1对MD5重复，实际发现了{md5_dupes}对" 