"""
测试文件：检测存储使用情况
用于识别并报告当前持久化存储后端使用情况。
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest

from knowledge_distiller_kd.storage.storage_interface import StorageInterface
from knowledge_distiller_kd.storage.file_storage import FileStorage
from knowledge_distiller_kd.storage.sqlite_storage import init_db, SessionLocal


def find_storage_implementations() -> List[Tuple[str, str]]:
    """
    查找所有实现了 StorageInterface 的类及其文件路径。
    
    Returns:
        List[Tuple[str, str]]: 包含 (类名, 文件路径) 的列表
    """
    storage_classes = []
    storage_dir = Path("knowledge_distiller_kd/storage")
    
    for file_path in storage_dir.glob("*.py"):
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
            
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 检查类是否继承自 StorageInterface
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "StorageInterface":
                        storage_classes.append((node.name, str(file_path)))
                    elif isinstance(base, ast.Attribute) and base.attr == "StorageInterface":
                        storage_classes.append((node.name, str(file_path)))
    
    return storage_classes


def find_storage_instantiations() -> List[Tuple[str, str, int]]:
    """
    查找存储类的实例化位置。
    
    Returns:
        List[Tuple[str, str, int]]: 包含 (类名, 文件路径, 行号) 的列表
    """
    instantiations = []
    project_root = Path(".")
    
    for file_path in project_root.rglob("*.py"):
        if "venv" in str(file_path) or ".git" in str(file_path):
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content)
            
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["FileStorage", "ORMStorage"]:
                        instantiations.append((node.func.id, str(file_path), node.lineno))
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ["FileStorage", "ORMStorage"]:
                        instantiations.append((node.func.attr, str(file_path), node.lineno))
    
    return instantiations


def find_init_db_calls() -> List[Tuple[str, int]]:
    """
    查找所有对 init_db() 的调用点。
    
    Returns:
        List[Tuple[str, int]]: 包含 (文件路径, 行号) 的列表
    """
    init_db_calls = []
    project_root = Path(".")
    
    for file_path in project_root.rglob("*.py"):
        if "venv" in str(file_path) or ".git" in str(file_path):
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content)
            
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "init_db":
                    init_db_calls.append((str(file_path), node.lineno))
                elif isinstance(node.func, ast.Attribute) and node.func.attr == "init_db":
                    init_db_calls.append((str(file_path), node.lineno))
    
    return init_db_calls


def find_engine_storage_injections() -> List[Tuple[str, str, int]]:
    """
    查找 KnowledgeDistillerEngine 构造时注入 storage 的位置。
    
    Returns:
        List[Tuple[str, str, int]]: 包含 (存储类型, 文件路径, 行号) 的列表
    """
    injections = []
    project_root = Path(".")
    
    for file_path in project_root.rglob("*.py"):
        if "venv" in str(file_path) or ".git" in str(file_path):
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content)
            
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "KnowledgeDistillerEngine":
                    for keyword in node.keywords:
                        if keyword.arg == "storage":
                            if isinstance(keyword.value, ast.Name):
                                injections.append((keyword.value.id, str(file_path), node.lineno))
                            elif isinstance(keyword.value, ast.Call):
                                if isinstance(keyword.value.func, ast.Name):
                                    injections.append((keyword.value.func.id, str(file_path), node.lineno))
                                elif isinstance(keyword.value.func, ast.Attribute):
                                    injections.append((keyword.value.func.attr, str(file_path), node.lineno))
    
    return injections


def check_cli_storage_switching() -> bool:
    """
    检查 cli.py 中是否存在根据配置或参数动态选择存储后端的逻辑分支。
    
    Returns:
        bool: 是否存在动态存储切换逻辑
    """
    cli_path = Path("knowledge_distiller_kd/cli.py")
    if not cli_path.exists():
        return False
        
    with open(cli_path, "r", encoding="utf-8") as f:
        content = f.read()
        tree = ast.parse(content)
        
    # 检查是否存在条件语句
    has_conditions = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp)):
            # 检查条件是否与存储相关
            if isinstance(node.test, ast.Name):
                if node.test.id in ["use_sqlite", "use_file_storage", "storage_type"]:
                    has_conditions = True
                    break
            elif isinstance(node.test, ast.Compare):
                if isinstance(node.test.left, ast.Name):
                    if node.test.left.id in ["storage_type", "config.storage_type"]:
                        has_conditions = True
                        break
    
    return has_conditions


def generate_storage_usage_report() -> str:
    """
    生成存储使用情况报告。
    
    Returns:
        str: Markdown 格式的报告
    """
    # 1. 查找存储实现类
    storage_classes = find_storage_implementations()
    
    # 2. 查找存储实例化位置
    instantiations = find_storage_instantiations()
    
    # 3. 查找 init_db() 调用
    init_db_calls = find_init_db_calls()
    
    # 4. 查找 Engine 构造时的 storage 注入
    engine_injections = find_engine_storage_injections()
    
    # 5. 检查 CLI 中的存储切换逻辑
    has_storage_switching = check_cli_storage_switching()
    
    # 生成报告
    report = ["# 存储使用情况报告\n"]
    
    # 1. 存储实现类
    report.append("## 1. 存储实现类")
    if storage_classes:
        for class_name, file_path in storage_classes:
            report.append(f"- {class_name} ({file_path})")
    else:
        report.append("未找到存储实现类")
    
    # 2. 存储实例化位置
    report.append("\n## 2. 存储实例化位置")
    if instantiations:
        for class_name, file_path, line_no in instantiations:
            report.append(f"- {class_name} 在 {file_path}:{line_no}")
    else:
        report.append("未找到存储实例化")
    
    # 3. init_db() 调用
    report.append("\n## 3. init_db() 调用")
    if init_db_calls:
        for file_path, line_no in init_db_calls:
            report.append(f"- {file_path}:{line_no}")
    else:
        report.append("未找到 init_db() 调用")
    
    # 4. Engine 构造时的 storage 注入
    report.append("\n## 4. Engine 构造时的 storage 注入")
    if engine_injections:
        for storage_type, file_path, line_no in engine_injections:
            report.append(f"- 注入 {storage_type} 在 {file_path}:{line_no}")
    else:
        report.append("未找到 storage 注入")
    
    # 5. CLI 存储切换逻辑
    report.append("\n## 5. CLI 存储切换逻辑")
    if has_storage_switching:
        report.append("检测到动态存储切换逻辑")
    else:
        report.append("未检测到动态存储切换逻辑")
    
    return "\n".join(report)


def test_generate_storage_usage_report():
    """测试生成存储使用情况报告"""
    report = generate_storage_usage_report()
    
    # 验证报告格式
    assert report.startswith("# 存储使用情况报告")
    assert "## 1. 存储实现类" in report
    assert "## 2. 存储实例化位置" in report
    assert "## 3. init_db() 调用" in report
    assert "## 4. Engine 构造时的 storage 注入" in report
    assert "## 5. CLI 存储切换逻辑" in report
    
    # 验证存储实现类
    assert "FileStorage" in report
    
    # 验证存储实例化
    assert "cli.py" in report
    
    # 验证 init_db 调用
    assert "engine.py" in report
    
    # 验证 storage 注入
    assert "KnowledgeDistillerEngine" in report


if __name__ == "__main__":
    # 直接运行此文件时生成报告
    report = generate_storage_usage_report()
    print(report) 