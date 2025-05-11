"""
数据库初始化脚本，用于确保数据库表结构与模型定义一致。
该脚本将检查当前数据库结构，并自动应用必要的迁移。
"""

import os
import sys
import logging
import importlib.util
from pathlib import Path
from loguru import logger

# 引入项目相关模块
from knowledge_distiller_kd.storage.models_sqlalchemy import Base
from knowledge_distiller_kd.storage.sqlite_storage import engine

# 找到项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MIGRATIONS_DIR = PROJECT_ROOT / "migrations"
DB_PATH = PROJECT_ROOT / "instance" / "kd_database.sqlite"

def load_migration_script(script_path):
    """动态加载迁移脚本"""
    spec = importlib.util.spec_from_file_location("migration", script_path)
    migration_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration_module)
    return migration_module

def is_migration_needed():
    """检查是否需要进行数据库迁移"""
    # 如果数据库文件不存在，不需要迁移
    if not DB_PATH.exists():
        return False
    
    # 这里可以添加更复杂的检查逻辑，例如版本比较
    # 目前简单地检查是否有迁移脚本文件存在
    return any(MIGRATIONS_DIR.glob("*.py"))

def ensure_database_structure():
    """确保数据库结构与当前模型定义一致"""
    # 确保基本表结构存在
    Base.metadata.create_all(engine)
    
    # 检查是否需要运行迁移脚本
    if is_migration_needed():
        logger.info("检测到数据库需要迁移")
        
        # 按序号排序迁移脚本
        migration_scripts = sorted(MIGRATIONS_DIR.glob("*.py"))
        
        for script_path in migration_scripts:
            logger.info(f"运行迁移脚本: {script_path.name}")
            try:
                migration_module = load_migration_script(script_path)
                migration_module.run_migration()
            except Exception as e:
                logger.error(f"运行迁移脚本 {script_path.name} 时出错: {e}")
                raise
    else:
        logger.info("数据库结构已是最新，无需迁移")
    
    logger.info("数据库结构检查完成")

if __name__ == "__main__":
    ensure_database_structure() 