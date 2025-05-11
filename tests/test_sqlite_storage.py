"""
测试SQLite存储模块的功能
"""

import os
import tempfile
import pytest
from sqlalchemy import inspect
from typing import Generator

# 导入需要测试的模块 - 更新导入路径
from knowledge_distiller_kd.storage.sqlite_storage import (
    init_db, 
    engine, 
    SessionLocal,
    get_database_url,
    ensure_db_directory
)
from knowledge_distiller_kd.core.models import Base


@pytest.fixture(scope="function")
def setup_test_env():
    """设置测试环境，使用内存数据库"""
    # 设置测试环境变量
    os.environ["TESTING"] = "1"
    yield
    # 清理环境变量
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


def test_get_database_url(setup_test_env):
    """测试获取数据库URL功能"""
    # 测试环境下应返回内存数据库URL
    assert "memory" in get_database_url()
    
    # 移除测试标记，应返回文件数据库URL
    del os.environ["TESTING"]
    assert "memory" not in get_database_url()
    
    # 恢复测试标记
    os.environ["TESTING"] = "1"


def test_ensure_db_directory(setup_test_env):
    """测试确保数据库目录存在的功能"""
    # 测试环境下不创建目录
    ensure_db_directory()
    # 手动改为文件数据库模式测试目录创建
    del os.environ["TESTING"]
    ensure_db_directory()
    from knowledge_distiller_kd.core.constants import DEFAULT_DB_DIR
    assert os.path.exists(DEFAULT_DB_DIR)
    # 恢复测试标记
    os.environ["TESTING"] = "1"


def test_session_creation(setup_test_env):
    """测试会话创建功能"""
    # 初始化数据库
    init_db()
    
    # 创建一个会话
    session = SessionLocal()
    # 验证会话已创建
    assert session is not None
    # 确认可以开始事务
    session.begin()
    # 回滚事务
    session.rollback()
    # 关闭会话
    session.close()


def test_init_db(setup_test_env):
    """测试数据库初始化功能，验证是否创建了所有定义的表"""
    # 调用初始化方法
    init_db()
    
    # 使用sqlalchemy的inspect检查表是否存在
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    # 验证核心表是否被创建
    required_tables = ["files", "blocks", "analysis_results", "user_decisions"]
    for table in required_tables:
        assert table in tables, f"表 {table} 未被创建"
    
    # 验证表结构（以blocks表为例）
    columns = {col["name"] for col in inspector.get_columns("blocks")}
    required_columns = {"id", "block_id", "file_id", "text", "block_type"}
    for column in required_columns:
        assert column in columns, f"blocks表缺少列 {column}" 