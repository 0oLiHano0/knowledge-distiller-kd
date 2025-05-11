import os
import pytest
import subprocess
import sqlite3
from pathlib import Path
import shutil

from sqlalchemy import inspect, create_engine

from knowledge_distiller_kd.core.constants import TEST_DATABASE_URL


@pytest.fixture(scope="module")
def alembic_config_exists():
    """检查alembic配置文件是否存在"""
    alembic_ini = Path("alembic.ini")
    env_py = Path("alembic/env.py")
    versions_dir = Path("alembic/versions")
    
    # 确保文件和目录存在
    assert alembic_ini.exists(), "alembic.ini 文件不存在"
    assert env_py.exists(), "alembic/env.py 文件不存在"
    assert versions_dir.exists(), "alembic/versions 目录不存在"
    
    # 读取配置文件内容进行验证
    with open(alembic_ini, "r") as f:
        config_content = f.read()
    
    with open(env_py, "r") as f:
        env_content = f.read()
        
    yield {
        "config_content": config_content,
        "env_content": env_content
    }


@pytest.fixture(scope="function")
def test_db_path():
    """创建测试用的独立数据库文件"""
    # 使用临时数据库文件
    test_db_path = "data/test_kd_tool.db"
    
    # 如果文件已存在，则先删除
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(test_db_path), exist_ok=True)
    
    yield test_db_path
    
    # 测试后清理
    if os.path.exists(test_db_path):
        os.remove(test_db_path)


def test_alembic_config_content(alembic_config_exists):
    """测试alembic配置文件内容是否正确"""
    config_content = alembic_config_exists["config_content"]
    env_content = alembic_config_exists["env_content"]
    
    # 检查alembic.ini配置
    assert "sqlalchemy.url" in config_content, "sqlalchemy.url 配置缺失"
    
    # 检查env.py中是否正确导入了模型
    assert "from knowledge_distiller_kd.storage.models_sqlalchemy import Base" in env_content, "未找到Base模型导入"
    assert "target_metadata = Base.metadata" in env_content, "未设置target_metadata为Base.metadata"


def test_alembic_revision_create():
    """测试alembic能否创建新迁移脚本"""
    versions_dir = Path("alembic/versions")
    
    # 确保versions目录存在
    if not versions_dir.exists():
        versions_dir.mkdir(exist_ok=True)
    
    # 获取当前脚本文件数
    original_scripts = list(versions_dir.glob("*.py"))
    
    try:
        # 创建一个测试版本脚本
        result = subprocess.run(
            ["alembic", "revision", "-m", "test_script"], 
            check=True, 
            capture_output=True, 
            text=True
        )
        assert result.returncode == 0, f"创建迁移脚本失败: {result.stderr}"
        
        # 检查是否生成了新脚本文件
        new_scripts = list(versions_dir.glob("*.py"))
        assert len(new_scripts) > len(original_scripts), "未生成新的迁移脚本文件"
        
    except subprocess.CalledProcessError as e:
        pytest.fail(f"alembic revision 命令执行出错: {e}")


def test_sqlite_db_creation_with_alembic(test_db_path):
    """测试使用Alembic创建SQLite数据库"""
    # 创建一个临时的alembic.ini文件用于测试
    temp_alembic_ini = "alembic.test.ini"
    shutil.copy("alembic.ini", temp_alembic_ini)
    
    try:
        # 修改临时配置文件使用测试数据库
        with open(temp_alembic_ini, "r") as f:
            content = f.read()
        
        content = content.replace(
            "sqlalchemy.url = sqlite:///./data/kd_tool.db",
            f"sqlalchemy.url = sqlite:///./{test_db_path}"
        )
        
        with open(temp_alembic_ini, "w") as f:
            f.write(content)
        
        # 使用临时配置文件执行数据库迁移
        result = subprocess.run(
            ["alembic", "-c", temp_alembic_ini, "upgrade", "head"], 
            check=True, 
            capture_output=True, 
            text=True
        )
        assert result.returncode == 0, f"数据库迁移失败: {result.stderr}"
        
        # 验证表是否被创建
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        # 查询所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        # 验证表是否存在
        for table_name in ["documents", "blocks", "analyses", "decisions", "alembic_version"]:
            assert table_name in tables, f"数据库中缺少表 {table_name}"
        
        conn.close()
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_alembic_ini):
            os.remove(temp_alembic_ini) 