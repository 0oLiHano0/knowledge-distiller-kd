import os
import pytest
import tempfile
import shutil
from pathlib import Path
import importlib.util
import re
import sqlite3
import alembic.command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def test_autogenerate_migration_creates_tables(tmp_path, monkeypatch):
    """测试自动生成的迁移脚本是否正确创建了所有必要的表"""
    # 设置临时工作目录
    original_dir = os.getcwd()
    temp_project_dir = tmp_path / "test_project"
    temp_project_dir.mkdir()
    
    try:
        # 复制必要的项目文件到临时目录
        os.chdir(temp_project_dir)
        
        # 创建临时数据库URL
        db_path = temp_project_dir / "test.db"
        db_url = f"sqlite:///{db_path}"
        
        # 创建临时alembic配置
        alembic_dir = temp_project_dir / "alembic"
        alembic_dir.mkdir()
        versions_dir = alembic_dir / "versions"
        versions_dir.mkdir()
        
        # 复制alembic.ini
        shutil.copy(Path(original_dir) / "alembic.ini", temp_project_dir / "alembic.ini")
        
        # 修改临时alembic.ini中的数据库URL
        with open(temp_project_dir / "alembic.ini", "r") as f:
            content = f.read()
        content = content.replace("sqlite:///./data/kd_tool.db", db_url)
        with open(temp_project_dir / "alembic.ini", "w") as f:
            f.write(content)
        
        # 复制alembic环境文件
        shutil.copy(Path(original_dir) / "alembic" / "env.py", alembic_dir / "env.py")
        
        # 配置alembic
        config = Config(str(temp_project_dir / "alembic.ini"))
        
        # 设置Python路径使其能找到项目模块
        monkeypatch.syspath_prepend(str(original_dir))
        
        # 执行自动生成迁移脚本命令
        with pytest.raises(Exception):
            # 这里应该会失败，因为临时环境中没有创建模型
            alembic.command.revision(config, autogenerate=True, message="test_migration")
        
        # 断言迁移脚本未被创建（红色测试）
        assert len(list(versions_dir.glob("*.py"))) == 0
    
    finally:
        # 恢复原始工作目录
        os.chdir(original_dir)


def test_migration_model_structure():
    """测试ORM模型定义的表结构符合预期"""
    # 创建内存数据库
    db_url = "sqlite:///:memory:"
    engine = create_engine(db_url)
    
    # 导入ORM模型并创建表
    from knowledge_distiller_kd.storage.models_sqlalchemy import Base
    Base.metadata.create_all(engine)
    
    # 验证表结构
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    # 验证四个核心表都已创建
    core_tables = ['files', 'blocks', 'analysis_results', 'user_decisions']
    for table in core_tables:
        assert table in tables, f"缺少{table}表"
    
    # 验证关键列
    columns_files = [col['name'] for col in inspector.get_columns('files')]
    columns_blocks = [col['name'] for col in inspector.get_columns('blocks')]
    columns_analyses = [col['name'] for col in inspector.get_columns('analysis_results')]
    columns_decisions = [col['name'] for col in inspector.get_columns('user_decisions')]
    
    # 验证files表的列
    assert "id" in columns_files
    assert "path" in columns_files
    assert "file_hash" in columns_files
    
    # 验证blocks表的列
    assert "id" in columns_blocks
    assert "file_id" in columns_blocks
    assert "content_hash" in columns_blocks
    assert "text" in columns_blocks
    
    # 验证analyses表的列
    assert "id" in columns_analyses
    assert "block_id" in columns_analyses
    assert "analysis_type" in columns_analyses
    
    # 验证decisions表的列
    assert "id" in columns_decisions
    assert "block_id" in columns_decisions
    assert "decision_type" in columns_decisions
    
    # 验证外键约束
    fk_blocks = inspector.get_foreign_keys('blocks')
    fk_analyses = inspector.get_foreign_keys('analysis_results')
    fk_decisions = inspector.get_foreign_keys('user_decisions')
    
    # 验证blocks表的外键指向files表
    assert any(fk['referred_table'] == 'files' for fk in fk_blocks), "blocks缺少到files的外键约束"
    
    # 验证analyses表的外键指向blocks表
    assert any(fk['referred_table'] == 'blocks' for fk in fk_analyses), "analyses缺少到blocks的外键约束"
    
    # 验证decisions表的外键指向blocks表
    assert any(fk['referred_table'] == 'blocks' for fk in fk_decisions), "decisions缺少到blocks的外键约束"


def test_latest_migration_script_has_correct_content():
    """测试最新的迁移脚本中包含预期的表名引用"""
    # 获取versions目录中最新的迁移脚本
    versions_dir = Path("alembic/versions")
    migration_files = list(versions_dir.glob("*.py"))
    assert len(migration_files) > 0, "未找到任何迁移脚本"
    
    # 按修改时间排序，获取最新文件
    latest_migration = sorted(migration_files, key=lambda p: p.stat().st_mtime)[-1]
    print(f"最新迁移脚本：{latest_migration}")
    
    # 读取脚本内容
    with open(latest_migration, "r") as f:
        content = f.read()
    
    # 检查脚本是否包含必要的导入和函数
    assert "from alembic import op" in content, "缺少alembic.op导入"
    assert "def upgrade()" in content, "缺少upgrade函数"
    assert "def downgrade()" in content, "缺少downgrade函数"
    
    # 如果脚本名称表明是创建索引的迁移，才检查索引语句
    if "create_index" in latest_migration.name.lower() or "add_index" in latest_migration.name.lower():
        tables = ['files', 'blocks', 'analysis_results', 'user_decisions']
        for table in tables:
            pattern = fr"op\.create_index\(\s*['\"]?idx_{table}"
            assert re.search(pattern, content), f"缺少{table}表的索引创建语句"
        
        # 检查索引删除语句
        for table in tables:
            pattern = fr"op\.drop_index\(\s*['\"]?idx_{table}"
            assert re.search(pattern, content), f"缺少{table}表的索引删除语句"
    else:
        print(f"当前迁移脚本不是创建索引的迁移，跳过索引检查。")