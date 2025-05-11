"""
SQLite存储模块，提供数据库连接和会话管理功能。
"""

from pathlib import Path
import os
from typing import Optional, Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import QueuePool

from knowledge_distiller_kd.core.constants import DEFAULT_DB_DIR, TEST_DATABASE_URL
from knowledge_distiller_kd.storage.models_sqlalchemy import Base, Document, Block, Analysis, Decision

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 创建SQLite数据库文件存储目录
INSTANCE_DIR = PROJECT_ROOT / "instance"

def ensure_db_directory():
    """
    确保数据库目录存在
    
    如果是测试环境，则使用内存数据库，不需要创建目录
    """
    if "TESTING" in os.environ:
        return
    
    os.makedirs(INSTANCE_DIR, exist_ok=True)

def get_database_url() -> str:
    """
    获取数据库URL
    
    如果是测试环境，返回内存数据库URL
    否则返回文件数据库URL
    
    Returns:
        str: 数据库URL
    """
    if "TESTING" in os.environ:
        return TEST_DATABASE_URL
    
    return f"sqlite:///{INSTANCE_DIR}/kd_database.sqlite"

# 确保数据库目录存在
ensure_db_directory()

# 数据库URL
DATABASE_URL = get_database_url()

# 创建engine，启用外键约束
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=QueuePool,
    pool_pre_ping=True,
    pool_recycle=3600
)

# 创建session工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """
    初始化数据库，创建所有表
    如果数据库结构与模型定义不一致，会自动应用迁移
    """
    # 创建表结构
    Base.metadata.create_all(bind=engine)
    
    # 延迟导入以避免循环引用
    try:
        # 使用标志变量来防止递归调用
        if not getattr(init_db, "_is_running", False):
            init_db._is_running = True
            try:
                from knowledge_distiller_kd.storage.utils.db_init import ensure_database_structure
                ensure_database_structure()
            finally:
                init_db._is_running = False
    except ImportError:
        # 如果还没有实现迁移功能，只创建表结构
        pass

# 初始化标志
init_db._is_running = False

def get_db() -> Generator[Session, None, None]:
    """
    创建一个数据库会话，用完后自动关闭。
    
    主要用于依赖注入模式，确保会话在使用后被正确关闭。
    
    Yields:
        Session: SQLAlchemy会话对象
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 