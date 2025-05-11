"""
SQLite存储模块，提供数据库连接和会话管理功能。
"""

from pathlib import Path
import os
from typing import Optional, Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from knowledge_distiller_kd.core.constants import DATABASE_URL, DEFAULT_DB_DIR, TEST_DATABASE_URL
from knowledge_distiller_kd.storage.models_sqlalchemy import Base, Document, Block, Analysis, Decision


# 确定数据库URL，如果是测试环境则使用内存数据库
def get_database_url() -> str:
    """
    获取数据库URL，根据环境确定是使用生产数据库还是测试数据库。
    
    测试环境下使用内存数据库，生产环境使用文件数据库。
    
    Returns:
        str: 数据库URL
    """
    if os.environ.get("TESTING") == "1":
        return TEST_DATABASE_URL
    return DATABASE_URL


# 确保数据库目录存在
def ensure_db_directory() -> None:
    """
    确保数据库目录存在，如果不存在则创建。
    
    只有在非内存数据库模式下才会执行创建目录操作。
    """
    if "memory" not in get_database_url():
        db_dir = Path(DEFAULT_DB_DIR)
        db_dir.mkdir(exist_ok=True, parents=True)


# 初始化数据库引擎
ensure_db_directory()
engine = create_engine(
    get_database_url(),
    connect_args={"check_same_thread": False},
    echo=False  # 设置为True可以查看SQL语句执行情况，用于调试
)

# 创建会话工厂
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def init_db() -> None:
    """
    初始化数据库，创建所有定义的表。
    
    如果表已存在，则不会重复创建。
    """
    # 创建所有在Base中注册的表
    Base.metadata.create_all(bind=engine)


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