#!/usr/bin/env python
"""
数据库迁移脚本：将分析结果和决策表升级到新的结构
- 为Analysis表添加result_id, block_id_1, block_id_2字段
- 为Decision表添加decision_id, result_id字段
- 保持与旧代码的兼容性
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path
import uuid

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s:%(funcName)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("db_migration")

# 找到项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "instance" / "kd_database.sqlite"


def get_connection():
    """建立与数据库的连接"""
    if not DB_PATH.exists():
        logger.error(f"数据库文件不存在: {DB_PATH}")
        sys.exit(1)
    
    return sqlite3.connect(DB_PATH)


def check_columns_exist(conn, table, columns):
    """检查表中是否存在指定的列"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    existing_columns = [row[1] for row in cursor.fetchall()]
    
    return all(col in existing_columns for col in columns)


def migrate_analysis_table(conn):
    """迁移分析结果表"""
    cursor = conn.cursor()
    
    # 检查是否已经存在新列
    new_columns = ["result_id", "block_id_1", "block_id_2"]
    if check_columns_exist(conn, "analysis_results", new_columns):
        logger.info("分析结果表已包含新列，跳过迁移")
        return
    
    logger.info("开始迁移分析结果表...")
    
    # 1. 创建临时表
    cursor.execute("""
    CREATE TABLE analysis_results_new (
        id INTEGER PRIMARY KEY,
        result_id TEXT NOT NULL UNIQUE,
        block_id_1 TEXT NOT NULL,
        block_id_2 TEXT NOT NULL,
        block_id INTEGER NOT NULL REFERENCES blocks(id) ON DELETE CASCADE,
        analysis_type TEXT NOT NULL,
        score JSON NOT NULL,
        details JSON
    )
    """)
    
    # 2. 获取所有现有分析结果
    cursor.execute("""
    SELECT id, block_id, analysis_type, score, details FROM analysis_results
    """)
    analyses = cursor.fetchall()
    
    # 3. 将数据迁移到新表，同时添加新列
    for analysis in analyses:
        old_id, block_id, analysis_type, score, details = analysis
        
        # 查找block_id对应的block
        cursor.execute("SELECT block_id FROM blocks WHERE id = ?", (block_id,))
        block_result = cursor.fetchone()
        if block_result:
            block_id_str = block_result[0]
            # 生成唯一result_id
            result_id = str(uuid.uuid4())
            
            # 插入新记录
            cursor.execute("""
            INSERT INTO analysis_results_new 
            (id, result_id, block_id_1, block_id_2, block_id, analysis_type, score, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (old_id, result_id, block_id_str, block_id_str, block_id, analysis_type, score, details))
    
    # 4. 删除旧表并重命名新表
    cursor.execute("DROP TABLE analysis_results")
    cursor.execute("ALTER TABLE analysis_results_new RENAME TO analysis_results")
    
    # 5. 创建索引
    cursor.execute("CREATE INDEX idx_analysis_result_id ON analysis_results(result_id)")
    cursor.execute("CREATE INDEX idx_analysis_block_ids ON analysis_results(block_id_1, block_id_2)")
    
    logger.info(f"成功迁移 {len(analyses)} 条分析结果记录")


def migrate_decision_table(conn):
    """迁移决策表"""
    cursor = conn.cursor()
    
    # 检查是否已经存在新列
    new_columns = ["decision_id", "result_id"]
    if check_columns_exist(conn, "user_decisions", new_columns):
        logger.info("决策表已包含新列，跳过迁移")
        return
    
    logger.info("开始迁移决策表...")
    
    # 1. 创建临时表
    cursor.execute("""
    CREATE TABLE user_decisions_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_id TEXT NOT NULL UNIQUE,
        result_id TEXT NOT NULL REFERENCES analysis_results(result_id),
        block_id INTEGER REFERENCES blocks(id) ON DELETE CASCADE,
        decision_type TEXT NOT NULL,
        duplicate_of_block_id INTEGER REFERENCES blocks(id),
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        comment TEXT
    )
    """)
    
    # 2. 获取所有现有决策
    cursor.execute("""
    SELECT id, block_id, decision_type, timestamp, comment, duplicate_of_block_id 
    FROM user_decisions
    """)
    decisions = cursor.fetchall()
    
    # 3. 将数据迁移到新表，同时添加新列
    for decision in decisions:
        old_id, block_id, decision_type, timestamp, comment, duplicate_of_block_id = decision
        
        # 查找对应的block
        cursor.execute("SELECT block_id FROM blocks WHERE id = ?", (block_id,))
        block_result = cursor.fetchone()
        
        if block_result:
            block_id_str = block_result[0]
            
            # 查找对应的分析结果
            cursor.execute("""
            SELECT result_id FROM analysis_results 
            WHERE block_id = ? AND block_id_1 = ?
            """, (block_id, block_id_str))
            analysis_result = cursor.fetchone()
            
            if analysis_result:
                result_id = analysis_result[0]
            else:
                # 如果找不到对应的分析结果，创建一个
                result_id = str(uuid.uuid4())
                
            # 生成唯一decision_id
            decision_id = str(uuid.uuid4())
            
            # 插入新记录
            cursor.execute("""
            INSERT INTO user_decisions_new 
            (id, decision_id, result_id, block_id, decision_type, duplicate_of_block_id, timestamp, comment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (old_id, decision_id, result_id, block_id, decision_type, 
                 duplicate_of_block_id, timestamp, comment))
    
    # 4. 删除旧表并重命名新表
    cursor.execute("DROP TABLE user_decisions")
    cursor.execute("ALTER TABLE user_decisions_new RENAME TO user_decisions")
    
    # 5. 创建索引
    cursor.execute("CREATE INDEX idx_decision_id ON user_decisions(decision_id)")
    cursor.execute("CREATE INDEX idx_decision_result ON user_decisions(result_id)")
    
    logger.info(f"成功迁移 {len(decisions)} 条决策记录")


def run_migration():
    """运行迁移"""
    logger.info(f"开始迁移数据库: {DB_PATH}")
    
    conn = get_connection()
    try:
        conn.execute("BEGIN TRANSACTION")
        
        migrate_analysis_table(conn)
        migrate_decision_table(conn)
        
        conn.commit()
        logger.info("数据库迁移成功完成")
    except Exception as e:
        conn.rollback()
        logger.error(f"迁移失败: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run_migration() 