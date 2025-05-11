"""drop_raw_element_type_from_blocks

Revision ID: f20a2b7cc75c
Revises: e5851bc0270a
Create Date: 2023-07-01 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import text

# revision identifiers, used by Alembic.
revision: str = 'f20a2b7cc75c'
down_revision: Union[str, None] = 'e5851bc0270a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: 从blocks表中移除raw_element_type列，先确保数据转移到block_type"""
    # 检查raw_element_type列是否存在
    conn = op.get_bind()
    
    # 检查sqlite数据库中blocks表是否存在raw_element_type列
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('blocks')]
    
    if 'raw_element_type' in columns:
        # 1. 首先确保block_type列存在数据，从raw_element_type转移过来
        conn.execute(text("""
            UPDATE blocks 
            SET block_type = raw_element_type 
            WHERE block_type IS NULL AND raw_element_type IS NOT NULL
        """))
        
        # 2. 删除raw_element_type列
        op.drop_column('blocks', 'raw_element_type')
    else:
        # 列不存在，跳过此步骤
        print("raw_element_type列不存在，跳过数据迁移和列删除操作")


def downgrade() -> None:
    """Downgrade schema: 恢复raw_element_type列"""
    # 1. 重新添加raw_element_type列
    op.add_column('blocks', sa.Column('raw_element_type', sa.String(), nullable=True))
    
    # 2. 将block_type的值回填到raw_element_type
    conn = op.get_bind()
    conn.execute(text("""
        UPDATE blocks 
        SET raw_element_type = block_type 
        WHERE block_type IS NOT NULL
    """)) 