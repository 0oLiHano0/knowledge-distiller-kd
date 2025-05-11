"""init_tables

Revision ID: e5851bc0270a
Revises: 1a2b3c4d5e6f
Create Date: 2025-05-11 10:03:56.569196

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5851bc0270a'
down_revision: Union[str, None] = '1a2b3c4d5e6f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 这个迁移脚本不再创建表，因为表已经在前一个迁移中创建
    # 相反，我们添加一些有用的索引
    op.create_index('idx_documents_file_hash', 'documents', ['file_hash'])
    op.create_index('idx_blocks_content_hash', 'blocks', ['content_hash'])
    op.create_index('idx_blocks_document_id', 'blocks', ['document_id'])
    op.create_index('idx_analyses_block_id', 'analyses', ['block_id'])
    op.create_index('idx_decisions_block_id', 'decisions', ['block_id'])


def downgrade() -> None:
    """Downgrade schema."""
    # 删除添加的索引
    op.drop_index('idx_decisions_block_id', 'decisions')
    op.drop_index('idx_analyses_block_id', 'analyses')
    op.drop_index('idx_blocks_document_id', 'blocks')
    op.drop_index('idx_blocks_content_hash', 'blocks')
    op.drop_index('idx_documents_file_hash', 'documents')
