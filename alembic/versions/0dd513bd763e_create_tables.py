"""create tables

Revision ID: 0dd513bd763e
Revises: a8f09560da3e
Create Date: 2025-05-11 09:00:38.333472

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON


# revision identifiers, used by Alembic.
revision: str = '0dd513bd763e'
down_revision: Union[str, None] = 'a8f09560da3e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema by creating all necessary tables."""
    # 创建 documents 表
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('path', sa.String, unique=True, nullable=False),
        sa.Column('file_hash', sa.String, nullable=False),
        sa.Column('type', sa.String),
        sa.Column('size', sa.Integer),
        sa.Column('ctime', sa.DateTime),
        sa.Column('mtime', sa.DateTime),
        sa.Column('ingest_time', sa.DateTime, server_default=sa.func.now()),
        sa.Column('status', sa.String),
    )
    
    # 创建 blocks 表
    op.create_table(
        'blocks',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('document_id', sa.Integer, sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('content_hash', sa.String),
        sa.Column('simhash', sa.String),
        sa.Column('text', sa.Text),
        sa.Column('raw_element_type', sa.String),
        sa.Column('processing_status', sa.String),
        sa.Column('meta_data', sa.JSON),
    )
    
    # 创建 analyses 表
    op.create_table(
        'analyses',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('block_id', sa.Integer, sa.ForeignKey('blocks.id', ondelete='CASCADE'), nullable=False),
        sa.Column('analysis_type', sa.String, nullable=False),
        sa.Column('score', sa.JSON),
        sa.Column('details', sa.JSON),
    )
    
    # 创建 decisions 表
    op.create_table(
        'decisions',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('block_id', sa.Integer, sa.ForeignKey('blocks.id', ondelete='CASCADE'), nullable=False),
        sa.Column('decision_type', sa.String, nullable=False),
        sa.Column('duplicate_of_block_id', sa.Integer, sa.ForeignKey('blocks.id')),
        sa.Column('timestamp', sa.DateTime, server_default=sa.func.now()),
        sa.Column('comment', sa.Text),
    )


def downgrade() -> None:
    """Downgrade schema by dropping all tables in reverse order."""
    op.drop_table('decisions')
    op.drop_table('analyses')
    op.drop_table('blocks')
    op.drop_table('documents')
