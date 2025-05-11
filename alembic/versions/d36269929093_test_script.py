"""test_script

Revision ID: d36269929093
Revises: df077b6e50ee
Create Date: 2025-05-11 12:13:47.156380

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd36269929093'
down_revision: Union[str, None] = 'df077b6e50ee'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
