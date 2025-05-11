"""test_script

Revision ID: 52523ffecd0e
Revises: d36269929093
Create Date: 2025-05-11 12:14:02.594814

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '52523ffecd0e'
down_revision: Union[str, None] = 'd36269929093'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
