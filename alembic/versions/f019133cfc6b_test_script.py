"""test_script

Revision ID: f019133cfc6b
Revises: a7741eda842b
Create Date: 2025-05-11 22:33:03.242859

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f019133cfc6b'
down_revision: Union[str, None] = 'a7741eda842b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
