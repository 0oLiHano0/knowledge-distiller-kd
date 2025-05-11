"""test_script

Revision ID: 07eacad5eb72
Revises: c4ec342054d9
Create Date: 2025-05-11 22:30:12.186708

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '07eacad5eb72'
down_revision: Union[str, None] = 'c4ec342054d9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
