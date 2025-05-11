"""test_script

Revision ID: a8f09560da3e
Revises: 5c4ad02e95a2
Create Date: 2025-05-11 08:54:51.673850

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a8f09560da3e'
down_revision: Union[str, None] = '5c4ad02e95a2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
