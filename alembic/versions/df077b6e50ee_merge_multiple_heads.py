"""merge multiple heads

Revision ID: df077b6e50ee
Revises: b20a2b7cc75b, f20a2b7cc75c
Create Date: 2025-05-11 12:12:56.084572

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'df077b6e50ee'
down_revision: Union[str, None] = ('b20a2b7cc75b', 'f20a2b7cc75c')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
