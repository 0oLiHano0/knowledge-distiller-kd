"""test_script

Revision ID: fa095d454662
Revises: f019133cfc6b
Create Date: 2025-05-11 22:38:49.081667

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fa095d454662'
down_revision: Union[str, None] = 'f019133cfc6b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
