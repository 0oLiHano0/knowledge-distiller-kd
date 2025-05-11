"""test_script

Revision ID: b20a2b7cc75b
Revises: e5851bc0270a
Create Date: 2025-05-11 10:59:48.431101

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b20a2b7cc75b'
down_revision: Union[str, None] = 'e5851bc0270a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
