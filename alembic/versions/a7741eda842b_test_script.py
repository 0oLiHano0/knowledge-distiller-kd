"""test_script

Revision ID: a7741eda842b
Revises: 07eacad5eb72
Create Date: 2025-05-11 22:31:49.509124

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a7741eda842b'
down_revision: Union[str, None] = '07eacad5eb72'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
