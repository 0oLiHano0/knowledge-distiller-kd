"""test_script

Revision ID: c4ec342054d9
Revises: 62b49c991b50
Create Date: 2025-05-11 12:55:24.083554

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4ec342054d9'
down_revision: Union[str, None] = '62b49c991b50'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
