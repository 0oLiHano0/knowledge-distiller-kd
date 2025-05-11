"""test_script

Revision ID: 62b49c991b50
Revises: 52523ffecd0e
Create Date: 2025-05-11 12:14:13.667207

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '62b49c991b50'
down_revision: Union[str, None] = '52523ffecd0e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
