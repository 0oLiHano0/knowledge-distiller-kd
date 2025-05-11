"""test_script

Revision ID: 39d86b663373
Revises: 0dd513bd763e
Create Date: 2025-05-11 08:56:06.601349

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '39d86b663373'
down_revision: Union[str, None] = '0dd513bd763e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
