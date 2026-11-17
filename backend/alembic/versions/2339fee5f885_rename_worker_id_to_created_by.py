"""rename worker id to created by

Revision ID: 2339fee5f885
Revises: 7a2f8f75714c
Create Date: 2026-11-15 15:27:30.902313

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2339fee5f885'
down_revision: Union[str, Sequence[str], None] = '7a2f8f75714c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "participants",
        "worker_id",
        new_column_name="created_by",
    )


def downgrade() -> None:
    op.alter_column(
        "participants",
        "created_by",
        new_column_name="worker_id",
    )