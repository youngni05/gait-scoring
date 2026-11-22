"""Store registration time for new participants without backfilling old rows."""
from alembic import op
import sqlalchemy as sa

revision = "e83b7c201a64"
down_revision = "a9310e8c6f02"
branch_labels = None
depends_on = None


def upgrade():
    # Adding the default with the column would stamp all existing rows as today.
    op.add_column("participants", sa.Column("created_at", sa.DateTime(timezone=True), nullable=True))
    op.alter_column("participants", "created_at", server_default=sa.text("now()"))


def downgrade():
    op.drop_column("participants", "created_at")
