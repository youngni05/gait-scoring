"""Keep a corrected capture day and its age snapshot alongside the original recording."""
from alembic import op
import sqlalchemy as sa

revision = "f4c831ad620e"
down_revision = "a9310e8c6f02"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("gait_measurements", sa.Column("corrected_capture_at", sa.DateTime(timezone=True)))
    op.add_column("gait_measurements", sa.Column("corrected_capture_age", sa.SmallInteger()))


def downgrade():
    op.drop_column("gait_measurements", "corrected_capture_age")
    op.drop_column("gait_measurements", "corrected_capture_at")
