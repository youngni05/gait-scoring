"""Persist queued captures and their complete, reproducible report."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "cc61b325a13f"
down_revision = "b419d223e901"
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column("gait_measurements", "age", nullable=True)
    op.alter_column("gait_measurements", "measured_at", nullable=True)
    op.add_column("gait_measurements", sa.Column("analysis_status", sa.String(16)))
    op.add_column("gait_measurements", sa.Column("received_at", sa.DateTime(timezone=True)))
    op.add_column("gait_analysis_results", sa.Column("report_json", JSONB()))


def downgrade():
    # The former schema cannot represent an unknown capture date or age.
    # Refuse before removing data if such captures exist.
    if op.get_bind().scalar(sa.text("SELECT EXISTS (SELECT 1 FROM gait_measurements WHERE age IS NULL OR measured_at IS NULL)")):
        raise RuntimeError("Back up and resolve captures with unknown date or age before downgrading")
    op.drop_column("gait_analysis_results", "report_json")
    op.drop_column("gait_measurements", "analysis_status")
    op.drop_column("gait_measurements", "received_at")
    op.alter_column("gait_measurements", "measured_at", nullable=False)
    op.alter_column("gait_measurements", "age", nullable=False)
