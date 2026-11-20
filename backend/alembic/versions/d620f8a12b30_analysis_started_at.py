"""Store analysis start separately; never relabel historical capture dates."""
from alembic import op
import sqlalchemy as sa

revision = "d620f8a12b30"
down_revision = "b419d223e901"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("gait_measurements", sa.Column("analysis_started_at", sa.DateTime(timezone=True), nullable=True))
    op.alter_column("gait_measurements", "measured_at", nullable=True)


def downgrade():
    # A downgrade must not silently turn analysis dates into capture dates.
    if op.get_bind().execute(sa.text("SELECT EXISTS (SELECT 1 FROM gait_measurements WHERE analysis_started_at IS NOT NULL)")).scalar():
        raise RuntimeError("Cannot downgrade while analysis-start records exist; preserve/export them first")
    op.alter_column("gait_measurements", "measured_at", nullable=False)
    op.drop_column("gait_measurements", "analysis_started_at")
