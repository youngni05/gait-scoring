"""Add nullable processing metadata and transactional request deduplication.

Upgrade preserves existing data. Downgrade removes new metadata and request keys.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "b419d223e901"
down_revision = "8b73d4a1c902"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("gait_measurements", sa.Column("processing_metadata", postgresql.JSONB(none_as_null=True), nullable=True))
    op.create_table(
        "measurement_requests",
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id"), primary_key=True),
        sa.Column("request_key", sa.UUID(), primary_key=True),
        sa.Column("participant_id", sa.UUID(), sa.ForeignKey("participants.id"), nullable=False),
        sa.Column("body_sha256", sa.String(64), nullable=False),
        sa.Column("measurement_id", sa.UUID(), sa.ForeignKey("gait_measurements.id"), nullable=False),
    )


def downgrade():
    op.drop_table("measurement_requests")
    op.drop_column("gait_measurements", "processing_metadata")
