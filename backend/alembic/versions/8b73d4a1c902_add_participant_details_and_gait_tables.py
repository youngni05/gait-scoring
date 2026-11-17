"""Add nullable participant details and gait measurement/result tables.

Revision ID: 8b73d4a1c902
Revises: 2339fee5f885

Upgrade preserves existing users and participants. Downgrade removes the
new tables and columns, including any data subsequently stored in them.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "8b73d4a1c902"
down_revision: Union[str, Sequence[str], None] = "2339fee5f885"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("participants", sa.Column("gender", sa.String(1), nullable=True))
    op.add_column("participants", sa.Column("birth_date", sa.Date(), nullable=True))
    op.add_column("participants", sa.Column("height_cm", sa.Numeric(5, 2), nullable=True))
    op.create_check_constraint("ck_participants_gender", "participants", "gender IN ('M', 'F')")
    op.create_check_constraint("ck_participants_height_cm", "participants", "height_cm > 0")
    op.create_index("ix_participants_created_by", "participants", ["created_by"])

    op.create_table(
        "gait_measurements",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("participant_id", sa.UUID(), nullable=False),
        sa.Column("age", sa.SmallInteger(), nullable=False),
        sa.Column("keypoints_json", postgresql.JSONB(none_as_null=True), nullable=False),
        sa.Column("extractor_name", sa.String(100), nullable=False),
        sa.Column("extractor_version", sa.String(100), nullable=False),
        sa.Column("measured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.UUID(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["participant_id"], ["participants.id"]),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.CheckConstraint("age >= 0", name="ck_gait_measurements_age"),
    )
    op.create_index(
        "ix_gait_measurements_participant_measured_id",
        "gait_measurements", ["participant_id", "measured_at", "id"],
    )
    op.create_index("ix_gait_measurements_created_by", "gait_measurements", ["created_by"])

    op.create_table(
        "gait_analysis_results",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("measurement_id", sa.UUID(), nullable=False),
        sa.Column("gdi_score", sa.Double(), nullable=True),
        sa.Column("stride_length_cm", sa.Double(), nullable=True),
        sa.Column("cadence_spm", sa.Double(), nullable=True),
        sa.Column("hip_graph_json", postgresql.JSONB(none_as_null=True), nullable=True),
        sa.Column("knee_graph_json", postgresql.JSONB(none_as_null=True), nullable=True),
        sa.Column("ankle_graph_json", postgresql.JSONB(none_as_null=True), nullable=True),
        sa.Column("model_version", sa.String(100), nullable=False),
        sa.Column("analyzed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["measurement_id"], ["gait_measurements.id"]),
        sa.UniqueConstraint("measurement_id", name="uq_gait_analysis_results_measurement_id"),
    )


def downgrade() -> None:
    op.drop_table("gait_analysis_results")
    op.drop_index("ix_gait_measurements_created_by", table_name="gait_measurements")
    op.drop_index("ix_gait_measurements_participant_measured_id", table_name="gait_measurements")
    op.drop_table("gait_measurements")
    op.drop_index("ix_participants_created_by", table_name="participants")
    op.drop_constraint("ck_participants_height_cm", "participants", type_="check")
    op.drop_constraint("ck_participants_gender", "participants", type_="check")
    op.drop_column("participants", "height_cm")
    op.drop_column("participants", "birth_date")
    op.drop_column("participants", "gender")
