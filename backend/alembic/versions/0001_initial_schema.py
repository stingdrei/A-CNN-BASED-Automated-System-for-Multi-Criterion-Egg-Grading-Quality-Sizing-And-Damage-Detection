"""Create the initial users, predictions, and detection box tables.

Revision ID: 0001_initial_schema
Revises:
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("username", sa.String(100), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column(
            "is_active", sa.Boolean(), nullable=False, server_default=sa.true()
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_username", "users", ["username"], unique=True)

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("file_name", sa.String(255), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("file_type", sa.String(20), nullable=False),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="processing",
        ),
        sa.Column(
            "total_detections",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "damaged_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "not_damaged_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("processing_time_ms", sa.Integer()),
        sa.Column("result_json", sa.JSON()),
        sa.Column("annotated_path", sa.Text()),
        sa.Column("error_message", sa.Text()),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime()),
    )
    op.create_index(
        "ix_predictions_user_id", "predictions", ["user_id"], unique=False
    )

    op.create_table(
        "detection_boxes",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "prediction_id",
            sa.Integer(),
            sa.ForeignKey("predictions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("class_name", sa.String(50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("x1", sa.Integer(), nullable=False),
        sa.Column("y1", sa.Integer(), nullable=False),
        sa.Column("x2", sa.Integer(), nullable=False),
        sa.Column("y2", sa.Integer(), nullable=False),
        sa.Column(
            "annotated", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
    )
    op.create_index(
        "ix_detection_boxes_prediction_id",
        "detection_boxes",
        ["prediction_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_detection_boxes_prediction_id", table_name="detection_boxes"
    )
    op.drop_table("detection_boxes")
    op.drop_index("ix_predictions_user_id", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
