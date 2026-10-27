"""Add tray operator/metadata and persistence constraints.

Revision ID: a1b2c3d4e5f6
Revises: 9f2b7c4d1e6a
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "9f2b7c4d1e6a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("predictions", sa.Column("operator", sa.String(100)))
    op.add_column("predictions", sa.Column("inference_metadata", sa.JSON()))

    op.execute(
        "UPDATE predictions SET tray_id = 'legacy-' || id::text "
        "WHERE tray_id IS NULL"
    )
    op.execute(
        "UPDATE predictions SET operator = users.username "
        "FROM users WHERE predictions.user_id = users.id "
        "AND predictions.operator IS NULL"
    )
    op.alter_column("predictions", "tray_id", nullable=False)

    op.drop_index("ix_predictions_tray_id", table_name="predictions")
    op.create_index(
        "ix_predictions_tray_id",
        "predictions",
        ["tray_id"],
        unique=True,
    )

    op.create_check_constraint(
        "ck_predictions_status",
        "predictions",
        "status IN ('processing', 'completed', 'failed')",
    )
    op.create_check_constraint(
        "ck_detection_boxes_grade",
        "detection_boxes",
        "grade IS NULL OR grade IN ('AA', 'A', 'B', 'C', 'N/A', 'Reject')",
    )
    op.create_check_constraint(
        "ck_detection_boxes_provisional_grade",
        "detection_boxes",
        "provisional_grade IS NULL OR "
        "provisional_grade IN ('AA', 'A', 'B', 'C', 'N/A', 'Reject')",
    )
    op.create_check_constraint(
        "ck_detection_boxes_damage_status",
        "detection_boxes",
        "damage_status IS NULL OR "
        "damage_status IN ('Damaged', 'Not Damaged', 'Needs Review')",
    )
    op.create_check_constraint(
        "ck_detection_boxes_cleanliness_status",
        "detection_boxes",
        "cleanliness_status IS NULL OR cleanliness_status IN "
        "('Clean', 'Stained', 'Needs Review', 'Unavailable', 'Unknown')",
    )
    op.create_check_constraint(
        "ck_detection_boxes_size",
        "detection_boxes",
        "size IS NULL OR size IN ('Small', 'Medium', 'Large', 'Unknown')",
    )
    op.create_check_constraint(
        "ck_detection_boxes_confidence",
        "detection_boxes",
        "confidence >= 0 AND confidence <= 1",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_detection_boxes_confidence", "detection_boxes", type_="check"
    )
    op.drop_constraint("ck_detection_boxes_size", "detection_boxes", type_="check")
    op.drop_constraint(
        "ck_detection_boxes_cleanliness_status", "detection_boxes", type_="check"
    )
    op.drop_constraint(
        "ck_detection_boxes_damage_status", "detection_boxes", type_="check"
    )
    op.drop_constraint(
        "ck_detection_boxes_provisional_grade",
        "detection_boxes",
        type_="check",
    )
    op.drop_constraint("ck_detection_boxes_grade", "detection_boxes", type_="check")
    op.drop_constraint("ck_predictions_status", "predictions", type_="check")

    op.drop_index("ix_predictions_tray_id", table_name="predictions")
    op.create_index(
        "ix_predictions_tray_id",
        "predictions",
        ["tray_id"],
        unique=False,
    )
    op.alter_column("predictions", "tray_id", nullable=True)
    op.drop_column("predictions", "inference_metadata")
    op.drop_column("predictions", "operator")
