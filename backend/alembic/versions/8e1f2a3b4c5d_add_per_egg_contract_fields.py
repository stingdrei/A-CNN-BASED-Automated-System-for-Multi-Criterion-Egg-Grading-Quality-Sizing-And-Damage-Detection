"""Add provisional per-egg contract fields.

Revision ID: 8e1f2a3b4c5d
Revises: 7d9a4d2c1e3f
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "8e1f2a3b4c5d"
down_revision: Union[str, None] = "7d9a4d2c1e3f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("predictions", sa.Column("tray_id", sa.String(64)))
    op.add_column("predictions", sa.Column("captured_at", sa.DateTime()))
    op.create_index(
        "ix_predictions_tray_id", "predictions", ["tray_id"], unique=False
    )

    op.alter_column(
        "detection_boxes",
        "size_category",
        new_column_name="size",
        existing_type=sa.String(20),
    )
    op.add_column("detection_boxes", sa.Column("egg_id", sa.Integer()))
    op.add_column("detection_boxes", sa.Column("length_mm", sa.Float()))
    op.add_column("detection_boxes", sa.Column("breadth_mm", sa.Float()))
    op.add_column(
        "detection_boxes", sa.Column("geometry_quality", sa.String(30))
    )
    op.add_column(
        "detection_boxes", sa.Column("provisional_grade", sa.String(10))
    )
    op.add_column(
        "detection_boxes",
        sa.Column(
            "requires_review",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "detection_boxes", sa.Column("cleanliness_status", sa.String(20))
    )
    op.add_column(
        "detection_boxes", sa.Column("cleanliness_confidence", sa.Float())
    )
    op.add_column(
        "detection_boxes",
        sa.Column("cleanliness_model_version", sa.String(255)),
    )
    op.add_column(
        "detection_boxes",
        sa.Column("detection_model_version", sa.String(255)),
    )
    op.add_column(
        "detection_boxes", sa.Column("calibration_version", sa.String(50))
    )
    op.alter_column(
        "detection_boxes",
        "review_reason",
        existing_type=sa.String(100),
        type_=sa.String(255),
        existing_nullable=True,
    )

    # Bring legacy rows onto the provisional contract: old A/B/C/AA grades
    # were computed without cleanliness, so they cannot remain final grades.
    op.execute(
        "UPDATE detection_boxes SET size = CASE lower(size) "
        "WHEN 'small' THEN 'Small' WHEN 'medium' THEN 'Medium' "
        "WHEN 'large' THEN 'Large' ELSE 'Unknown' END"
    )
    op.execute(
        "UPDATE detection_boxes SET provisional_grade = grade "
        "WHERE provisional_grade IS NULL"
    )
    op.execute(
        "UPDATE detection_boxes SET grade = 'N/A' "
        "WHERE grade IN ('A', 'B', 'C', 'AA')"
    )
    op.execute(
        "UPDATE detection_boxes SET cleanliness_status = 'Unavailable', "
        "geometry_quality = COALESCE(geometry_quality, 'unknown'), "
        "calibration_version = COALESCE(calibration_version, 'uncalibrated'), "
        "requires_review = true"
    )


def downgrade() -> None:
    op.alter_column(
        "detection_boxes",
        "review_reason",
        existing_type=sa.String(255),
        type_=sa.String(100),
        existing_nullable=True,
    )
    op.drop_column("detection_boxes", "calibration_version")
    op.drop_column("detection_boxes", "detection_model_version")
    op.drop_column("detection_boxes", "cleanliness_model_version")
    op.drop_column("detection_boxes", "cleanliness_confidence")
    op.drop_column("detection_boxes", "cleanliness_status")
    op.drop_column("detection_boxes", "requires_review")
    op.drop_column("detection_boxes", "provisional_grade")
    op.drop_column("detection_boxes", "geometry_quality")
    op.drop_column("detection_boxes", "breadth_mm")
    op.drop_column("detection_boxes", "length_mm")
    op.drop_column("detection_boxes", "egg_id")
    op.alter_column(
        "detection_boxes",
        "size",
        new_column_name="size_category",
        existing_type=sa.String(20),
    )

    op.drop_index("ix_predictions_tray_id", table_name="predictions")
    op.drop_column("predictions", "captured_at")
    op.drop_column("predictions", "tray_id")
