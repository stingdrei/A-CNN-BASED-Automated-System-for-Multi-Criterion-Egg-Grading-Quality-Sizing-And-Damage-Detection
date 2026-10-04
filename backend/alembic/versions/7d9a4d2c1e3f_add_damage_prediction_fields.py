"""Add damage inference fields to predictions.

Revision ID: 7d9a4d2c1e3f
Revises: 518703374959
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "7d9a4d2c1e3f"
down_revision: Union[str, None] = "518703374959"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "predictions",
        sa.Column("needs_review_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("detection_boxes", sa.Column("damage_status", sa.String(20)))
    op.add_column("detection_boxes", sa.Column("damage_confidence", sa.Float()))
    op.add_column(
        "detection_boxes",
        sa.Column("damage_model_version", sa.String(255)),
    )
    op.add_column("detection_boxes", sa.Column("review_reason", sa.String(100)))


def downgrade() -> None:
    op.drop_column("detection_boxes", "review_reason")
    op.drop_column("detection_boxes", "damage_model_version")
    op.drop_column("detection_boxes", "damage_confidence")
    op.drop_column("detection_boxes", "damage_status")
    op.drop_column("predictions", "needs_review_count")
