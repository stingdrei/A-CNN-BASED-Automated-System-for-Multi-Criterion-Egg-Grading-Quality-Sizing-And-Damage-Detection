"""Normalize legacy grade values to the locked contract.

Revision ID: 9f2b7c4d1e6a
Revises: 8e1f2a3b4c5d
"""

from typing import Sequence, Union

from alembic import op


revision: str = "9f2b7c4d1e6a"
down_revision: Union[str, None] = "8e1f2a3b4c5d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


ALLOWED_GRADES = ("A", "B", "C", "N/A", "Reject")


def upgrade() -> None:
    allowed = ", ".join(f"'{grade}'" for grade in ALLOWED_GRADES)
    op.execute(
        "UPDATE detection_boxes SET provisional_grade = CASE upper("
        "COALESCE(provisional_grade, grade)) "
        "WHEN 'AA' THEN 'A' "
        "WHEN 'A' THEN 'A' WHEN 'B' THEN 'B' WHEN 'C' THEN 'C' "
        "WHEN 'REJECT' THEN 'Reject' ELSE 'N/A' END"
    )
    op.execute(
        "UPDATE detection_boxes SET grade = 'N/A' "
        f"WHERE grade IS NULL OR grade = '' OR grade NOT IN ({allowed})"
    )
    op.execute(
        "UPDATE detection_boxes SET requires_review = true "
        "WHERE cleanliness_status = 'Unavailable'"
    )


def downgrade() -> None:
    # Data normalization is not reversed; the previous schema remains valid.
    pass
