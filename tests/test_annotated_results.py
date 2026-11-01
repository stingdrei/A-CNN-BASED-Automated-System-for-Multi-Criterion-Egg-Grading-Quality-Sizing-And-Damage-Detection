import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.schemas import PredictionDetail


def _prediction(annotated_path):
    return SimpleNamespace(
        id=42,
        file_name="tray.jpg",
        file_type="image",
        tray_id="a" * 32,
        captured_at=datetime(2026, 11, 1, 10, 0, 0),
        operator="operator",
        inference_metadata={},
        status="completed",
        total_detections=1,
        damaged_count=0,
        not_damaged_count=1,
        needs_review_count=0,
        processing_time_ms=120,
        created_at=datetime(2026, 11, 1, 10, 0, 0),
        completed_at=datetime(2026, 11, 1, 10, 0, 1),
        result_json=None,
        annotated_path=annotated_path,
        detection_boxes=[],
    )


class AnnotatedResultTests(unittest.TestCase):
    def test_detail_exposes_download_url_when_annotation_exists(self):
        detail = PredictionDetail.model_validate(
            _prediction("/tmp/uploads/annotated_tray.jpg")
        ).model_dump()

        self.assertEqual(
            detail["annotated_url"], "/api/v1/predictions/42/download"
        )
        # The filesystem path stays internal; the URL is what the UI uses.
        self.assertEqual(
            detail["annotated_path"], "/tmp/uploads/annotated_tray.jpg"
        )

    def test_detail_has_no_url_without_annotation(self):
        detail = PredictionDetail.model_validate(_prediction(None)).model_dump()

        self.assertIsNone(detail["annotated_url"])


if __name__ == "__main__":
    unittest.main()
