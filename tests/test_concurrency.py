import csv
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.damage_inference import DamageClassifier
from app.services import csv_service
from src.model_factory import build_model, checkpoint_metadata


class ClassifierInitRaceTests(unittest.TestCase):
    def test_concurrent_first_use_loads_checkpoint_once(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "damage.pth"
            model = build_model("custom", num_classes=2, pretrained=False)
            checkpoint = {
                "state_dict": model.state_dict(),
                "class_names": ["Damaged", "Not Damaged"],
                "img_size": 224,
            }
            checkpoint.update(
                checkpoint_metadata("custom", num_classes=2, pretrained=False)
            )
            torch.save(checkpoint, path)

            classifier = DamageClassifier(
                checkpoint_path=str(path), device="cpu"
            )
            barrier = threading.Barrier(8)
            original_load = torch.load
            load_calls = []

            def counting_load(*args, **kwargs):
                load_calls.append(1)
                return original_load(*args, **kwargs)

            errors = []

            def worker():
                try:
                    barrier.wait(timeout=5)
                    classifier._ensure_model()
                except Exception as exc:  # pragma: no cover - failure path
                    errors.append(exc)

            with patch("torch.load", side_effect=counting_load):
                threads = [threading.Thread(target=worker) for _ in range(8)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=30)

            self.assertEqual(errors, [])
            self.assertEqual(len(load_calls), 1)
            self.assertIsNotNone(classifier._transform)
            self.assertIsNotNone(classifier._model)


class CsvAppendRaceTests(unittest.TestCase):
    def test_concurrent_appends_write_one_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "eggs.csv"
            row = {field: "" for field in csv_service.CSV_FIELDS}
            row["prediction_id"] = "1"

            def fake_rows(prediction):
                return [row]

            barrier = threading.Barrier(12)

            def worker():
                barrier.wait(timeout=5)
                csv_service.append_prediction(
                    object(), path=str(destination)
                )

            with patch.object(csv_service, "rows_for_prediction", fake_rows):
                threads = [threading.Thread(target=worker) for _ in range(12)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=30)

            with destination.open(newline="") as handle:
                lines = list(csv.reader(handle))

            self.assertEqual(lines[0], csv_service.CSV_FIELDS)
            self.assertEqual(len(lines), 1 + 12)
            self.assertEqual(
                sum(1 for line in lines if line[:1] == ["timestamp"]), 1
            )


if __name__ == "__main__":
    unittest.main()
