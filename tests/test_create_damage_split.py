import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from create_damage_split import collect, remove_duplicates, split


class DamageSplitTests(unittest.TestCase):
    def test_collects_all_images_and_removes_cross_label_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for class_name, prefix in (("Damaged", "damaged"), ("Not Damaged", "not_damaged")):
                class_dir = root / class_name
                class_dir.mkdir()
                for index in range(10):
                    (class_dir / f"{prefix}_{index}.jpg").write_bytes(
                        f"{prefix}-{index}".encode()
                    )
            (root / "Damaged" / "duplicate.jpg").write_bytes(b"same")
            (root / "Not Damaged" / "duplicate.jpg").write_bytes(b"same")

            found = collect(root)
            cleaned, conflicts, same_label = remove_duplicates(found)
            train, val, test = split(cleaned, 0.15, 0.15, 42)

        self.assertEqual(len(found), 22)
        self.assertEqual(len(cleaned), 20)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["labels"], [0, 1])
        self.assertEqual(same_label, [])
        self.assertTrue(
            {"filename", "label", "source_id", "sha256"}.issubset(
                cleaned.columns
            )
        )
        self.assertEqual(len(train) + len(val) + len(test), 20)
        all_hashes = set(train["sha256"]) | set(val["sha256"]) | set(test["sha256"])
        self.assertEqual(len(all_hashes), 20)


if __name__ == "__main__":
    unittest.main()
