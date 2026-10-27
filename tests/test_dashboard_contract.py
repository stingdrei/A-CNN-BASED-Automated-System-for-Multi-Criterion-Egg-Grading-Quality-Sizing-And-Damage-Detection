import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.database import Base
from app.models import DetectionBox, Prediction, User
from app.routers.dashboard import get_grade_distribution


class DashboardContractTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        self.user = User(
            email="dashboard@example.com",
            username="dashboard",
            password_hash="x",
            is_active=True,
        )
        self.db.add(self.user)
        self.db.commit()
        self.db.refresh(self.user)

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _prediction(self, user_id):
        prediction = Prediction(
            user_id=user_id,
            file_name="tray.jpg",
            file_path="/tmp/tray.jpg",
            file_type="image",
            status="completed",
        )
        self.db.add(prediction)
        self.db.commit()
        self.db.refresh(prediction)
        return prediction

    def _box(self, prediction_id, grade):
        self.db.add(
            DetectionBox(
                prediction_id=prediction_id,
                class_name="egg",
                confidence=0.9,
                x1=0,
                y1=0,
                x2=10,
                y2=10,
                grade=grade,
            )
        )

    def test_grade_distribution_accumulates_aa_into_a(self):
        prediction = self._prediction(self.user.id)
        for grade in ("A", "AA", "A", "B", "C", "N/A", "Reject"):
            self._box(prediction.id, grade)
        self.db.commit()

        result = get_grade_distribution(db=self.db, current_user=self.user)

        self.assertEqual(result.grade_a, 3)
        self.assertEqual(result.grade_b, 1)
        self.assertEqual(result.grade_c, 1)
        self.assertEqual(result.grade_na, 1)
        self.assertEqual(result.grade_reject, 1)
        self.assertEqual(result.total, 7)

    def test_grade_distribution_excludes_other_users(self):
        other = User(
            email="other@example.com",
            username="other",
            password_hash="x",
            is_active=True,
        )
        self.db.add(other)
        self.db.commit()
        self.db.refresh(other)
        own = self._prediction(self.user.id)
        foreign = self._prediction(other.id)
        self._box(own.id, "A")
        self._box(foreign.id, "Reject")
        self.db.commit()

        result = get_grade_distribution(db=self.db, current_user=self.user)

        self.assertEqual(result.grade_a, 1)
        self.assertEqual(result.grade_reject, 0)
        self.assertEqual(result.total, 1)


if __name__ == "__main__":
    unittest.main()
