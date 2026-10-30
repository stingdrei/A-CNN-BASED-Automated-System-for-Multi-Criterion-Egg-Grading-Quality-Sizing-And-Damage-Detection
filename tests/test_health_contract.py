import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.routers import dashboard


class HealthContractTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_health_uses_request_scoped_session(self):
        # A leaked session (``next(get_db())``) was the prior bug; the DI
        # parameter guarantees FastAPI closes the session after the request.
        self.assertIn("db", inspect.signature(dashboard.health_check).parameters)

    def test_health_is_healthy_only_when_all_models_and_db_are_ready(self):
        with patch.object(dashboard.model_registry, "detector_loaded", True), \
                patch.object(dashboard.model_registry, "damage_loaded", True), \
                patch.object(dashboard.model_registry, "cleanliness_loaded", True):
            response = dashboard.health_check(db=self.db)
        self.assertEqual(response.status, "healthy")
        self.assertTrue(response.database_connected)
        self.assertTrue(response.cleanliness_model_loaded)

    def test_health_degrades_without_cleanliness_model(self):
        with patch.object(dashboard.model_registry, "detector_loaded", True), \
                patch.object(dashboard.model_registry, "damage_loaded", True), \
                patch.object(dashboard.model_registry, "cleanliness_loaded", False):
            response = dashboard.health_check(db=self.db)
        self.assertEqual(response.status, "degraded")
        self.assertFalse(response.cleanliness_model_loaded)


if __name__ == "__main__":
    unittest.main()
