import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import DEFAULT_DEV_SECRET, Settings


def production_settings(**overrides):
    values = {
        "_env_file": None,
        "ENVIRONMENT": "production",
        "DEBUG": False,
        "SECRET_KEY": "a" * 32,
        "DATABASE_URL": "postgresql://eggs:secret@db.example/eggs",
        "CORS_ORIGINS": "https://eggs.example",
    }
    values.update(overrides)
    return Settings(**values)


class ConfigSecurityTests(unittest.TestCase):
    def test_production_accepts_hardened_settings(self):
        settings = production_settings()
        self.assertEqual(settings.cors_origins, ["https://eggs.example"])

    def test_production_rejects_default_secret(self):
        with self.assertRaises(ValidationError):
            production_settings(SECRET_KEY=DEFAULT_DEV_SECRET)

    def test_production_rejects_short_secret(self):
        with self.assertRaises(ValidationError):
            production_settings(SECRET_KEY="short")

    def test_production_rejects_debug(self):
        with self.assertRaises(ValidationError):
            production_settings(DEBUG=True)

    def test_production_rejects_wildcard_cors(self):
        with self.assertRaises(ValidationError):
            production_settings(CORS_ORIGINS="*")

    def test_production_rejects_localhost_database(self):
        with self.assertRaises(ValidationError):
            production_settings(
                DATABASE_URL="postgresql://wii@localhost/eggcvdatabase"
            )

    def test_cors_origins_trims_and_drops_empty_entries(self):
        settings = production_settings(
            CORS_ORIGINS=" https://a.example, ,https://b.example "
        )
        self.assertEqual(
            settings.cors_origins, ["https://a.example", "https://b.example"]
        )


if __name__ == "__main__":
    unittest.main()
