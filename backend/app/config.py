from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_DEV_SECRET = "your-secret-key-change-in-production-min-32-chars-long"


class Settings(BaseSettings):
    APP_NAME: str = "Egg-CV API"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    VERSION: str = "1.0.0"

    DATABASE_URL: str = "postgresql://wii@localhost/eggcvdatabase"

    SECRET_KEY: str = DEFAULT_DEV_SECRET
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    CORS_ORIGINS: str = (
        "http://localhost:5173,http://localhost:3000,http://localhost:5174"
    )

    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE_MB: int = 50

    MODEL_PATH: str = str(
        PROJECT_ROOT / "egg_detection" / "finetune_egg_v1" / "weights" / "best.pt"
    )
    DAMAGE_MODEL_PATH: str = str(
        PROJECT_ROOT / "models" / "egg_grader_crop_resnet18_v1.pth"
    )
    EGG_CSV_PATH: str = str(PROJECT_ROOT / "reports" / "egg_statistics_v2.csv")
    CONFIDENCE_THRESHOLD: float = 0.75
    DAMAGE_DECISION_THRESHOLD: float = 0.18
    DAMAGE_REVIEW_THRESHOLD: float = 0.75
    CLEANLINESS_MODEL_PATH: str = str(
        PROJECT_ROOT / "models" / "egg_grader_cleanliness_v1.pth"
    )
    CLEANLINESS_DECISION_THRESHOLD: float = 0.5
    CLEANLINESS_REVIEW_THRESHOLD: float = 0.75
    DETECTION_CROP_PADDING: float = 0.1
    MIN_CROP_SIZE: int = 8
    MAX_TRAY_EGGS: int = 5
    DETECTION_DUPLICATE_IOU: float = 0.85
    MM_PER_PIXEL: float = 0.09
    # Fitted coefficient for W = k * L * B^2; 0.0 means not yet calibrated.
    WEIGHT_K: float = 0.0
    CALIBRATION_VERSION: str = "uncalibrated"

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.CORS_ORIGINS.split(",")
            if origin.strip()
        ]

    @model_validator(mode="after")
    def validate_production_security(self):
        if self.ENVIRONMENT != "production":
            return self
        if self.SECRET_KEY == DEFAULT_DEV_SECRET or len(self.SECRET_KEY) < 32:
            raise ValueError(
                "SECRET_KEY must be set to a unique value of at least 32 "
                "characters in production"
            )
        if self.DEBUG:
            raise ValueError("DEBUG must be false in production")
        if "*" in self.cors_origins:
            raise ValueError("Wildcard CORS origin is not allowed in production")
        if "localhost" in self.DATABASE_URL or "127.0.0.1" in self.DATABASE_URL:
            raise ValueError(
                "DATABASE_URL points at localhost in production; set the "
                "deployment database URL"
            )
        return self

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()