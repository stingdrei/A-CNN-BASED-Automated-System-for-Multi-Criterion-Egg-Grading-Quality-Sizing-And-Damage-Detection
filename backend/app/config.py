from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    APP_NAME: str = "Egg-CV API"
    DEBUG: bool = True
    VERSION: str = "1.0.0"

    DATABASE_URL: str = "postgresql://wii@localhost/eggcvdatabase"

    SECRET_KEY: str = "your-secret-key-change-in-production-min-32-chars-long"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

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
    DAMAGE_DECISION_THRESHOLD: float = 0.5
    DAMAGE_REVIEW_THRESHOLD: float = 0.75
    CLEANLINESS_MODEL_PATH: str = str(
        PROJECT_ROOT / "models" / "egg_grader_cleanliness_v1.pth"
    )
    CLEANLINESS_DECISION_THRESHOLD: float = 0.5
    CLEANLINESS_REVIEW_THRESHOLD: float = 0.75
    DETECTION_CROP_PADDING: float = 0.1
    MIN_CROP_SIZE: int = 8
    MM_PER_PIXEL: float = 0.09
    # Fitted coefficient for W = k * L * B^2; 0.0 means not yet calibrated.
    WEIGHT_K: float = 0.0
    CALIBRATION_VERSION: str = "uncalibrated"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()