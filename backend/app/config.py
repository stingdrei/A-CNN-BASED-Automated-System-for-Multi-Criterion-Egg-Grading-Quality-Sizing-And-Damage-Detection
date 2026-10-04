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
    DAMAGE_MODEL_PATH: str = str(PROJECT_ROOT / "models" / "egg_grader.pth")
    CONFIDENCE_THRESHOLD: float = 0.75
    DAMAGE_REVIEW_THRESHOLD: float = 0.75
    DETECTION_CROP_PADDING: float = 0.1
    MIN_CROP_SIZE: int = 8
    MM_PER_PIXEL: float = 0.09

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()