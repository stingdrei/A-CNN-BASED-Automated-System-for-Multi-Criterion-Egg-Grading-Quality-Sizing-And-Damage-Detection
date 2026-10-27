from pydantic import BaseModel, ConfigDict, EmailStr, Field, computed_field
from typing import Optional, List
from datetime import datetime


class UserBase(BaseModel):
    email: EmailStr
    username: str


class UserCreate(UserBase):
    password: str


class UserResponse(UserBase):
    id: int
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[int] = None


class EggConfidence(BaseModel):
    detection: float
    damage: Optional[float] = None
    cleanliness: Optional[float] = None


class EggGeometry(BaseModel):
    length_mm: Optional[float] = None
    breadth_mm: Optional[float] = None
    quality: Optional[str] = None


class ModelVersions(BaseModel):
    detector: Optional[str] = None
    damage: Optional[str] = None
    cleanliness: Optional[str] = None
    calibration: Optional[str] = None


class DetectionBoxResponse(BaseModel):
    id: int
    prediction_id: int
    egg_id: Optional[int] = None
    class_name: str
    detection_confidence: float = Field(
        validation_alias="confidence", exclude=True
    )
    x1: int
    y1: int
    x2: int
    y2: int
    annotated: bool
    grade: Optional[str] = None
    provisional_grade: Optional[str] = None
    size: Optional[str] = None
    weight_g: Optional[float] = None
    damage_status: Optional[str] = None
    damage_confidence: Optional[float] = Field(
        default=None, exclude=True
    )
    cleanliness_status: Optional[str] = None
    cleanliness_confidence: Optional[float] = Field(
        default=None, exclude=True
    )
    length_mm: Optional[float] = Field(default=None, exclude=True)
    breadth_mm: Optional[float] = Field(default=None, exclude=True)
    geometry_quality: Optional[str] = Field(default=None, exclude=True)
    detection_model_version: Optional[str] = Field(
        default=None, exclude=True
    )
    damage_model_version: Optional[str] = Field(
        default=None, exclude=True
    )
    cleanliness_model_version: Optional[str] = Field(
        default=None, exclude=True
    )
    calibration_version: Optional[str] = Field(
        default=None, exclude=True
    )
    requires_review: bool = False
    review_reason: Optional[str] = None

    @computed_field
    @property
    def confidence(self) -> EggConfidence:
        return EggConfidence(
            detection=self.detection_confidence,
            damage=self.damage_confidence,
            cleanliness=self.cleanliness_confidence,
        )

    @computed_field
    @property
    def geometry(self) -> EggGeometry:
        return EggGeometry(
            length_mm=self.length_mm,
            breadth_mm=self.breadth_mm,
            quality=self.geometry_quality,
        )

    @computed_field
    @property
    def model_versions(self) -> ModelVersions:
        return ModelVersions(
            detector=self.detection_model_version,
            damage=self.damage_model_version,
            cleanliness=self.cleanliness_model_version,
            calibration=self.calibration_version,
        )

    class Config:
        from_attributes = True


class PredictionResponse(BaseModel):
    id: int
    file_name: str
    file_type: str
    tray_id: Optional[str] = None
    captured_at: Optional[datetime] = None
    status: str
    total_detections: int
    damaged_count: int
    not_damaged_count: int
    needs_review_count: int
    processing_time_ms: Optional[int]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class PredictionDetail(PredictionResponse):
    result_json: Optional[dict]
    annotated_path: Optional[str]
    detection_boxes: List[DetectionBoxResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True


class PredictionCreate(BaseModel):
    confidence_threshold: float = 0.75
    save_annotated: bool = True


class DashboardStats(BaseModel):
    total_predictions: int
    total_images: int
    total_videos: int
    total_detections: int
    damaged_percentage: float
    avg_processing_time_ms: float
    recent_predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    damage_model_loaded: bool = False
    cleanliness_model_loaded: bool = False
    database_connected: bool


class UserSettings(BaseModel):
    mm_per_pixel: float = 0.5

    class Config:
        from_attributes = True


class UserSettingsUpdate(BaseModel):
    mm_per_pixel: float


class GradeDistribution(BaseModel):
    grade_a: int = 0
    grade_b: int = 0
    grade_c: int = 0
    grade_na: int = 0
    grade_reject: int = 0
    total: int = 0
