from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator
from typing import Optional, List
from datetime import datetime


ALLOWED_GRADES = {"A", "B", "C", "N/A", "Reject"}
ALLOWED_DAMAGE = {"Damaged", "Not Damaged", "Needs Review"}
ALLOWED_CLEANLINESS = {
    "Clean",
    "Stained",
    "Needs Review",
    "Unavailable",
    "Unknown",
}
ALLOWED_SIZES = {"Small", "Medium", "Large", "Unknown"}
ALLOWED_STATUSES = {"processing", "completed", "failed"}


class EggConfidence(BaseModel):
    detection: float = Field(ge=0, le=1)
    damage: Optional[float] = Field(default=None, ge=0, le=1)
    cleanliness: Optional[float] = Field(default=None, ge=0, le=1)


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
        validation_alias="confidence", exclude=True, ge=0, le=1
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
        default=None, exclude=True, ge=0, le=1
    )
    cleanliness_status: Optional[str] = None
    cleanliness_confidence: Optional[float] = Field(
        default=None, exclude=True, ge=0, le=1
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

    @field_validator("grade", "provisional_grade")
    @classmethod
    def validate_grade(cls, value):
        if value == "AA":
            return "A"
        if value is not None and value not in ALLOWED_GRADES:
            raise ValueError(f"Unsupported grade: {value!r}")
        return value

    @field_validator("damage_status")
    @classmethod
    def validate_damage_status(cls, value):
        if value is not None and value not in ALLOWED_DAMAGE:
            raise ValueError(f"Unsupported damage status: {value!r}")
        return value

    @field_validator("cleanliness_status")
    @classmethod
    def validate_cleanliness_status(cls, value):
        if value is not None and value not in ALLOWED_CLEANLINESS:
            raise ValueError(f"Unsupported cleanliness status: {value!r}")
        return value

    @field_validator("size")
    @classmethod
    def validate_size(cls, value):
        if value is not None and value not in ALLOWED_SIZES:
            raise ValueError(f"Unsupported size: {value!r}")
        return value

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
    tray_id: str
    captured_at: Optional[datetime] = None
    operator: Optional[str] = None
    inference_metadata: Optional[dict] = None
    status: str
    total_detections: int
    damaged_count: int
    not_damaged_count: int
    needs_review_count: int
    processing_time_ms: Optional[int]
    created_at: datetime
    completed_at: Optional[datetime]

    @field_validator("status")
    @classmethod
    def validate_status(cls, value):
        if value not in ALLOWED_STATUSES:
            raise ValueError(f"Unsupported prediction status: {value!r}")
        return value

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
    mm_per_pixel: float = Field(gt=0, le=10)

    class Config:
        from_attributes = True


class UserSettingsUpdate(BaseModel):
    mm_per_pixel: float = Field(gt=0, le=10)


class GradeDistribution(BaseModel):
    grade_a: int = 0
    grade_b: int = 0
    grade_c: int = 0
    grade_na: int = 0
    grade_reject: int = 0
    total: int = 0
