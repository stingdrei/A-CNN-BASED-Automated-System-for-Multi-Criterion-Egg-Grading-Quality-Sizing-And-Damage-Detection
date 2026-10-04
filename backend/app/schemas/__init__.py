from pydantic import BaseModel, EmailStr, Field
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


class DetectionBoxResponse(BaseModel):
    id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    annotated: bool
    size_category: Optional[str] = None
    weight_g: Optional[float] = None
    grade: Optional[str] = None
    damage_status: Optional[str] = None
    damage_confidence: Optional[float] = None
    damage_model_version: Optional[str] = None
    review_reason: Optional[str] = None

    class Config:
        from_attributes = True


class PredictionResponse(BaseModel):
    id: int
    file_name: str
    file_type: str
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
    status: str
    model_loaded: bool
    damage_model_loaded: bool = False
    database_connected: bool


class UserSettings(BaseModel):
    mm_per_pixel: float = 0.5

    class Config:
        from_attributes = True


class UserSettingsUpdate(BaseModel):
    mm_per_pixel: float


class GradeDistribution(BaseModel):
    grade_aa: int = 0
    grade_a: int = 0
    grade_b: int = 0
    grade_na: int = 0
    grade_reject: int = 0
    total: int = 0