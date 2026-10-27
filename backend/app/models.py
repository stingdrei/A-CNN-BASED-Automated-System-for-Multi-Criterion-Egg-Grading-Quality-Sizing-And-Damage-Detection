from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    mm_per_pixel = Column(Float, nullable=False, default=0.09)
    created_at = Column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    predictions = relationship(
        "Prediction",
        back_populates="user", cascade="all, delete-orphan"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    user_id = Column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    file_name = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_type = Column(String(20), nullable=False)
    tray_id = Column(String(64), nullable=True, index=True)
    captured_at = Column(DateTime, nullable=True)
    status = Column(String(20), nullable=False, default="processing")
    total_detections = Column(Integer, nullable=False, default=0)
    damaged_count = Column(Integer, nullable=False, default=0)
    not_damaged_count = Column(Integer, nullable=False, default=0)
    needs_review_count = Column(Integer, nullable=False, default=0)
    processing_time_ms = Column(Integer, nullable=True)
    result_json = Column(JSON, nullable=True)
    annotated_path = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="predictions")
    detection_boxes = relationship(
        "DetectionBox",
        back_populates="prediction", cascade="all, delete-orphan"
    )


class DetectionBox(Base):
    __tablename__ = "detection_boxes"

    id = Column(Integer, primary_key=True)
    prediction_id = Column(
        ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    egg_id = Column(Integer, nullable=True)
    class_name = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    x1 = Column(Integer, nullable=False)
    y1 = Column(Integer, nullable=False)
    x2 = Column(Integer, nullable=False)
    y2 = Column(Integer, nullable=False)
    annotated = Column(Boolean, nullable=False, default=False)
    size = Column(String(20), nullable=True)
    length_mm = Column(Float, nullable=True)
    breadth_mm = Column(Float, nullable=True)
    geometry_quality = Column(String(30), nullable=True)
    weight_g = Column(Float, nullable=True)
    grade = Column(String(10), nullable=True)
    provisional_grade = Column(String(10), nullable=True)
    requires_review = Column(Boolean, nullable=False, default=False)
    review_reason = Column(String(255), nullable=True)
    damage_status = Column(String(20), nullable=True)
    damage_confidence = Column(Float, nullable=True)
    damage_model_version = Column(String(255), nullable=True)
    cleanliness_status = Column(String(20), nullable=True)
    cleanliness_confidence = Column(Float, nullable=True)
    cleanliness_model_version = Column(String(255), nullable=True)
    detection_model_version = Column(String(255), nullable=True)
    calibration_version = Column(String(50), nullable=True)

    prediction = relationship("Prediction", back_populates="detection_boxes")
