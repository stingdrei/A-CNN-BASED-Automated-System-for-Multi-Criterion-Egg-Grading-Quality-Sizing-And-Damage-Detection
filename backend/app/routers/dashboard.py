from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, text

from app.database import get_db
from app.models import User, Prediction, DetectionBox
from app.schemas import DashboardStats, HealthResponse, GradeDistribution
from app.security import get_current_user
from app.ml.model_registry import model_registry

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats", response_model=DashboardStats)
def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    total = db.query(Prediction).filter(Prediction.user_id == current_user.id).count()
    total_images = db.query(Prediction).filter(
        Prediction.user_id == current_user.id,
        Prediction.file_type == "image"
    ).count()
    total_videos = db.query(Prediction).filter(
        Prediction.user_id == current_user.id,
        Prediction.file_type == "video"
    ).count()
    total_detections = db.query(func.sum(Prediction.total_detections)).filter(
        Prediction.user_id == current_user.id
    ).scalar() or 0
    
    damaged = db.query(func.sum(Prediction.damaged_count)).filter(
        Prediction.user_id == current_user.id
    ).scalar() or 0
    not_damaged = db.query(func.sum(Prediction.not_damaged_count)).filter(
        Prediction.user_id == current_user.id
    ).scalar() or 0
    
    damaged_percentage = (damaged / total_detections * 100) if total_detections > 0 else 0
    
    avg_time = db.query(func.avg(Prediction.processing_time_ms)).filter(
        Prediction.user_id == current_user.id,
        Prediction.status == "completed"
    ).scalar() or 0
    
    recent = db.query(Prediction).filter(
        Prediction.user_id == current_user.id
    ).order_by(
        Prediction.created_at.desc()
    ).limit(5).all()
    
    return DashboardStats(
        total_predictions=total,
        total_images=total_images,
        total_videos=total_videos,
        total_detections=total_detections,
        damaged_percentage=round(damaged_percentage, 1),
        avg_processing_time_ms=round(avg_time, 1),
        recent_predictions=recent
    )


@router.get("/health", response_model=HealthResponse)
def health_check():
    database_connected = False
    try:
        db = next(get_db())
        db.execute(text("SELECT 1"))
        database_connected = True
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy"
        if (
            model_registry.detector_loaded
            and model_registry.damage_loaded
            and database_connected
        )
        else "degraded",
        model_loaded=model_registry.detector_loaded,
        damage_model_loaded=model_registry.damage_loaded,
        database_connected=database_connected
    )


@router.get("/grades", response_model=GradeDistribution)
def get_grade_distribution(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    prediction_ids = db.query(Prediction.id).filter(
        Prediction.user_id == current_user.id
    ).subquery()
    
    grade_counts = db.query(
        DetectionBox.grade,
        func.count(DetectionBox.id)
    ).filter(
        DetectionBox.prediction_id.in_(prediction_ids)
    ).group_by(DetectionBox.grade).all()
    
    grades_dict = {"AA": 0, "A": 0, "B": 0, "N/A": 0, "Reject": 0}
    for grade, count in grade_counts:
        if grade in grades_dict:
            grades_dict[grade] = count
    
    total = sum(grades_dict.values())
    
    return GradeDistribution(
        grade_aa=grades_dict["AA"],
        grade_a=grades_dict["A"],
        grade_b=grades_dict["B"],
        grade_na=grades_dict["N/A"],
        grade_reject=grades_dict["Reject"],
        total=total
    )