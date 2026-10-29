from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, select, text

from app.database import get_db
from app.models import User, Prediction, DetectionBox
from app.operator import get_operator
from app.schemas import DashboardStats, HealthResponse, GradeDistribution
from app.ml.model_registry import model_registry

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats", response_model=DashboardStats)
def get_dashboard_stats(
    db: Session = Depends(get_db),
    operator: User = Depends(get_operator)
):
    total = db.query(Prediction).filter(Prediction.user_id == operator.id).count()
    total_images = db.query(Prediction).filter(
        Prediction.user_id == operator.id,
        Prediction.file_type == "image"
    ).count()
    total_detections = db.query(func.sum(Prediction.total_detections)).filter(
        Prediction.user_id == operator.id
    ).scalar() or 0
    
    damaged = db.query(func.sum(Prediction.damaged_count)).filter(
        Prediction.user_id == operator.id
    ).scalar() or 0
    not_damaged = db.query(func.sum(Prediction.not_damaged_count)).filter(
        Prediction.user_id == operator.id
    ).scalar() or 0
    
    damaged_percentage = (damaged / total_detections * 100) if total_detections > 0 else 0
    
    avg_time = db.query(func.avg(Prediction.processing_time_ms)).filter(
        Prediction.user_id == operator.id,
        Prediction.status == "completed"
    ).scalar() or 0
    
    recent = db.query(Prediction).filter(
        Prediction.user_id == operator.id
    ).order_by(
        Prediction.created_at.desc()
    ).limit(5).all()
    
    return DashboardStats(
        total_predictions=total,
        total_images=total_images,
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
    
    cleanliness_loaded = model_registry.cleanliness_loaded
    ready = (
        model_registry.detector_loaded
        and model_registry.damage_loaded
        and cleanliness_loaded
        and database_connected
    )
    return HealthResponse(
        status="healthy" if ready else "degraded",
        model_loaded=model_registry.detector_loaded,
        damage_model_loaded=model_registry.damage_loaded,
        cleanliness_model_loaded=cleanliness_loaded,
        database_connected=database_connected
    )


@router.get("/grades", response_model=GradeDistribution)
def get_grade_distribution(
    db: Session = Depends(get_db),
    operator: User = Depends(get_operator)
):
    prediction_ids = select(Prediction.id).where(
        Prediction.user_id == operator.id
    )
    
    grade_counts = db.query(
        DetectionBox.grade,
        func.count(DetectionBox.id)
    ).filter(
        DetectionBox.prediction_id.in_(prediction_ids)
    ).group_by(DetectionBox.grade).all()
    
    grades_dict = {"A": 0, "B": 0, "C": 0, "N/A": 0, "Reject": 0}
    for grade, count in grade_counts:
        if grade == "AA":
            grade = "A"
        if grade in grades_dict:
            grades_dict[grade] += count

    total = sum(grades_dict.values())

    return GradeDistribution(
        grade_a=grades_dict["A"],
        grade_b=grades_dict["B"],
        grade_c=grades_dict["C"],
        grade_na=grades_dict["N/A"],
        grade_reject=grades_dict["Reject"],
        total=total
    )