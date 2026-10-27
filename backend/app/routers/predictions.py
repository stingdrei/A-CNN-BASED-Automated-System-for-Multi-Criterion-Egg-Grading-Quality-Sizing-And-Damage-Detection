from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models import User, Prediction
from app.schemas import PredictionResponse, PredictionDetail
from app.security import get_current_user
from app.services.file_service import file_service
from app.services.prediction_service import prediction_service
from app.ml.model_registry import model_registry
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/upload", response_model=PredictionDetail)
async def upload_file(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.75, ge=0.1, le=0.99),
    save_annotated: bool = Query(True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not file_service.is_allowed_file(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported image type. Allowed: jpg, jpeg, png, webp, bmp"
        )

    try:
        file_path, file_type = file_service.save_file_sync(file)
        prediction = prediction_service.create_prediction(
            db,
            current_user.id,
            file.filename,
            file_path,
            file_type,
            operator=current_user.username,
        )
        if (
            not model_registry.detector_loaded
            or not model_registry.damage_loaded
            or model_registry.detector is None
            or model_registry.damage is None
        ):
            raise RuntimeError("Required prediction models are not available")
        model_registry.detector.confidence = confidence_threshold
        model_registry.detector.mm_per_pixel = current_user.mm_per_pixel
        cleanliness_classifier = (
            model_registry.cleanliness
            if model_registry.cleanliness_loaded
            else None
        )
        pred_service = PredictionService(
            model_registry.detector,
            model_registry.damage,
            cleanliness_classifier,
        )
        prediction = pred_service.run_prediction(
            db, prediction.id, confidence_threshold, save_annotated
        )
    except ValueError as e:
        if "file_path" in locals():
            file_service.delete_file(file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        if "file_path" in locals() and "prediction" not in locals():
            file_service.delete_file(file_path)
        if "prediction" in locals():
            prediction.status = "failed"
            prediction.error_message = "Prediction processing failed"
            db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction processing failed",
        ) from e
    
    return prediction


@router.get("", response_model=List[PredictionResponse])
def get_predictions(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return prediction_service.get_predictions(db, current_user.id, skip, limit)


@router.get("/{prediction_id}", response_model=PredictionDetail)
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    prediction = prediction_service.get_prediction_detail(
        db, prediction_id, current_user.id
    )
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )
    return prediction


@router.delete("/{prediction_id}")
def delete_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    success = prediction_service.delete_prediction(
        db, prediction_id, current_user.id
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )
    return {"message": "Prediction deleted"}


@router.get("/{prediction_id}/download")
def download_annotated(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    prediction = prediction_service.get_prediction_detail(
        db, prediction_id, current_user.id
    )
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )
    if not prediction.annotated_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotated file not found"
        )
    return FileResponse(
        prediction.annotated_path,
        media_type="image/jpeg",
        filename=f"annotated_{prediction.file_name}"
    )