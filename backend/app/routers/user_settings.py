from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.operator import get_operator
from app.schemas import UserSettings, UserSettingsUpdate

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=UserSettings)
def get_settings(
    db: Session = Depends(get_db),
    operator: User = Depends(get_operator)
):
    return UserSettings(mm_per_pixel=operator.mm_per_pixel)


@router.put("")
def update_settings(
    settings: UserSettingsUpdate,
    db: Session = Depends(get_db),
    operator: User = Depends(get_operator)
):
    operator.mm_per_pixel = settings.mm_per_pixel
    db.commit()
    return {"message": "Settings updated successfully", "mm_per_pixel": settings.mm_per_pixel}