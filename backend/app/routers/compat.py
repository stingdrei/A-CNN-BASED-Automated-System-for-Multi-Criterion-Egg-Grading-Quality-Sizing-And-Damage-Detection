"""Compatibility API aliases for the documented grading operations.

The primary API lives under ``/api/v1/predictions`` and
``/api/v1/dashboard``. These aliases preserve the short contract used by the
project documentation without duplicating inference or query logic.
"""

from typing import List

from fastapi import APIRouter

from app.routers.dashboard import get_dashboard_stats
from app.routers.predictions import get_predictions, upload_file
from app.schemas import DashboardStats, PredictionDetail, PredictionResponse

router = APIRouter(tags=["compat"])

router.add_api_route(
    "/grade",
    upload_file,
    methods=["POST"],
    response_model=PredictionDetail,
    name="grade_alias",
    summary="Alias for POST /predictions/upload",
)
router.add_api_route(
    "/records",
    get_predictions,
    methods=["GET"],
    response_model=List[PredictionResponse],
    name="records_alias",
    summary="Alias for GET /predictions",
)
router.add_api_route(
    "/stats",
    get_dashboard_stats,
    methods=["GET"],
    response_model=DashboardStats,
    name="stats_alias",
    summary="Alias for GET /dashboard/stats",
)
