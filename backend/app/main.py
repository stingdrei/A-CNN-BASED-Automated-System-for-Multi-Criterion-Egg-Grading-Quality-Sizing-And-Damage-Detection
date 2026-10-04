from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.config import settings
from app.database import engine, Base
from app.routers import auth, predictions, dashboard, user_settings
from app.ml.model_registry import model_registry

model_loaded = False
damage_model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loaded, damage_model_loaded
    try:
        model_loaded, damage_model_loaded = model_registry.load()
        print(
            "Models loaded: "
            f"detector={model_loaded}, damage_classifier={damage_model_loaded}"
        )
    except Exception as e:
        print(f"Warning: Could not load YOLO model: {e}")
    
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded files statically
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.include_router(auth.router, prefix="/api/v1/auth")
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(dashboard.router, prefix="/api/v1")
app.include_router(user_settings.router, prefix="/api/v1")


@app.get("/")
def read_root():
    return {"message": "Egg-CV API", "version": settings.VERSION}