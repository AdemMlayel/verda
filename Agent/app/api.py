from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.verda_agent import VerdaAgent

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Verda Agent API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent: VerdaAgent | None = None


@app.on_event("startup")
def startup_event():
    global agent
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    logger.info("🌱 Initializing Verda Agent...")

    class_names = [
        "Apple___apple_scab", "Apple___black_rot", "Apple___cedar_apple_rust", "Apple___healthy",
        "Bell_pepper___bacterial_spot", "Bell_pepper___healthy",
        "Cherry___healthy", "Cherry___powdery_mildew",
        "Corn_maize___cercospora_leaf_spot", "Corn_maize___common_rust", "Corn_maize___healthy",
        "Corn_maize___northern_leaf_blight",
        "Grape___black_rot", "Grape___esca_(black_measles)", "Grape___healthy", "Grape___leaf_blight",
        "Peach___bacterial_spot", "Peach___healthy",
        "Potato___early_blight", "Potato___healthy", "Potato___late_blight",
        "Strawberry___healthy", "Strawberry___leaf_scorch",
        "Tomato___bacterial_spot", "Tomato___early_blight", "Tomato___healthy", "Tomato___late_blight",
        "Tomato___leaf_mold", "Tomato___septoria_leaf_spot", "Tomato___yellow_leaf_curl_virus",
    ]

    agent = VerdaAgent(
        model_path=settings.model_path,
        class_names=class_names,
        care_data_path=settings.care_data_path,
    )

    logger.info("✅ Verda Agent ready")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_plant(
    image: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
):
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    suffix = Path(image.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        return agent.analyze(
            image_path=tmp_path,
            latitude=latitude,
            longitude=longitude,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
