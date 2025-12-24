from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Paths (relative to /app inside container)
    model_path: str = os.getenv("MODEL_PATH", "models/fine_tuned_mobilenet.pth")
    care_data_path: str = os.getenv("CARE_DATA_PATH", "data/care_details.xlsx")

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Weather
    weather_timeout_s: float = float(os.getenv("WEATHER_TIMEOUT_S", "10"))
    weather_retries: int = int(os.getenv("WEATHER_RETRIES", "2"))


settings = Settings()
