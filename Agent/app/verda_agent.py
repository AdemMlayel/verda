from __future__ import annotations

import logging
from typing import Any

from app.plant import PlantIdentifierAgent
from app.weather import UnifiedWeatherAgent

from app.core.pipeline import AnalysisPipeline
from app.core.policies import Policies
from app.core.schemas import ModelInfo
from app.services.inference_service import InferenceService
from app.services.weather_service import WeatherService
from app.services.care_service import CareService

logger = logging.getLogger(__name__)


class VerdaAgent:
    def __init__(self, model_path: str, class_names: list[str], care_data_path: str):
        classifier = PlantIdentifierAgent(model_path=model_path, class_names=class_names)
        weather_agent = UnifiedWeatherAgent()

        policies = Policies()

        # device info (best-effort)
        device = getattr(classifier, "device", None)
        device_name = str(device) if device is not None else "cpu"

        self.pipeline = AnalysisPipeline(
            inference=InferenceService(classifier=classifier, policies=policies),
            weather=WeatherService(weather_agent=weather_agent),
            care=CareService(excel_path=care_data_path),
            policies=policies,
            model_info=ModelInfo(name="mobilenetv2", version="1.0.0", device=device_name),
        )

        logger.info("✅ VerdaAgent initialized (Iteration 2: topK + decisions)")

    def analyze(self, image_path: str, latitude: float, longitude: float) -> dict[str, Any]:
        result = self.pipeline.run(image_path=image_path, latitude=latitude, longitude=longitude)

        # Preserve old keys, add new keys (non-breaking)
        return {
            "plant": result.plant,
            "health_status": result.health_status,
            "care_info": result.care_info,
            "weather": result.weather,
            "recommendation": result.recommendation,

            # New fields
            "model": {
                "name": result.model.name,
                "version": result.model.version,
                "device": result.model.device,
            },
            "topk": [{"label": x.label, "p": x.p} for x in result.topk],
            "quality_flags": {
                "low_confidence": result.quality_flags.low_confidence,
                "ambiguous": result.quality_flags.ambiguous,
                "weather_unavailable": result.quality_flags.weather_unavailable,
                "care_unavailable": result.quality_flags.care_unavailable,
            },
            "severity": result.decision.severity,
            "actions": [
                {
                    "type": a.type,
                    "decision": a.decision,
                    "reason": a.reason,
                    "evidence": a.evidence,
                }
                for a in result.decision.actions
            ],
            "explanations": result.decision.explanations,
            "fallback_mode": result.decision.fallback_mode,
        }
