from __future__ import annotations

from app.core.schemas import (
    AnalysisResult,
    ModelInfo,
    QualityFlags,
)
from app.core.response import ResponseBuilder
from app.core.reasoning import ReasoningEngine
from app.core.policies import Policies
from app.services.inference_service import InferenceService
from app.services.weather_service import WeatherService
from app.services.care_service import CareService


class AnalysisPipeline:
    def __init__(
        self,
        inference: InferenceService,
        weather: WeatherService,
        care: CareService,
        policies: Policies,
        reasoning: ReasoningEngine | None = None,
        response_builder: ResponseBuilder | None = None,
        model_info: ModelInfo | None = None,
    ):
        self.inference = inference
        self.weather = weather
        self.care = care
        self.policies = policies
        self.reasoning = reasoning or ReasoningEngine(policies=policies)
        self.response_builder = response_builder or ResponseBuilder()
        self.model_info = model_info or ModelInfo(name="mobilenetv2", version="1.0.0", device="cpu")

    def run(self, image_path: str, latitude: float, longitude: float) -> AnalysisResult:
        inf = self.inference.run(image_path=image_path, top_k=3)

        # quality flags derived from topk
        top1 = inf.topk[0].p if inf.topk else (inf.confidence or 0.0)
        top2 = inf.topk[1].p if len(inf.topk) > 1 else 0.0
        low_confidence = top1 < self.policies.thresholds.low_confidence
        ambiguous = (top1 - top2) < self.policies.thresholds.ambiguity_gap if len(inf.topk) > 1 else False

        weather_bundle = self.weather.get(latitude=latitude, longitude=longitude)

        care_info = self.care.get_care_info(
            plant_name=inf.plant_name,
            disease_status=inf.disease_status,
        )

        decision = self.reasoning.decide(inference=inf, care=care_info, weather=weather_bundle)

        recommendation = self.response_builder.build_recommendation(
            inference=inf,
            care=care_info,
            weather=weather_bundle,
            decision=decision,
        )

        quality_flags = QualityFlags(
            low_confidence=low_confidence,
            ambiguous=ambiguous,
            weather_unavailable=not weather_bundle.ok,
            care_unavailable=(care_info == {}),
        )

        # Keep old keys, but enrich with the new signals/decisions
        return AnalysisResult(
            plant=inf.plant_name,
            health_status=inf.disease_status,
            care_info=care_info,
            weather={
                "yesterday": weather_bundle.yesterday,
                "forecast": weather_bundle.forecast,
                "ok": weather_bundle.ok,
                "error": weather_bundle.error,
            },
            recommendation=recommendation,
            model=self.model_info,
            topk=inf.topk,
            quality_flags=quality_flags,
            decision=decision,
        )
