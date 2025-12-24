from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    device: str


@dataclass(frozen=True)
class TopKItem:
    label: str
    p: float


@dataclass(frozen=True)
class QualityFlags:
    low_confidence: bool
    ambiguous: bool
    weather_unavailable: bool
    care_unavailable: bool


@dataclass(frozen=True)
class InferenceResult:
    plant_name: str | None
    disease_status: str | None
    full_label: str | None
    confidence: float | None
    topk: list[TopKItem]
    raw: dict[str, Any]


@dataclass(frozen=True)
class WeatherBundle:
    yesterday: dict[str, Any] | None
    forecast: list[dict[str, Any]] | None
    ok: bool
    error: str | None


@dataclass(frozen=True)
class Action:
    type: str               # e.g. "watering", "temperature", "humidity", "disease"
    decision: str           # e.g. "skip", "delay", "ok", "warn", "treat", "uncertain"
    reason: str             # short code, e.g. "recent_rain", "below_ideal"
    evidence: dict[str, Any]  # numerical/context evidence


@dataclass(frozen=True)
class Decision:
    severity: str           # "Low" | "Medium" | "High"
    actions: list[Action]
    explanations: list[str]
    fallback_mode: str      # "none" | "uncertain_image" | "no_weather" | "no_care_data"


@dataclass(frozen=True)
class AnalysisResult:
    plant: str | None
    health_status: str | None
    care_info: dict[str, Any]
    weather: dict[str, Any]
    recommendation: str

    model: ModelInfo
    topk: list[TopKItem]
    quality_flags: QualityFlags
    decision: Decision
