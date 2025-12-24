from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    low_confidence: float = 0.55
    ambiguity_gap: float = 0.10


@dataclass(frozen=True)
class WateringPolicy:
    rain_skip_mm: float = 1.0
    rain_forecast_mm: float = 0.5


@dataclass(frozen=True)
class TemperaturePolicy:
    tolerance_c: float = 2.0


@dataclass(frozen=True)
class HumidityPolicy:
    high: float = 70.0
    medium_min: float = 40.0
    medium_max: float = 70.0
    low: float = 40.0


@dataclass(frozen=True)
class Policies:
    thresholds: Thresholds = Thresholds()
    watering: WateringPolicy = WateringPolicy()
    temperature: TemperaturePolicy = TemperaturePolicy()
    humidity: HumidityPolicy = HumidityPolicy()
