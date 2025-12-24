from __future__ import annotations

from typing import Any

from app.core.schemas import Action, Decision, InferenceResult, WeatherBundle
from app.core.policies import Policies


def _parse_temp_range(ideal: str) -> tuple[float, float] | None:
    try:
        cleaned = str(ideal).replace("°C", "").replace(" ", "")
        a, b = cleaned.split("-", 1)
        return float(a), float(b)
    except Exception:
        return None


class ReasoningEngine:
    def __init__(self, policies: Policies):
        self.policies = policies

    def decide(
        self,
        inference: InferenceResult,
        care: dict[str, Any],
        weather: WeatherBundle,
    ) -> Decision:
        explanations: list[str] = []
        actions: list[Action] = []

        # ---- Fallback detection ----
        fallback_mode = "none"

        # Inference quality flags
        top1 = inference.topk[0].p if inference.topk else (inference.confidence or 0.0)
        top2 = inference.topk[1].p if len(inference.topk) > 1 else 0.0
        gap = top1 - top2

        low_conf = top1 < self.policies.thresholds.low_confidence
        ambiguous = gap < self.policies.thresholds.ambiguity_gap if len(inference.topk) > 1 else False

        if low_conf or ambiguous:
            fallback_mode = "uncertain_image"
            explanations.append(
                f"Uncertain prediction (top1={top1:.2f}, top2={top2:.2f}, gap={gap:.2f})."
            )
            actions.append(
                Action(
                    type="disease",
                    decision="uncertain",
                    reason="low_confidence_or_ambiguous",
                    evidence={"top1": top1, "top2": top2, "gap": gap},
                )
            )
        else:
            actions.append(
                Action(
                    type="disease",
                    decision="treat" if (inference.disease_status and "healthy" not in inference.disease_status.lower()) else "ok",
                    reason="model_prediction",
                    evidence={"label": inference.full_label, "confidence": top1},
                )
            )

        # Weather availability
        if not weather.ok:
            if fallback_mode == "none":
                fallback_mode = "no_weather"
            explanations.append("Weather unavailable; recommendations will ignore weather-dependent rules.")
            actions.append(
                Action(
                    type="weather",
                    decision="unavailable",
                    reason="weather_api_failed",
                    evidence={"error": weather.error},
                )
            )

        # Care availability
        if not care:
            if fallback_mode == "none":
                fallback_mode = "no_care_data"
            explanations.append("Care data unavailable for this plant/status; using generic guidance.")
            actions.append(
                Action(
                    type="care",
                    decision="unavailable",
                    reason="no_matching_row",
                    evidence={},
                )
            )

        # ---- Rule evaluations (only if data exists) ----
        risk = 0.0

        # Temperature
        if weather.ok and care.get("ideal_temperature") and weather.yesterday and weather.yesterday.get("avg_temp") is not None:
            rng = _parse_temp_range(care.get("ideal_temperature"))
            if rng:
                min_t, max_t = rng
                avg_t = float(weather.yesterday["avg_temp"])
                tol = self.policies.temperature.tolerance_c

                if avg_t < (min_t - tol):
                    risk += 0.25
                    actions.append(Action(
                        type="temperature",
                        decision="warn",
                        reason="below_ideal",
                        evidence={"avg_temp": avg_t, "ideal_min": min_t, "ideal_max": max_t},
                    ))
                    explanations.append(f"Temperature below ideal: {avg_t:.1f}°C vs {min_t:.1f}-{max_t:.1f}°C.")
                elif avg_t > (max_t + tol):
                    risk += 0.25
                    actions.append(Action(
                        type="temperature",
                        decision="warn",
                        reason="above_ideal",
                        evidence={"avg_temp": avg_t, "ideal_min": min_t, "ideal_max": max_t},
                    ))
                    explanations.append(f"Temperature above ideal: {avg_t:.1f}°C vs {min_t:.1f}-{max_t:.1f}°C.")
                else:
                    actions.append(Action(
                        type="temperature",
                        decision="ok",
                        reason="within_range",
                        evidence={"avg_temp": avg_t, "ideal_min": min_t, "ideal_max": max_t},
                    ))

        # Humidity
        if weather.ok and care.get("humidity_preference") and weather.yesterday and weather.yesterday.get("avg_humidity") is not None:
            pref = str(care.get("humidity_preference")).lower()
            hum = float(weather.yesterday["avg_humidity"])
            hp = self.policies.humidity

            ok = False
            if "high" in pref and hum >= hp.high:
                ok = True
            elif "medium" in pref and hp.medium_min <= hum <= hp.medium_max:
                ok = True
            elif "low" in pref and hum < hp.low:
                ok = True

            if ok:
                actions.append(Action(
                    type="humidity",
                    decision="ok",
                    reason="matches_preference",
                    evidence={"avg_humidity": hum, "preference": pref},
                ))
            else:
                risk += 0.15
                actions.append(Action(
                    type="humidity",
                    decision="warn",
                    reason="outside_preference",
                    evidence={"avg_humidity": hum, "preference": pref},
                ))
                explanations.append(f"Humidity may be suboptimal: {hum:.0f}% for preference '{pref}'.")

        # Watering (rain)
        if weather.ok and weather.yesterday and weather.yesterday.get("total_precipitation") is not None:
            rain_y = float(weather.yesterday.get("total_precipitation") or 0.0)
            rain_coming = False
            if weather.forecast:
                try:
                    rain_coming = any(float(f.get("precip") or 0.0) > self.policies.watering.rain_forecast_mm for f in weather.forecast)
                except Exception:
                    rain_coming = False

            if rain_y > self.policies.watering.rain_skip_mm:
                actions.append(Action(
                    type="watering",
                    decision="skip",
                    reason="recent_rain",
                    evidence={"rain_yesterday_mm": rain_y},
                ))
                explanations.append("Recent rain detected; skipping watering is recommended.")
            elif rain_coming:
                actions.append(Action(
                    type="watering",
                    decision="delay",
                    reason="rain_forecast",
                    evidence={"forecast_window_h": len(weather.forecast)},
                ))
                explanations.append("Rain forecast soon; consider delaying watering.")
            else:
                actions.append(Action(
                    type="watering",
                    decision="ok",
                    reason="no_rain",
                    evidence={"rain_yesterday_mm": rain_y},
                ))

        # ---- Severity mapping ----
        # Increase risk if disease predicted with high confidence
        if fallback_mode == "none" and inference.disease_status and "healthy" not in inference.disease_status.lower():
            risk += 0.25

        # Clamp
        risk = max(0.0, min(1.0, risk))

        if risk >= 0.70:
            severity = "High"
        elif risk >= 0.35:
            severity = "Medium"
        else:
            severity = "Low"

        return Decision(
            severity=severity,
            actions=actions,
            explanations=explanations,
            fallback_mode=fallback_mode,
        )
