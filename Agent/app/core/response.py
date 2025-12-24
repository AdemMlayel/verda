from __future__ import annotations

from typing import Any

from app.core.schemas import Decision, InferenceResult, WeatherBundle


class ResponseBuilder:
    def build_recommendation(
        self,
        inference: InferenceResult,
        care: dict[str, Any],
        weather: WeatherBundle,
        decision: Decision,
    ) -> str:
        tips: list[str] = []

        plant_name = inference.plant_name or "Unknown"
        tips.append(f"🔎 Plant Name: {plant_name}")

        # Fallback mode messaging
        if decision.fallback_mode == "uncertain_image":
            tips.append("⚠️ The image diagnosis is uncertain. Try a clearer photo (good lighting, close-up leaf, plain background).")

        # Temperature summary from actions
        for a in decision.actions:
            if a.type == "temperature":
                if a.decision == "warn" and a.reason == "below_ideal":
                    tips.append("🌡️ Temperature is below the ideal range.")
                elif a.decision == "warn" and a.reason == "above_ideal":
                    tips.append("🌡️ Temperature is above the ideal range.")
                elif a.decision == "ok":
                    tips.append("🌡️ Temperature looks within range.")

        # Humidity summary
        for a in decision.actions:
            if a.type == "humidity":
                if a.decision == "warn":
                    tips.append("💧 Humidity may not be optimal.")
                elif a.decision == "ok":
                    tips.append("💧 Humidity looks suitable.")

        # Watering summary
        for a in decision.actions:
            if a.type == "watering":
                if a.decision == "skip":
                    tips.append("🌧️ It rained recently — skip watering today.")
                elif a.decision == "delay":
                    tips.append("☁️ Rain is forecasted — consider delaying watering.")
                elif a.decision == "ok":
                    tips.append("💦 No rain detected — watering may be needed depending on soil moisture.")

        # Care fields (still included)
        tips.append(f"☀️ Sunlight Needs: {care.get('sunlight', 'N/A')}")
        tips.append(f"🌱 Soil Type: {care.get('soil_type', 'N/A')}")

        # Severity
        tips.append(f"🧭 Severity: {decision.severity}")

        return "\n".join(tips)
