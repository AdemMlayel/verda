from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any

import requests

from app.config import settings


logger = logging.getLogger(__name__)


class UnifiedWeatherAgent:
    """
    Unified weather provider.

    - Yesterday weather: Open-Meteo archive API
    - Short-term forecast: MET Norway API

    Guarantees a STABLE output schema for downstream agents.
    """

    def __init__(self, app_name: str = "Verda", email: str = "contact@verda.ai"):
        self.metno_url = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
        self.metno_headers = {
            "User-Agent": f"{app_name} ({email})"
        }

        self.timeout = settings.weather_timeout_s
        self.retries = settings.weather_retries

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------
    def _get(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
        last_exc: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Weather request failed (%s/%s): %s",
                    attempt + 1,
                    self.retries + 1,
                    exc,
                )
                time.sleep(0.5)

        raise RuntimeError("Weather API request failed after retries") from last_exc

    # ------------------------------------------------------------------
    # Yesterday (historical)
    # ------------------------------------------------------------------
    def get_yesterday_weather(self, latitude: float, longitude: float) -> dict[str, float | None]:
        """
        Returns:
        {
            "avg_temp": float | None,
            "avg_humidity": float | None,
            "total_precipitation": float | None
        }
        """
        yesterday = datetime.utcnow() - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation",
            "timezone": "auto",
        }

        data = self._get(url, params=params)

        hourly = data.get("hourly", {})
        temps = [t for t in hourly.get("temperature_2m", []) if t is not None]
        humidity = [h for h in hourly.get("relative_humidity_2m", []) if h is not None]
        rain = [r for r in hourly.get("precipitation", []) if r is not None]

        avg_temp = sum(temps) / len(temps) if temps else None
        avg_humidity = sum(humidity) / len(humidity) if humidity else None
        total_precipitation = sum(rain) if rain else None

        logger.debug(
            "Yesterday weather lat=%s lon=%s → temp=%s humidity=%s rain=%s",
            latitude,
            longitude,
            avg_temp,
            avg_humidity,
            total_precipitation,
        )

        return {
            "avg_temp": avg_temp,
            "avg_humidity": avg_humidity,
            "total_precipitation": total_precipitation,
        }

    # ------------------------------------------------------------------
    # Short-term forecast
    # ------------------------------------------------------------------
    def get_today_forecast(self, latitude: float, longitude: float, hours: int = 6) -> list[dict[str, Any]]:
        """
        Returns list[dict]:
        [
            {
                "time": ISO8601 str,
                "temp": float | None,
                "humidity": float | None,
                "precip": float,
                "symbol": str
            },
            ...
        ]
        """
        params = {
            "lat": latitude,
            "lon": longitude,
        }

        data = self._get(self.metno_url, params=params, headers=self.metno_headers)

        timeseries = data.get("properties", {}).get("timeseries", [])
        forecast: list[dict[str, Any]] = []

        for entry in timeseries[:hours]:
            instant = entry.get("data", {}).get("instant", {}).get("details", {})
            next_hour = entry.get("data", {}).get("next_1_hours", {})

            forecast.append(
                {
                    "time": entry.get("time"),
                    "temp": instant.get("air_temperature"),
                    "humidity": instant.get("relative_humidity"),
                    "precip": next_hour.get("details", {}).get("precipitation_amount", 0.0),
                    "symbol": next_hour.get("summary", {}).get("symbol_code", "n/a"),
                }
            )

        logger.debug(
            "Forecast lat=%s lon=%s → %s entries",
            latitude,
            longitude,
            len(forecast),
        )

        return forecast


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    agent = UnifiedWeatherAgent()
    y = agent.get_yesterday_weather(34.43, 8.77)
    f = agent.get_today_forecast(34.43, 8.77)

    print("Yesterday:", y)
    print("Forecast:", f)
