from __future__ import annotations

from app.weather import UnifiedWeatherAgent
from app.core.schemas import WeatherBundle


class WeatherService:
    def __init__(self, weather_agent: UnifiedWeatherAgent):
        self.weather_agent = weather_agent

    def get(self, latitude: float, longitude: float) -> WeatherBundle:
        try:
            yesterday = self.weather_agent.get_yesterday_weather(latitude, longitude)
            forecast = self.weather_agent.get_today_forecast(latitude, longitude)
            return WeatherBundle(yesterday=yesterday, forecast=forecast, ok=True, error=None)
        except Exception as e:
            return WeatherBundle(yesterday=None, forecast=None, ok=False, error=str(e))
