from __future__ import annotations

from typing import Any

import pandas as pd


class CareService:
    """
    Excel-backed care lookup.
    Keeps existing column logic and output keys to avoid behavior change.
    """

    def __init__(self, excel_path: str):
        self.care_df = pd.read_excel(excel_path)

    def get_care_info(self, plant_name: str | None, disease_status: str | None) -> dict[str, Any]:
        if not plant_name or not disease_status:
            return {}

        if "Plant" not in self.care_df.columns or "Disease/Healthy" not in self.care_df.columns:
            # Preserve “fail-soft” behavior
            return {}

        mask = (
            (self.care_df["Plant"] == plant_name)
            & self.care_df["Disease/Healthy"]
            .astype(str)
            .str.lower()
            .str.contains(str(disease_status).lower(), na=False)
        )

        rows = self.care_df[mask]
        if rows.empty:
            return {}

        row = rows.iloc[0]
        return {
            "water_needs": row.get("Water Requirements"),
            "sunlight": row.get("Sunlight"),
            "soil_type": row.get("Soil Type"),
            "ideal_temperature": row.get("Temperature"),
            "humidity_preference": row.get("Humidity"),
            "preventive_tips": row.get("Prevention Tips"),
        }
