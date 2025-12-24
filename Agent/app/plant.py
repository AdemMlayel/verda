from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from app.mobilenet import MobileNetV2Classifier


logger = logging.getLogger(__name__)


class PlantIdentifierAgent:
    def __init__(self, model_path: str, class_names: list[str], device: str | None = None):
        self.class_names = class_names

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading plant classifier on device: %s", self.device)

        self.model = MobileNetV2Classifier(num_classes=len(class_names))
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        logger.info("PlantIdentifierAgent initialized successfully")

    def predict(self, image_path: str) -> dict[str, Any]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs[0], dim=0)
            top_prob, top_class = torch.max(probs, dim=0)

        full_label = self.class_names[top_class.item()].strip()

        if "___" in full_label:
            plant_name, disease_status = full_label.split("___", 1)
        else:
            plant_name = full_label
            disease_status = "Unknown"

        return {
            "plant_name": plant_name,
            "disease_status": disease_status,
            "full_label": full_label,
            "confidence": float(top_prob.item()),
        }
