from __future__ import annotations

from typing import Any

import torch

from app.plant import PlantIdentifierAgent
from app.core.schemas import InferenceResult, TopKItem
from app.core.policies import Policies


class InferenceService:
    def __init__(self, classifier: PlantIdentifierAgent, policies: Policies):
        self.classifier = classifier
        self.policies = policies

    def run(self, image_path: str, top_k: int = 3) -> InferenceResult:
        """
        Uses classifier.model + class_names to compute topK in a robust way.
        Falls back to classifier.predict() if needed.
        """
        # Try to compute topK using the underlying model for better signal quality
        try:
            # Use the classifier's internal preprocessing by calling predict then recompute topK not feasible.
            # Instead we rely on classifier.predict() for plant/status + confidence, and compute topK from model if available.
            pred = self.classifier.predict(image_path=image_path)

            # If classifier exposes model & transform & class_names, we can compute topK:
            model = getattr(self.classifier, "model", None)
            transform = getattr(self.classifier, "transform", None)
            class_names = getattr(self.classifier, "class_names", None)

            topk_items: list[TopKItem] = []
            if model is not None and transform is not None and class_names is not None:
                from PIL import Image
                image = Image.open(image_path).convert("RGB")
                tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.softmax(outputs[0], dim=0)
                    k = min(int(top_k), probs.shape[0])
                    vals, idxs = torch.topk(probs, k=k)
                for p, i in zip(vals.tolist(), idxs.tolist()):
                    topk_items.append(TopKItem(label=str(class_names[i]), p=float(p)))
            else:
                # Minimal topK from existing pred
                full_label = pred.get("full_label")
                conf = float(pred.get("confidence") or 0.0)
                topk_items = [TopKItem(label=str(full_label), p=conf)] if full_label else []

            return InferenceResult(
                plant_name=pred.get("plant_name"),
                disease_status=pred.get("disease_status"),
                full_label=pred.get("full_label"),
                confidence=float(pred.get("confidence")) if pred.get("confidence") is not None else None,
                topk=topk_items,
                raw=pred,
            )

        except Exception as e:
            # Hard failure should still surface; pipeline will catch and report
            raise RuntimeError(f"Inference failed: {e}") from e
