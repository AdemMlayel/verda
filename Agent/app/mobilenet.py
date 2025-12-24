from __future__ import annotations

import torch.nn as nn
from torchvision.models import mobilenet_v2


class ImageClassificationBase(nn.Module):
    """
    Kept minimal for inference compatibility.
    (Training-time methods are not required at runtime.)
    """
    pass


class MobileNetV2Classifier(ImageClassificationBase):
    """
    IMPORTANT:
    This class must match the training-time architecture and attribute names.
    Your training script used:
        self.network = mobilenet_v2(...)
        self.network.classifier[1] = Linear(...)
    and saved:
        torch.save(model.state_dict(), ...)
    so the state_dict keys are prefixed with 'network.'.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.network = mobilenet_v2(weights=None)
        # In training you used: nn.Linear(self.network.last_channel, num_classes)
        self.network.classifier[1] = nn.Linear(self.network.last_channel, num_classes)

    def forward(self, xb):
        return self.network(xb)
