import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class FootballActionRecognizer(nn.Module):
    """
    Architecture:
        Input → EfficientNetB0 (frozen) → GlobalAvgPool → Dense(256) → Dropout(0.5) → Dense(num_classes)

    Notes:
        - ImageNet mean/std normalization is applied externally in the DataLoader pipeline.
        - forward() returns raw logits (apply softmax externally for probabilities).
        - freeze_backbone() / unfreeze_from_block5() control fine-tuning phases.

    Args:
        num_classes: number of output classes (default 6)
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()

        efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # ── Backbone: EfficientNetB0 feature extractor ─────────────────────
        # features[0]  = stem Conv2dNormActivation
        # features[1]  = MBConv stage 1  (block1)
        # features[2]  = MBConv stage 2  (block2)
        # features[3]  = MBConv stage 3  (block3)
        # features[4]  = MBConv stage 4  (block4)
        # features[5]  = MBConv stage 5  (block5)  ← fine-tune from here
        # features[6]  = MBConv stage 6  (block6)
        # features[7]  = MBConv stage 7  (block7)
        # features[8]  = top Conv2dNormActivation
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool  # AdaptiveAvgPool2d(1)

        # ── Custom classification head ──────────────────────────────────────
        in_features = 1280  # EfficientNet-B0 output channels
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self.freeze_backbone()

    # ── Backbone freeze / unfreeze helpers ─────────────────────────────────

    def freeze_backbone(self):
        """Freeze all backbone parameters (Phase 1: train head only)."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_from_block5(self):
        """Freeze backbone up to block4, unfreeze block5+ (Phase 2 fine-tuning)."""
        for param in self.features.parameters():
            param.requires_grad = False
        for stage_idx in [5, 6, 7, 8]:
            for param in self.features[stage_idx].parameters():
                param.requires_grad = True

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def build_action_recognizer(
    num_classes: int = 6,
) -> tuple[FootballActionRecognizer, nn.Sequential]:
    """
    Build EfficientNetB0-based football action recognizer.

    Returns:
        model:    FootballActionRecognizer ready for training
        backbone: model.features (for fine-tuning control / GradCAM)
    """
    model = FootballActionRecognizer(num_classes=num_classes)
    return model, model.features
