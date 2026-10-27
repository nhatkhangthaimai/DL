"""
Face Anti-Spoofing Deep Learning Architectures
ResNet34 Baseline & ResNet34 + LBP Texture Feature Fusion Network
Author: Team STT 12
"""

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from typing import Dict, Any, Optional, Tuple


class ResNet34Baseline(nn.Module):
    """
    Baseline Deep Learning Model: ResNet34 pretrained on ImageNet
    without handcrafted texture fusion.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.4,
        hidden_dim: int = 256
    ):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        base_model = resnet34(weights=weights)

        # Retain convolutional layers and adaptive pooling
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        in_features = base_model.fc.in_features  # 512 for ResNet34

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 512-dim deep feature representation."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for pure ResNet34."""
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits

    def freeze_backbone(self):
        """Freezes all ResNet backbone weights to train only classification head."""
        for param in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in param.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes entire backbone for full fine-tuning."""
        for param in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in param.parameters():
                p.requires_grad = True


class ResNet34LBPFusion(nn.Module):
    """
    Main Proposed Architecture: Dual-Stream Deep Feature + LBP Texture Fusion Network.
    - Stream A: ResNet34 deep visual representation (512-dim)
    - Stream B: Multi-block Local Binary Patterns texture descriptor (160 or 944-dim)
    - Fusion: Concatenation -> MLP Fusion Head -> Binary Classification (Live / Spoof)
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        lbp_feature_dim: int = 944,
        resnet_feature_dim: int = 512,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.4
    ):
        super().__init__()
        self.lbp_feature_dim = lbp_feature_dim
        self.resnet_feature_dim = resnet_feature_dim

        # Stream A: Pretrained ResNet34 Backbone
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        base_model = resnet34(weights=weights)

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        # Stream B: Texture Projection (optional batchnorm/linear layer for texture balance)
        self.texture_bn = nn.BatchNorm1d(lbp_feature_dim)

        # Fusion Stream
        fusion_in_dim = resnet_feature_dim + lbp_feature_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden_dim, num_classes)
        )

    def extract_deep_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 512-dim features from RGB face tensor."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, rgb_images: torch.Tensor, lbp_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with feature fusion:
        - rgb_images: [Batch, 3, 224, 224]
        - lbp_features: [Batch, lbp_feature_dim]
        returns: logits [Batch, num_classes] (0: LIVE, 1: SPOOF)
        """
        # Deep representation
        deep_feat = self.extract_deep_features(rgb_images)  # [B, 512]

        # Normalized texture representation
        tex_feat = self.texture_bn(lbp_features)  # [B, lbp_dim]

        # Multimodal feature fusion via concatenation
        fused = torch.cat([deep_feat, tex_feat], dim=1)  # [B, 512 + lbp_dim]

        # Prediction logits
        logits = self.fusion_head(fused)
        return logits

    def freeze_backbone(self):
        """Freezes ResNet backbone to warm up fusion head."""
        for param in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in param.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes backbone for end-to-end fine-tuning."""
        for param in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in param.parameters():
                p.requires_grad = True


def build_model(config: Dict[str, Any], model_type: str = "fusion") -> nn.Module:
    """Factory helper to construct either ResNet34 baseline or ResNet34+LBP fusion model."""
    m_cfg = config.get("model", {})
    num_classes = m_cfg.get("num_classes", 2)
    pretrained = m_cfg.get("pretrained", True)
    dropout = m_cfg.get("dropout", 0.4)
    hidden_dim = m_cfg.get("fusion_hidden_dim", 256)

    if model_type.lower() == "baseline":
        model = ResNet34Baseline(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            hidden_dim=hidden_dim
        )
    else:
        lbp_dim = m_cfg.get("lbp_feature_dim", 944)
        model = ResNet34LBPFusion(
            num_classes=num_classes,
            pretrained=pretrained,
            lbp_feature_dim=lbp_dim,
            resnet_feature_dim=m_cfg.get("resnet_feature_dim", 512),
            fusion_hidden_dim=hidden_dim,
            dropout=dropout
        )

    return model
