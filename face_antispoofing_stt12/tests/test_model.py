"""
Unit Tests for Deep Learning Model Architectures
Author: Team STT 12
"""

import pytest
import torch
import torch.nn as nn

from src.model import ResNet34Baseline, ResNet34LBPFusion, build_model


def test_baseline_resnet34_forward():
    model = ResNet34Baseline(num_classes=2, pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)

    assert logits.shape == (4, 2)
    assert not torch.isnan(logits).any()


def test_fusion_resnet34_lbp_forward():
    lbp_dim = 944
    model = ResNet34LBPFusion(num_classes=2, pretrained=False, lbp_feature_dim=lbp_dim)

    x_rgb = torch.randn(4, 3, 224, 224)
    x_lbp = torch.randn(4, lbp_dim)
    logits = model(x_rgb, x_lbp)

    assert logits.shape == (4, 2)
    assert not torch.isnan(logits).any()


def test_gradient_flow_and_backprop():
    lbp_dim = 944
    model = ResNet34LBPFusion(num_classes=2, pretrained=False, lbp_feature_dim=lbp_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x_rgb = torch.randn(2, 3, 224, 224)
    x_lbp = torch.randn(2, lbp_dim)
    targets = torch.tensor([0, 1], dtype=torch.long)

    logits = model(x_rgb, x_lbp)
    loss = criterion(logits, targets)
    loss.backward()

    # Check that fusion head weights received gradients
    has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.fusion_head.parameters())
    assert has_grad, "Fusion head did not receive non-zero gradients!"

    optimizer.step()


def test_freeze_and_unfreeze_backbone():
    model = ResNet34LBPFusion(num_classes=2, pretrained=False)

    model.freeze_backbone()
    for param in model.conv1.parameters():
        assert not param.requires_grad
    for param in model.layer1.parameters():
        assert not param.requires_grad

    # Fusion head must remain trainable
    for param in model.fusion_head.parameters():
        assert param.requires_grad

    model.unfreeze_backbone()
    for param in model.conv1.parameters():
        assert param.requires_grad
    for param in model.layer1.parameters():
        assert param.requires_grad
