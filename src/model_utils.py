"""Model architectures used in Part 2 (Model Building).

- ``SimpleCNN``: hand-built baseline (4 conv blocks, batch norm, dropout), from scratch.
- ``build_efficientnet_v2_s``: ImageNet-pretrained EfficientNet-V2-S adapted to 32x32
  inputs (stem stride 1 fix) with a swapped 10-way head.
- ``WideResNet``: paper-compliant WRN-28-10 (Zagoruyko & Komodakis, 2016) — pre-activation
  blocks, width factor 10, dropout 0.3 in every block — trained from scratch.
"""

import urllib.error
from pathlib import Path

import torch
import torch.hub
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

__all__ = ["SimpleCNN", "build_efficientnet_v2_s", "WRNBasicBlock", "WideResNet"]


class SimpleCNN(nn.Module):
    """Small CNN from scratch: 4 conv blocks (32->64->128->256) + 2 FC layers."""

    def __init__(self, num_classes=10, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def build_efficientnet_v2_s(num_classes=10, device=None):
    """ImageNet-pretrained EfficientNet-V2-S adapted for CIFAR-10.

    Adaptations (all required for correctness on 32x32 inputs):
      1. stem stride 2 -> 1, so the first conv does not collapse 32x32 detail;
      2. 1000-class ImageNet head -> 10-way head (body stays frozen-pretrained);
      3. classifier-head dropout 0.2 -> 0.3 (anti-overfitting, see Part 2 notes).

    Falls back to the local torch.hub cache when the network is unavailable.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = models.efficientnet_v2_s(weights="DEFAULT")
    except urllib.error.URLError:
        cache_file = (Path(torch.hub.get_dir()) / "checkpoints"
                      / "efficientnet_v2_s-dd5fe13b.pth")
        model = models.efficientnet_v2_s(weights=None)
        model.load_state_dict(torch.load(cache_file, map_location="cpu", weights_only=True))
        print("Network unavailable -- loaded EfficientNet-V2-S weights from local cache.")

    model.features[0][0].stride = (1, 1)                    # 1) keep 32x32 spatial detail
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)  # 2) CIFAR-10 head
    for module in model.classifier:
        if isinstance(module, nn.Dropout):
            module.p = 0.3                                  # 3) raise head dropout 0.2 -> 0.3
    return model.to(device)


class WRNBasicBlock(nn.Module):
    """Pre-activation residual block with optional 1x1 shortcut and dropout 0.3."""

    def __init__(self, in_planes, out_planes, stride=1, dropout_rate=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    """Wide ResNet (Zagoruyko & Komodakis, 2016) for 32x32 CIFAR-style inputs."""

    def __init__(self, depth=28, widen_factor=10, num_classes=10):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth must be of the form 6n+4"
        num_blocks = (depth - 4) // 6                     # 4 blocks per stage for depth=28
        base = 16 * widen_factor                          # 160 channels at the first stage

        self.conv1 = nn.Conv2d(3, base, 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(base, base, num_blocks, stride=1)
        self.layer2 = self._make_layer(base, base * 2, num_blocks, stride=2)
        self.layer3 = self._make_layer(base * 2, base * 4, num_blocks, stride=2)
        self.bn = nn.BatchNorm2d(base * 4)
        self.fc = nn.Linear(base * 4, num_classes)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride):
        layers = [WRNBasicBlock(in_planes, out_planes, stride)]
        layers += [WRNBasicBlock(out_planes, out_planes) for _ in range(num_blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, out.size(-1))             # global average pooling
        out = out.view(out.size(0), -1)
        return self.fc(out)
