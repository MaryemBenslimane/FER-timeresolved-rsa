"""
ResNet18 / ResNet50 networks

"""
import os
import re
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import weights_path

CONFIGS = {18: (BasicBlock, [2, 2, 2, 2]), 50: (Bottleneck, [3, 4, 6, 3])}


def make_stage(block, in_planes, planes, n_blocks, stride):
    downsample = None
    if stride != 1 or in_planes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers = [block(in_planes, planes, stride, downsample)]
    layers += [block(planes * block.expansion, planes) for _ in range(1, n_blocks)]
    return nn.Sequential(*layers)


class Backbone(nn.Module):

    def __init__(self, depth, n_input_channels=3):
        super().__init__()
        block, n_blocks = CONFIGS[depth]
        self.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        planes, in_planes = [64, 128, 256, 512], 64
        for i, (p, n) in enumerate(zip(planes, n_blocks)):
            setattr(self, f"block{i + 1}", make_stage(block, in_planes, p, n, 1 if i == 0 else 2))
            in_planes = p * block.expansion
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.out_features = in_planes

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.block4(self.block3(self.block2(self.block1(x))))
        return torch.flatten(self.avgpool(x), 1)


class ResNet(nn.Module):
    """ResNet18 or ResNet50.
    Keyword Arguments:
        depth {int} -- 18 or 50.
        num_classes {int} -- Number of output classes of the `classifier` head. (default: {1000})
        n_input_channels {int} -- Input channels of conv1: 3 for RGB, 1 for grayscale.
    """
    def __init__(self, depth: int = 18, num_classes: int = 1000, n_input_channels: int = 3) -> None:
        super().__init__()
        self.model = Backbone(depth, n_input_channels)
        self.classifier = nn.Linear(self.model.out_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.classifier(self.model(x))


def from_torchvision(state):
    """torchvision key names (conv1, layer1.., fc) -> this layout (model.conv1, model.block1.., classifier)."""
    out = {}
    for k, v in state.items():
        k = k.removeprefix("module.")
        if k.startswith("fc."):
            out["classifier." + k[3:]] = v
        elif not k.startswith(("model.", "classifier.")):
            out["model." + re.sub(r"^layer(\d)\.", r"block\1.", k)] = v
        else:
            out[k] = v
    return out


def load_state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
            break
    return from_torchvision(obj)


def build(depth: int, pretrained: bool = False, num_classes: int = 1000, n_input_channels: int = 3,
          transfer: bool = False, weights: str = None) -> ResNet:
    """pretrained -> load `weights` (a path, or a file name under net_weights/) strictly.
    transfer -> keep the checkpoint backbone and give the model a new `num_classes` head; a
    3-channel conv1 is summed over channels when n_input_channels is 1."""
    model = ResNet(depth, num_classes, n_input_channels)
    if not pretrained:
        return model
    path = weights if os.path.isfile(weights) else os.path.join(weights_path, weights)
    state = load_state(path)
    if transfer:
        state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
        if n_input_channels == 1 and state["model.conv1.weight"].shape[1] == 3:
            state["model.conv1.weight"] = state["model.conv1.weight"].sum(dim=1, keepdim=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("classifier.") for k in missing):
            raise RuntimeError(f"{path} does not match ResNet{depth}: missing={missing}, unexpected={unexpected}")
        return model
    model.load_state_dict(state)
    return model


def ResNet18(pretrained: bool = False, num_classes: int = 1000, n_input_channels: int = 3,
             transfer: bool = False, weights: str = None) -> ResNet:
    return build(18, pretrained, num_classes, n_input_channels, transfer, weights)


def ResNet50(pretrained: bool = False, num_classes: int = 1000, n_input_channels: int = 3,
             transfer: bool = False, weights: str = None) -> ResNet:
    return build(50, pretrained, num_classes, n_input_channels, transfer, weights)
