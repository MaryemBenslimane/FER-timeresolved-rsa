"""
VGG16 network 


"""
import os
import torch
from torch import nn

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import weights_path

CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
VGGFACE_CONV = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3",
                "conv4_1", "conv4_2", "conv4_3", "conv5_1", "conv5_2", "conv5_3"]


def make_layers(n_input_channels=3):
    layers, channels = [], n_input_channels
    for v in CFG:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers += [nn.Conv2d(channels, v, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            channels = v
    return nn.Sequential(*layers)


class VGG16(nn.Module):
    """VGG16 without batch normalization.
    Keyword Arguments:
        num_classes {int} -- Number of output classes. (default: {1000}; 2622 for VGG-Face)
        n_input_channels {int} -- Input channels of the first conv: 3 for RGB, 1 for grayscale.
        dropout_prob {float} -- Dropout in the classifier. (default: {0.5})
    """
    def __init__(self, num_classes: int = 1000, n_input_channels: int = 3, dropout_prob: float = 0.5) -> None:
        super().__init__()
        self.features = make_layers(n_input_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(dropout_prob),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(dropout_prob),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(self.features(x))
        return self.classifier(torch.flatten(x, 1))


def convert_state(state):
    """Strip `module.`; map MatConvNet VGG-Face names to torchvision names."""
    state = {k.removeprefix("module."): v for k, v in state.items()}
    if not any(k.startswith("conv1_1.") for k in state):
        return state
    conv_indexes = [i for i, m in enumerate(make_layers()) if isinstance(m, nn.Conv2d)]
    out = {}
    for name, idx in zip(VGGFACE_CONV, conv_indexes):
        for s in ("weight", "bias"):
            out[f"features.{idx}.{s}"] = state[f"{name}.{s}"]
    for name, idx in (("fc6", 0), ("fc7", 3), ("fc8", 6)):
        for s in ("weight", "bias"):
            out[f"classifier.{idx}.{s}"] = state[f"{name}.{s}"]
    return out


def load_state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model_state_dict"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
            break
    return convert_state(obj)


def build(pretrained: bool = False, num_classes: int = 1000, n_input_channels: int = 3,
          transfer: bool = False, weights: str = None) -> VGG16:
    """pretrained -> load `weights` (a path, or a file name under net_weights/) strictly.
    transfer -> keep the checkpoint weights except the last Linear and give the model a new
    `num_classes` output; a 3-channel first conv is summed over channels when n_input_channels is 1."""
    model = VGG16(num_classes, n_input_channels)
    if not pretrained:
        return model
    path = weights if os.path.isfile(weights) else os.path.join(weights_path, weights)
    state = load_state(path)
    if transfer:
        state = {k: v for k, v in state.items() if not k.startswith("classifier.6.")}
        if n_input_channels == 1 and state["features.0.weight"].shape[1] == 3:
            state["features.0.weight"] = state["features.0.weight"].sum(dim=1, keepdim=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("classifier.6.") for k in missing):
            raise RuntimeError(f"{path} does not match VGG16: missing={missing}, unexpected={unexpected}")
        return model
    model.load_state_dict(state)
    return model


def VGG16_ImageNet(pretrained: bool = False, num_classes: int = 1000, n_input_channels: int = 3,
                   transfer: bool = False, weights: str = "vgg16_imagenet") -> VGG16:
    return build(pretrained, num_classes, n_input_channels, transfer, weights)


def VGGFace(pretrained: bool = False, num_classes: int = 2622, transfer: bool = False,
            weights: str = "vgg16_vggface") -> VGG16:
    return build(pretrained, num_classes, 3, transfer, weights)
