import torch
import torch.nn as nn
from typing import Any

from ._builder import build_model_with_cfg
from .registry import register_model


__all__ = ["AlexNet", "alexnet"]

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3):
        self.num_classes = num_classes
        self.in_chans = in_chans

        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_chans, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6400, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, self.num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def _create_alexnet(variant, pretrained=False, **kwargs):
    """
    Constructs an alexnet model
    """
    model_kwargs = dict(
        **kwargs,
    )
    print(model_kwargs)
    return build_model_with_cfg(
        AlexNet,
        variant,
        pretrained,
        **model_kwargs,
    )


@register_model
def alexnet(pretrained=False, **kwargs) -> AlexNet:
    """ HMAX """
    model = _create_alexnet('alexnet', pretrained=pretrained)
    return model