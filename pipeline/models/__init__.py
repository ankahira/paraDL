"""A ResNet implementation but using :class:`nn.Sequential`. :func:`resnet101`
returns a :class:`nn.Sequential` instead of ``ResNet``.

This code is transformed :mod:`torchvision.models.resnet`.

"""
from collections import OrderedDict
from typing import Any, List

from torch import nn
import torchvision.models as models

from models.bottleneck import bottleneck
from models.flatten_sequential import flatten_sequential



__all__ = ['resnet101']


def build_resnet(layers: List[int],
                 num_classes: int = 1000,
                 inplace: bool = False
                 ) -> nn.Sequential:
    """Builds a ResNet as a simple sequential model.

    Note:
        The implementation is copied from :mod:`torchvision.models.resnet`.

    """
    inplanes = 64

    def make_layer(planes: int,
                   blocks: int,
                   stride: int = 1,
                   inplace: bool = False,
                   ) -> nn.Sequential:
        nonlocal inplanes

        downsample = None
        if stride != 1 or inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

        layers = []
        layers.append(bottleneck(inplanes, planes, stride, downsample, inplace))
        inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(bottleneck(inplanes, planes, inplace=inplace))

        return nn.Sequential(*layers)

    # Build ResNet as a sequential model.
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
        ('bn1', nn.BatchNorm2d(64)),
        ('relu', nn.ReLU()),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

        ('layer1', make_layer(64, layers[0], inplace=inplace)),
        ('layer2', make_layer(128, layers[1], stride=2, inplace=inplace)),
        ('layer3', make_layer(256, layers[2], stride=2, inplace=inplace)),
        ('layer4', make_layer(512, layers[3], stride=2, inplace=inplace)),

        ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
        ('flat', nn.Flatten()),
        ('fc', nn.Linear(512 * 4, num_classes)),
    ]))

    # Flatten nested sequentials.
    model = flatten_sequential(model)

    # Initialize weights for Conv2d and BatchNorm2d layers.
    # Stolen from torchvision-0.4.0.
    def init_weight(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            return

        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            return

    model.apply(init_weight)

    return model


def resnet101(**kwargs: Any) -> nn.Sequential:
    """Constructs a ResNet-101 model."""
    return build_resnet([3, 4, 23, 3], **kwargs)

def resnet152(**kwargs: Any) -> nn.Sequential:
    """Constructs a Renset- 152 Model """
    return build_resnet([3, 8, 36, 3], **kwargs)



cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg():
    batch_norm = False
    layers = []
    in_channels = 3
    for v in cfg["D"]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    layers.append(nn.Flatten())
    layers.append(nn.Linear(512 * 7 * 7, 4096))
    layers.append(nn.ReLU(True))
    layers.append( nn.Dropout())
    layers.append( nn.Linear(4096, 4096))
    layers.append( nn.ReLU(True))
    layers.append( nn.Dropout())
    layers.append(nn.Linear(4096, 1000))
    return nn.Sequential(*layers)









