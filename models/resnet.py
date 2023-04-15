from typing import List, Optional, Type

import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    '''3x3 convolution with padding'''
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    '''1x1 convolution'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    '''Residual Block'''
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ) -> None:
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the basic Residual Block.'''
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super(PreActBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        num_layers: int,
        blocks: List[int],
        channels: List[int],
        pool_size: int,
        num_classes: int = 10
    ) -> None:
        super(ResNet, self).__init__()

        n = num_layers
        fc_in = channels[n-1] * ((32 // (pool_size * 2**(n-1))) ** 2)
        strides = [2 if i > 0 else 1 for i in range(n)]

        self.in_planes = channels[0]
        self.conv1 = conv3x3(3, self.in_planes)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.pool_size = pool_size
        self.layers = []

        for i in range(n):
            exec("self.layer{} = self._make_layer(block, channels[{}], blocks[{}], {})"
                 .format(i+1, i, i, strides[i]))
            exec("self.layers.append(self.layer{})".format(i+1))

        self.avgpool = nn.functional.avg_pool2d
        self.fc = nn.Linear(fc_in, num_classes)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        for layer in self.layers:
            out = layer(out)

        out = self.avgpool(out, self.pool_size)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def resnet10() -> ResNet:
    layers = 4
    blocks = [1, 1, 1, 1]
    channels = [64, 128, 256, 512]
    pool_size = 4
    return ResNet(BasicBlock, layers, blocks, channels, pool_size)


def resnet18() -> ResNet:
    layers = 4
    blocks = [2, 2, 2, 2]
    channels = [64, 157, 211, 270]
    pool_size = 4
    return ResNet(BasicBlock, layers, blocks, channels, pool_size)


def resnet20() -> ResNet:
    layers = 4
    blocks = [2, 2, 2, 3]
    channels = [64, 128, 185, 245]
    pool_size = 4
    return ResNet(BasicBlock, layers, blocks, channels, pool_size)


def resnet22_1() -> ResNet:
    layers = 4
    blocks = [3, 3, 2, 3]
    channels = [64, 128, 128, 256]
    pool_size = 4
    return ResNet(BasicBlock, layers, blocks, channels, pool_size)


def resnet22_2() -> ResNet:
    layers = 4
    blocks = [3, 2, 3, 2]
    channels = [64, 128, 192, 256]
    pool_size = 4
    return ResNet(BasicBlock, layers, blocks, channels, pool_size)


def resnet32() -> ResNet:
    layers = 4
    blocks = [4, 4, 4, 3]
    channels = [32, 64, 128, 256]
    pool_size = 4
    return ResNet(BasicBlock, layers, blocks, channels, pool_size)


def test():
    net = resnet32()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
    trainable_params = sum(p.numel()
                           for p in net.parameters() if p.requires_grad)
    print(trainable_params)


# test()
