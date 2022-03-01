from configparser import Interpolation
from matplotlib.pyplot import axis
import torch.nn as nn
import torch
import torch.nn.functional as F

from models.nlce import NLCE
from models.resnet import Bottleneck, BasicBlock


class Network(nn.Module):
    def __init__(self, block, num_blocks, in_channels):
        super(Network, self).__init__()
        self.in_planes = 64

        # Process Conv1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # NLCE modules
        self.nlce2 = NLCE(C_in=64)
        self.nlce3 = NLCE(C_in=128)
        self.nlce4 = NLCE(C_in=256)
        self.nlce5 = NLCE(C_in=512)

        # Lateral layers
        self.latlayer2 = nn.Conv2d( 64, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer5 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # To supervision
        self.sup2 = self._make_bottleneck(2, False)
        self.sup3 = self._make_bottleneck(2, False)
        self.sup4 = self._make_bottleneck(2, False)
        self.sup5 = self._make_bottleneck(2, False)

        # Bottleneck operations and concat
        self.bottle3 = self._make_bottleneck(32, True)
        self.bottle4 = self._make_bottleneck(16, True)
        self.bottle5 = self._make_bottleneck(8, True)
        self.bottle = nn.Sequential(
                        nn.Conv2d(120, 60, kernel_size=1),
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.Conv2d(60, 30, kernel_size=1),
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.Conv2d(30, 2, kernel_size=1)
        )

    def _make_bottleneck(self, out_channel, upsample):
        layers = []

        c = 64
        while c > out_channel:
            layers.append(nn.Conv2d(c, c//2, kernel_size=1, stride=1, padding=0))
            c = c // 2
            if upsample:
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        _, _, H, W = x.size()
        # Bottom-up
        c1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # NLCE modules
        e2 = self.nlce2(c2)
        e3 = self.nlce3(c3)
        e4 = self.nlce4(c4)
        e5 = self.nlce5(c5)

        # Top-down
        p5 = self.smooth5(self.latlayer5(e5))
        p4 = self.smooth4(self._upsample_add(p5, self.latlayer4(e4)))
        p3 = self.smooth3(self._upsample_add(p4, self.latlayer3(e3)))
        p2 = self.smooth2(self._upsample_add(p3, self.latlayer2(e2)))

        # Supervision
        s2 = F.softmax(self.sup2(p2), dim=1)
        s3 = F.softmax(self.sup3(p3), dim=1)
        s4 = F.softmax(self.sup4(p4), dim=1)
        s5 = F.softmax(self.sup5(p5), dim=1)

        # Bottleneck operations and concat
        p5 = self.bottle5(p5)
        p4 = self.bottle4(p4)
        p3 = self.bottle3(p3)

        out = self.bottle(torch.cat((p2, p3, p4, p5), 1))
        out = F.softmax(out, dim=1)
        
        return out, s2, s3, s4, s5

def Network_channel1():
    return Network(BasicBlock, [3, 4, 6, 3], 1)

def Network_channel3():
    return Network(BasicBlock, [3, 4, 6, 3], 3)
