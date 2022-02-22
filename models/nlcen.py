from matplotlib.pyplot import axis
import torch.nn as nn
import torch
import torch.nn.functional as F

from nlce import NLCE
from resnet import Bottleneck, BasicBlock


class Network(nn.Module):
    def __init__(self, block, num_blocks):
        super(Network, self).__init__()
        self.in_planes = 64

        # Process Conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # NLCE modules
        self.nlce2 = NLCE(C_in=256, C1=128)
        self.nlce3 = NLCE(C_in=512, C1=128)
        self.nlce4 = NLCE(C_in=1024, C1=128)
        self.nlce5 = NLCE(C_in=2048, C1=128)

        # Lateral layers
        self.latlayer2 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer5 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.smooth5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

        # Bottleneck operations
        self.bottle3 = self._make_bottleneck(256, 1)
        self.bottle4 = self._make_bottleneck(256, 2)
        self.bottle5 = self._make_bottleneck(256, 3)
        self.bottle = nn.Conv2d(480, 1, kernel_size=1)

    def _make_bottleneck(self, channel_in, times):
        layers = []
        c = channel_in
        for _ in range(times):
            layers.append(nn.Conv2d(c, c//2, kernel_size=1))
            c //= 2
        
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
        p5 = self.latlayer5(e5)
        p4 = self._upsample_add(p5, self.latlayer4(e4))
        p3 = self._upsample_add(p4, self.latlayer3(e3))
        p2 = self._upsample_add(p3, self.latlayer2(e2))

        # Segmentation from feature maps
        p5_s = self.F.upsample(self.smooth5(p5), size=(H, W), mode='bilinear')
        p4_s = self.F.upsample(self.smooth4(p4), size=(H, W), mode='bilinear')
        p3_s = self.F.upsample(self.smooth3(p3), size=(H, W), mode='bilinear')
        p2_s = self.F.upsample(self.smooth2(p2), size=(H, W), mode='bilinear')

        # Bottleneck operations
        p3 = self.bottle3(p3)
        p4 = self.bottle4(p4)
        p5 = self.bottle5(p5)

        out = torch.cat((p2, p3, p4, p5), dim=1)
        out = self.bottle(out)
        
        return out, p2_s, p3_s, p4_s, p5_s
