from __future__ import absolute_import

from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import sys
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    """
    one 3x3 convolution with padding\n
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        # print("--------------------"+ str(in_planes) +"," + str(planes))        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Cross_Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Cross_Wide_ResNet, self).__init__()
        self.inplanes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1_1 = conv3x3(3,nStages[0])
        self.conv1_2 = conv3x3(3,nStages[0])
        
        blocksSize = [(32, 32), (32, 16), (16, 8)]
        channels = [16*k, 32*k, 64*k]
        strides = [1, 2, 2]
        self.net1Blocks = nn.ModuleList()
        self.net1CrossNet = nn.ModuleList()
        self.net2Blocks = nn.ModuleList()
        self.net2CrossNet = nn.ModuleList()
        for stage in range(3):
            bkplanes = self.inplanes
            self.net1Blocks.append(self._wide_layer(wide_basic, channels[stage], n, dropout_rate, strides[stage]))
            self.inplanes = bkplanes
            self.net2Blocks.append(self._wide_layer(wide_basic, channels[stage], n, dropout_rate, strides[stage]))
            stageCrossNet1 = nn.ModuleList()
            stageCrossNet2 = nn.ModuleList()
            for to in range(stage+1, 3):
                stageCrossNet1.append(self._make_fusion_layer(channels[stage], channels[to], blocksSize[stage][1], int(blocksSize[stage][1]/blocksSize[to][1])))
                stageCrossNet2.append(self._make_fusion_layer(channels[stage], channels[to], blocksSize[stage][1], int(blocksSize[stage][1]/blocksSize[to][1])))
            self.net1CrossNet.append(stageCrossNet1)
            self.net2CrossNet.append(stageCrossNet2)      

        self.bn1_1 = nn.BatchNorm2d(nStages[3])
        self.bn1_2 = nn.BatchNorm2d(nStages[3])
        self.classfier3_1 = nn.Linear(nStages[3], num_classes)
        self.classfier3_2 = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.inplanes, planes, dropout_rate, stride))
            self.inplanes = planes

        return nn.Sequential(*layers)
    
    def _make_fusion_layer(self, in_planes, out_planes, in_size, minification):
        layers = []
        layers.append(nn.BatchNorm2d(in_planes))
        layers.append(nn.Conv2d(in_planes, out_planes, minification, minification, padding=0, bias=False))
        # layers.append(nn.AvgPool2d(minification, minification))
        layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        fmap = []
        crossFusionKnowledge = []

        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)

        net1Knowledge = []
        net2Knowledge = []
        for stage in range(3):
            x1 = self.net1Blocks[stage](x1)
            x2 = self.net2Blocks[stage](x2)
            
            temp1 = x1
            temp2 = x2
            for preNum in range(0, stage):
                temp1 = temp1 + net1Knowledge[preNum][stage-preNum-1]
                temp2 = temp2 + net2Knowledge[preNum][stage-preNum-1]
            crossFusionKnowledge.append((torch.flatten(temp1,1), torch.flatten(temp2,1)))

            stageKnowledge1 = []
            stageKnowledge2 = []
            for to in range(stage+1, 3):
                stageKnowledge1.append(self.net1CrossNet[stage][to-stage-1](x1))
                stageKnowledge2.append(self.net2CrossNet[stage][to-stage-1](x2))
            net1Knowledge.append(stageKnowledge1)
            net2Knowledge.append(stageKnowledge2)

        x1 = F.relu(self.bn1_1(x1), True)
        x2 = F.relu(self.bn1_2(x2), True)
        fmap.append(x1)
        fmap.append(x2)

        x1 = F.avg_pool2d(x1, 8)
        x1 = x1.view(x1.size(0), -1)
        x2 = F.avg_pool2d(x2, 8)
        x2 = x2.view(x2.size(0), -1)

        x1 = self.classfier3_1(x1)
        x2 = self.classfier3_2(x2)
        return x1, x2, crossFusionKnowledge, fmap


class Fusion_module(nn.Module):
    def __init__(self,channel,numclass,sptial):
        super(Fusion_module, self).__init__()
        self.fc2   = nn.Linear(channel, numclass)
        self.conv1 =  nn.Conv2d(channel*2, channel*2, kernel_size=3, stride=1, padding=1, groups=channel*2, bias=False)
        self.bn1 = nn.BatchNorm2d(channel * 2)
        self.conv1_1 = nn.Conv2d(channel*2, channel, kernel_size=1, groups=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channel)


        self.sptial = sptial


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #self.avg = channel
    def forward(self, x,y):
        atmap = []
        input = torch.cat((x,y),1)

        x = F.relu(self.bn1((self.conv1(input))))
        x = F.relu(self.bn1_1(self.conv1_1(x)))

        atmap.append(x)  
        x = F.avg_pool2d(x, self.sptial)
        x = x.view(x.size(0), -1)

        out = self.fc2(x)
        atmap.append(out)

        return out


def cross_wide_resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return Cross_Wide_ResNet(**kwargs)
