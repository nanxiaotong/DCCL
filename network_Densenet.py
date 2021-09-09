import os
import torch
from torch import Tensor
import torch.nn as nn
# from .utils import load_state_dict_from_url
from typing import Tuple, Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
import math

##################################################### densenet #############################################
norm_mean, norm_var = 0.0, 1.0

cov_cfg=[(3*i+1) for i in range(12*3+2+1)]


class DenseBasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, dropRate=0):
        super(DenseBasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3,
                               padding=1, bias=False)

        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out

class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out




class CrossKDDenseNet(nn.Module):
    def __init__(self, depth=40, block=DenseBasicBlock,
        dropRate=0, num_classes=1000, growthRate=12, compressionRate=2):
        super(CrossKDDenseNet, self).__init__()
        self.total_feature_maps = {}


        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3 if depth==40 else (depth - 4) // 6
        transition = Transition

        self.covcfg=cov_cfg

        self.growthRate = growthRate
        self.dropRate = dropRate

        self.inplanes = growthRate * 2



        self.conv1_1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False) # 3->24
        self.conv1_2 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False) # 3->24
        # self.dense1_1 = self._make_denseblock(block, n)
        # self.trans1_1 = self._make_transition(transition, compressionRate)
        # self.dense2_1 = self._make_denseblock(block, n)
        # self.trans2_1 = self._make_transition(transition, compressionRate)
        # self.dense3_1 = self._make_denseblock(block, n) #在cifar上只有三个block
        # self.trans3_1 = self._make_transition(transition, compressionRate)
        self.bn_1 = nn.BatchNorm2d(self.inplanes)
        self.bn_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(129, num_classes)
        self.fc2 = nn.Linear(129, num_classes)

        blocksSize = [(112,56), (56, 28), (28, 14)]
        channels = [84,114,129]
        # strides = [1, 2, 2, 2]

        self.net1Blocks = nn.ModuleList()
        self.net1CrossNet = nn.ModuleList()
        self.net2Blocks = nn.ModuleList()
        self.net2CrossNet = nn.ModuleList()


        for stage in range(3):
            bkplanes = self.inplanes
            self.net1Blocks.append(nn.Sequential(*[self._make_denseblock(block, n),self._make_transition(transition, compressionRate)]))
            self.inplanes = bkplanes
            self.net2Blocks.append(nn.Sequential(*[self._make_denseblock(block, n),self._make_transition(transition, compressionRate)]))
            stageCrossNet1 = nn.ModuleList()
            stageCrossNet2 = nn.ModuleList()
            for to in range(stage + 1, 3):
                stageCrossNet1.append(self._make_fusion_layer(channels[stage], channels[to], blocksSize[stage][1],
                                                              int(blocksSize[stage][1] / blocksSize[to][1])))
                stageCrossNet2.append(self._make_fusion_layer(channels[stage], channels[to], blocksSize[stage][1],
                                                              int(blocksSize[stage][1] / blocksSize[to][1])))
            self.net1CrossNet.append(stageCrossNet1)
            self.net2CrossNet.append(stageCrossNet2)
        #
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        # self.fc2 = nn.Linear(512 * block.expansion, num_classes)

        self.reset_parameters()
        self.register_hook()

    def _make_denseblock(self, block, blocks):
        layers = []
        inplanes = self.inplanes
        for i in range(blocks):
            layers.append(
                block(inplanes + i * self.growthRate, outplanes=int(self.growthRate), dropRate=self.dropRate))

        self.inplanes = inplanes + blocks * self.growthRate  # 到下一层了
        return nn.Sequential(*layers)

    def _make_transition(self, transition, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return transition(inplanes, outplanes)

    def reset_parameters(self):

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def register_hook(self):

        self.extract_layers = ['dense1', 'dense2', 'relu']

        def get_activation(maps, name):
            def get_output_hook(module, input, output):
                maps[name] = output

            return get_output_hook

        def add_hook(model, maps, extract_layers):
            for name, module in model.named_modules():
                if name in extract_layers:
                    module.register_forward_hook(get_activation(maps, name))

        add_hook(self, self.total_feature_maps, self.extract_layers)

    def _make_fusion_layer(self, in_planes, out_planes, in_size, minification):
        layers = []
        layers.append(nn.Conv2d(in_planes, out_planes, minification, minification, padding=0, bias=False))
        # layers.append(nn.AvgPool2d(minification, minification))
        layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)



    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x1 = self.conv1_1(x)

        x2 = self.conv1_2(x)


        net1Knowledge = []
        net2Knowledge = []
        for stage in range(3):
            x1 = self.net1Blocks[stage](x1)
            # print(x1.size())
            x2 = self.net2Blocks[stage](x2)

            crossFusionKnowledge = []
            temp1 = x1
            temp2 = x2
            for preNum in range(0, stage):
                temp1 = temp1 + net1Knowledge[preNum][stage - preNum - 1]
                temp2 = temp2 + net2Knowledge[preNum][stage - preNum - 1]
            crossFusionKnowledge.append((torch.flatten(temp1, 1), torch.flatten(temp2, 1)))

            stageKnowledge1 = []
            stageKnowledge2 = []
            for to in range(stage + 1, 3):
                stageKnowledge1.append(self.net1CrossNet[stage][to - stage - 1](x1))
                stageKnowledge2.append(self.net2CrossNet[stage][to - stage - 1](x2))
            net1Knowledge.append(stageKnowledge1)
            net2Knowledge.append(stageKnowledge2)

        fmap = []
        fmap.append(x1)
        fmap.append(x2)

        '''
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        '''
        # print(x1.size())
        x1 = self.relu(x1)
        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc(x1)

        x2 = self.relu(x2)
        x2 = self.avgpool2(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc(x2)
        return x1, x2, crossFusionKnowledge, fmap

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, List, List]:
        return self._forward_impl(x)
