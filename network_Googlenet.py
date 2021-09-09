import os
import torch
from torch import Tensor
import torch.nn as nn
# from .utils import load_state_dict_from_url
from typing import Tuple, Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
import math

########################################### googlenet ###########################################
class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red1, n5x5red2, n5x5, pool_planes, tmp_name):
        super(Inception, self).__init__()
        self.tmp_name=tmp_name

        self.n1x1 = n1x1
        self.n3x3 = n3x3
        self.n5x5 = n5x5
        self.pool_planes = pool_planes

        # 1x1 conv branch
        if self.n1x1:
            conv1x1 = nn.Conv2d(in_planes, n1x1, kernel_size=1)
            conv1x1.tmp_name = self.tmp_name

            self.branch1x1 = nn.Sequential(
                conv1x1,
                nn.BatchNorm2d(n1x1),
                nn.ReLU(True),
            )

        # 1x1 conv -> 3x3 conv branch
        if self.n3x3:
            conv3x3_1=nn.Conv2d(in_planes, n3x3red, kernel_size=1)
            conv3x3_2=nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1)
            conv3x3_1.tmp_name = self.tmp_name
            conv3x3_2.tmp_name = self.tmp_name

            self.branch3x3 = nn.Sequential(
                conv3x3_1,
                nn.BatchNorm2d(n3x3red),
                nn.ReLU(True),
                conv3x3_2,
                nn.BatchNorm2d(n3x3),
                nn.ReLU(True),
            )

        # 1x1 conv -> 5x5 conv branch
        if self.n5x5 > 0:
            conv5x5_1 = nn.Conv2d(in_planes, n5x5red1, kernel_size=1)
            conv5x5_2 = nn.Conv2d(n5x5red1, n5x5red2, kernel_size=3, padding=1)
            conv5x5_3 = nn.Conv2d(n5x5red2, n5x5, kernel_size=3, padding=1)
            conv5x5_1.tmp_name = self.tmp_name
            conv5x5_2.tmp_name = self.tmp_name
            conv5x5_3.tmp_name = self.tmp_name

            self.branch5x5 = nn.Sequential(
                conv5x5_1,
                nn.BatchNorm2d(n5x5red1),
                nn.ReLU(True),
                conv5x5_2,
                nn.BatchNorm2d(n5x5red2),
                nn.ReLU(True),
                conv5x5_3,
                nn.BatchNorm2d(n5x5),
                nn.ReLU(True),
            )

        # 3x3 pool -> 1x1 conv branch
        if self.pool_planes > 0:
            conv_pool = nn.Conv2d(in_planes, pool_planes, kernel_size=1)
            conv_pool.tmp_name = self.tmp_name

            self.branch_pool = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                conv_pool,
                nn.BatchNorm2d(pool_planes),
                nn.ReLU(True),
            )

    def forward(self, x):
        out = []
        y1 = self.branch1x1(x)
        out.append(y1)

        y2 = self.branch3x3(x)
        out.append(y2)

        y3 = self.branch5x5(x)
        out.append(y3)

        y4 = self.branch_pool(x)
        out.append(y4)
        return torch.cat(out, 1)


class cross_kd_googlenet(nn.Module):
    def __init__(self, block=Inception, filters=None, layer_cfg=None, num_classes=1000, leader=False, trans_fusion_info=None):
        super(cross_kd_googlenet, self).__init__()

        self.total_feature_maps = {}
        self.leader = leader
        self.layer_cfg = layer_cfg

        # conv_pre = nn.Conv2d(3, 192, kernel_size=3, padding=1)
        # conv_pre.tmp_name='pre_layer'
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        self.pre_layers2 = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        if filters is None:
            filters = [
                [64, 128, 32, 32],
                [128, 192, 96, 64],
                [192, 208, 48, 64],
                [160, 224, 64, 64],
                [128, 256, 64, 64],
                [112, 288, 64, 64],
                [256, 320, 128, 128],
                [256, 320, 128, 128],
                [384, 384, 128, 128]
            ]

        self.filters=filters

        self.inception_a3 = block(in_planes=192,
                                  n1x1=filters[0][0],
                                  n3x3red=96 if self.layer_cfg is None else self.layer_cfg[0],
                                  n3x3=filters[0][1],
                                  n5x5red1=16 if self.layer_cfg is None else self.layer_cfg[1],
                                  n5x5red2=filters[0][2] if self.layer_cfg is None else self.layer_cfg[2],
                                  n5x5=filters[0][2],
                                  pool_planes=filters[0][3],
                                  tmp_name='a3')
        self.inception_b3 = block(in_planes=sum(filters[0]),
                                  n1x1=filters[1][0],
                                  n3x3red=128 if self.layer_cfg is None else self.layer_cfg[3],
                                  n3x3=filters[1][1],
                                  n5x5red1=32 if self.layer_cfg is None else self.layer_cfg[4],
                                  n5x5red2=filters[1][2] if self.layer_cfg is None else self.layer_cfg[5],
                                  n5x5=filters[1][2],
                                  pool_planes=filters[1][3],
                                  tmp_name='b3')

        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception_a4 = block(in_planes=sum(filters[1]),
                                  n1x1=filters[2][0],
                                  n3x3red=96 if self.layer_cfg is None else self.layer_cfg[6],
                                  n3x3=filters[2][1],
                                  n5x5red1=16 if self.layer_cfg is None else self.layer_cfg[7],
                                  n5x5red2=filters[2][2] if self.layer_cfg is None else self.layer_cfg[8],
                                  n5x5=filters[2][2],
                                  pool_planes=filters[2][3],
                                  tmp_name='a4')
        self.inception_b4 = block(in_planes=sum(filters[2]),
                                  n1x1=filters[3][0],
                                  n3x3red=112 if self.layer_cfg is None else self.layer_cfg[9],
                                  n3x3=filters[3][1],
                                  n5x5red1=24 if self.layer_cfg is None else self.layer_cfg[10],
                                  n5x5red2=filters[3][2] if self.layer_cfg is None else self.layer_cfg[11],
                                  n5x5=filters[3][2],
                                  pool_planes=filters[3][3],
                                  tmp_name='b4')
        self.inception_c4 = block(in_planes=sum(filters[3]),
                                  n1x1=filters[4][0],
                                  n3x3red=128 if self.layer_cfg is None else self.layer_cfg[12],
                                  n3x3=filters[4][1],
                                  n5x5red1=24 if self.layer_cfg is None else self.layer_cfg[13],
                                  n5x5red2=filters[4][2] if self.layer_cfg is None else self.layer_cfg[14],
                                  n5x5=filters[4][2],
                                  pool_planes=filters[4][3],
                                  tmp_name='c4')
        self.inception_d4 = block(in_planes=sum(filters[4]),
                                  n1x1=filters[5][0],
                                  n3x3red=144 if self.layer_cfg is None else self.layer_cfg[15],
                                  n3x3=filters[5][1],
                                  n5x5red1=32 if self.layer_cfg is None else self.layer_cfg[16],
                                  n5x5red2=filters[5][2] if self.layer_cfg is None else self.layer_cfg[17],
                                  n5x5=filters[5][2],
                                  pool_planes=filters[5][3],
                                  tmp_name='d4')
        self.inception_e4 = block(in_planes=sum(filters[5]),
                                  n1x1=filters[6][0],
                                  n3x3red=160 if self.layer_cfg is None else self.layer_cfg[18],
                                  n3x3=filters[6][1],
                                  n5x5red1=32 if self.layer_cfg is None else self.layer_cfg[19],
                                  n5x5red2=filters[6][2] if self.layer_cfg is None else self.layer_cfg[20],
                                  n5x5=filters[6][2],
                                  pool_planes=filters[6][3],
                                  tmp_name='e4')

        self.inception_a5 = block(in_planes=sum(filters[6]),
                                  n1x1=filters[7][0],
                                  n3x3red=160 if self.layer_cfg is None else self.layer_cfg[21],
                                  n3x3=filters[7][1],
                                  n5x5red1=32 if self.layer_cfg is None else self.layer_cfg[22],
                                  n5x5red2=filters[7][2] if self.layer_cfg is None else self.layer_cfg[23],
                                  n5x5=filters[7][2],
                                  pool_planes=filters[7][3],
                                  tmp_name='a5')
        self.inception_b5 = block(in_planes=sum(filters[7]),
                                  n1x1=filters[8][0],
                                  n3x3red=192 if self.layer_cfg is None else self.layer_cfg[24],
                                  n3x3=filters[8][1],
                                  n5x5red1=48 if self.layer_cfg is None else self.layer_cfg[25],
                                  n5x5red2=filters[8][2] if self.layer_cfg is None else self.layer_cfg[26],
                                  n5x5=filters[8][2],
                                  pool_planes=filters[8][3],
                                  tmp_name='b5')

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(sum(filters[-1]), num_classes)



        self.inception_a3_2 = block(in_planes=192,
                                  n1x1=filters[0][0],
                                  n3x3red=96 if self.layer_cfg is None else self.layer_cfg[0],
                                  n3x3=filters[0][1],
                                  n5x5red1=16 if self.layer_cfg is None else self.layer_cfg[1],
                                  n5x5red2=filters[0][2] if self.layer_cfg is None else self.layer_cfg[2],
                                  n5x5=filters[0][2],
                                  pool_planes=filters[0][3],
                                  tmp_name='a3')
        self.inception_b3_2 = block(in_planes=sum(filters[0]),
                                  n1x1=filters[1][0],
                                  n3x3red=128 if self.layer_cfg is None else self.layer_cfg[3],
                                  n3x3=filters[1][1],
                                  n5x5red1=32 if self.layer_cfg is None else self.layer_cfg[4],
                                  n5x5red2=filters[1][2] if self.layer_cfg is None else self.layer_cfg[5],
                                  n5x5=filters[1][2],
                                  pool_planes=filters[1][3],
                                  tmp_name='b3')

        self.maxpool1_2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.maxpool2_2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception_a4_2 = block(in_planes=sum(filters[1]),
                                  n1x1=filters[2][0],
                                  n3x3red=96 if self.layer_cfg is None else self.layer_cfg[6],
                                  n3x3=filters[2][1],
                                  n5x5red1=16 if self.layer_cfg is None else self.layer_cfg[7],
                                  n5x5red2=filters[2][2] if self.layer_cfg is None else self.layer_cfg[8],
                                  n5x5=filters[2][2],
                                  pool_planes=filters[2][3],
                                  tmp_name='a4')
        self.inception_b4_2 = block(in_planes=sum(filters[2]),
                                  n1x1=filters[3][0],
                                  n3x3red=112 if self.layer_cfg is None else self.layer_cfg[9],
                                  n3x3=filters[3][1],
                                  n5x5red1=24 if self.layer_cfg is None else self.layer_cfg[10],
                                  n5x5red2=filters[3][2] if self.layer_cfg is None else self.layer_cfg[11],
                                  n5x5=filters[3][2],
                                  pool_planes=filters[3][3],
                                  tmp_name='b4')
        self.inception_c4_2 = block(in_planes=sum(filters[3]),
                                  n1x1=filters[4][0],
                                  n3x3red=128 if self.layer_cfg is None else self.layer_cfg[12],
                                  n3x3=filters[4][1],
                                  n5x5red1=24 if self.layer_cfg is None else self.layer_cfg[13],
                                  n5x5red2=filters[4][2] if self.layer_cfg is None else self.layer_cfg[14],
                                  n5x5=filters[4][2],
                                  pool_planes=filters[4][3],
                                  tmp_name='c4')
        self.inception_d4_2 = block(in_planes=sum(filters[4]),
                                  n1x1=filters[5][0],
                                  n3x3red=144 if self.layer_cfg is None else self.layer_cfg[15],
                                  n3x3=filters[5][1],
                                  n5x5red1=32 if self.layer_cfg is None else self.layer_cfg[16],
                                  n5x5red2=filters[5][2] if self.layer_cfg is None else self.layer_cfg[17],
                                  n5x5=filters[5][2],
                                  pool_planes=filters[5][3],
                                  tmp_name='d4')
        self.inception_e4_2 = block(in_planes=sum(filters[5]),
                                  n1x1=filters[6][0],
                                  n3x3red=160 if self.layer_cfg is None else self.layer_cfg[18],
                                  n3x3=filters[6][1],
                                  n5x5red1=32 if self.layer_cfg is None else self.layer_cfg[19],
                                  n5x5red2=filters[6][2] if self.layer_cfg is None else self.layer_cfg[20],
                                  n5x5=filters[6][2],
                                  pool_planes=filters[6][3],
                                  tmp_name='e4')

        self.inception_a5_2 = block(in_planes=sum(filters[6]),
                                  n1x1=filters[7][0],
                                  n3x3red=160 if self.layer_cfg is None else self.layer_cfg[21],
                                  n3x3=filters[7][1],
                                  n5x5red1=32 if self.layer_cfg is None else self.layer_cfg[22],
                                  n5x5red2=filters[7][2] if self.layer_cfg is None else self.layer_cfg[23],
                                  n5x5=filters[7][2],
                                  pool_planes=filters[7][3],
                                  tmp_name='a5')
        self.inception_b5_2 = block(in_planes=sum(filters[7]),
                                  n1x1=filters[8][0],
                                  n3x3red=192 if self.layer_cfg is None else self.layer_cfg[24],
                                  n3x3=filters[8][1],
                                  n5x5red1=48 if self.layer_cfg is None else self.layer_cfg[25],
                                  n5x5red2=filters[8][2] if self.layer_cfg is None else self.layer_cfg[26],
                                  n5x5=filters[8][2],
                                  pool_planes=filters[8][3],
                                  tmp_name='b5')





        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.linear2 = nn.Linear(sum(filters[-1]), num_classes)

        blocksSize = [(224,112), (112, 56), (56,56)]
        channels = [480,832,1024]
        # strides = [1, 2, 2, 2]

        self.net1Blocks = nn.ModuleList()
        self.net1CrossNet = nn.ModuleList()
        self.net2Blocks = nn.ModuleList()
        self.net2CrossNet = nn.ModuleList()


        for stage in range(3):
            # bkplanes = self.inplanes
            if stage == 0:
                self.net1Blocks.append(nn.Sequential(*[self.inception_a3,self.inception_b3,self.maxpool1]))
                # self.inplanes = bkplanes
                self.net2Blocks.append(nn.Sequential(*[self.inception_a3_2,self.inception_b3_2,self.maxpool1_2]))
            elif stage ==1:
                self.net1Blocks.append(nn.Sequential(*[self.inception_a4,self.inception_b4,self.inception_c4,self.inception_d4,self.inception_e4,self.maxpool2]))
                # self.inplanes = bkplanes
                self.net2Blocks.append(nn.Sequential(*[self.inception_a4_2,self.inception_b4_2,self.inception_c4_2,self.inception_d4_2,self.inception_e4_2,self.maxpool2_2]))
            elif stage ==2:
                self.net1Blocks.append(nn.Sequential(*[self.inception_a5,self.inception_b5]))
                # self.inplanes = bkplanes
                self.net2Blocks.append(nn.Sequential(*[self.inception_a5_2,self.inception_b5_2]))

            stageCrossNet1 = nn.ModuleList()
            stageCrossNet2 = nn.ModuleList()
            for to in range(stage + 1, 3):
                stageCrossNet1.append(self._make_fusion_layer(channels[stage], channels[to], blocksSize[stage][1],
                                                              int(blocksSize[stage][1] / blocksSize[to][1])))
                stageCrossNet2.append(self._make_fusion_layer(channels[stage], channels[to], blocksSize[stage][1],
                                                              int(blocksSize[stage][1] / blocksSize[to][1])))
            self.net1CrossNet.append(stageCrossNet1)
            self.net2CrossNet.append(stageCrossNet2)

        self.reset_parameters()
        self.register_hook()

    def register_hook(self):

        self.extract_layers = ['inception_b3', 'inception_e4', 'inception_b5','inception_b3_2', 'inception_e4_2', 'inception_b5_2']

        def get_activation(maps, name):
            def get_output_hook(module, input, output):
                maps[name] = output

            return get_output_hook

        def add_hook(model, maps, extract_layers):
            for name, module in model.named_modules():
                if name in extract_layers:
                    module.register_forward_hook(get_activation(maps, name))

        add_hook(self, self.total_feature_maps, self.extract_layers)

    def reset_parameters(self):

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_fusion_layer(self, in_planes, out_planes, in_size, minification):
        layers = []
        layers.append(nn.Conv2d(in_planes, out_planes, minification, minification, padding=0, bias=False))
        # layers.append(nn.AvgPool2d(minification, minification))
        layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x1 = self.pre_layers(x)
        x2 = self.pre_layers2(x)


        net1Knowledge = []
        net2Knowledge = []
        for stage in range(3):
            x1 = self.net1Blocks[stage](x1)

            x2 = self.net2Blocks[stage](x2)

            crossFusionKnowledge = []
            temp1 = x1
            temp2 = x2
            for preNum in range(0, stage):
                temp1 = temp1 + net1Knowledge[preNum][stage - preNum - 1]
                temp2 = temp2 + net2Knowledge[preNum][stage - preNum - 1]
                # print('-----------', temp1.size())
            crossFusionKnowledge.append((torch.flatten(temp1, 1), torch.flatten(temp2, 1)))

            stageKnowledge1 = []
            stageKnowledge2 = []
            for to in range(stage + 1, 3):
                stageKnowledge1.append(self.net1CrossNet[stage][to - stage - 1](x1))
                stageKnowledge2.append(self.net2CrossNet[stage][to - stage - 1](x2))

            net1Knowledge.append(stageKnowledge1)
            # print('over1')
            net2Knowledge.append(stageKnowledge2)
            # print('over2')
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

        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.linear(x1)

        x2 = self.avgpool2(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.linear2(x2)
        return x1, x2, crossFusionKnowledge, fmap

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, List, List]:
        return self._forward_impl(x)

