'''
VGG16 for CIFAR-10/100 Dataset.
Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['cross_vgg16', 'cross_vgg19']

#cfg = {
#    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#}

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

        # print("------------------------------"+ str(x.size())+"," + str(y.size())+ ","+ str(input.size()))


        x = F.relu(self.bn1((self.conv1(input))))
        # print("----------------1--------------"+ str(x.size()))
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        # print("----------------2--------------"+ str(x.size()))

        atmap.append(x)
        # print("----------------3--------------"+ str(x.size()))           
        x = F.avg_pool2d(x, self.sptial)
        # print("----------------4--------------"+ str(x.size()))        
        x = x.view(x.size(0), -1)

        out = self.fc2(x)
        atmap.append(out)

        return out


class CrossVGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, dropout = 0.0, KD= False):
        super(CrossVGG, self).__init__()
        self.KD = KD
        self.inplances = 64
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
                
        if depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4
        
        blocksSize = [(16, 8), (8, 4), (4, 2), (2, 1)]
        channels = [128, 256, 512, 512]
        self.net1Blocks = nn.ModuleList()
        self.net1CrossNet = nn.ModuleList()
        self.net2Blocks = nn.ModuleList()
        self.net2CrossNet = nn.ModuleList()
        for stage in range(4):
            bkplances = self.inplances
            self.net1Blocks.append(self._make_layers(channels[stage], num_layer))
            self.inplances = bkplances
            self.net2Blocks.append(self._make_layers(channels[stage], num_layer))
            stageCrossNet1 = nn.ModuleList()
            stageCrossNet2 = nn.ModuleList()
            for to in range(stage+1, 4):
                stageCrossNet1.append(self._make_fusion_layer(channels[stage], channels[to], blocksSize[stage][1], int(blocksSize[stage][1]/blocksSize[to][1])))
                stageCrossNet2.append(self._make_fusion_layer(channels[stage], channels[to], blocksSize[stage][1], int(blocksSize[stage][1]/blocksSize[to][1])))
            self.net1CrossNet.append(stageCrossNet1)
            self.net2CrossNet.append(stageCrossNet2)      
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(512, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _make_layers(self, input, num_layer):    
        layers=[]
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplances, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplances = input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def _make_fusion_layer(self, in_planes, out_planes, in_size, minification):
        layers = []
        layers.append(nn.Conv2d(in_planes, out_planes, minification, minification, padding=0, bias=False))
        # layers.append(nn.AvgPool2d(minification, minification))
        layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x2 = self.conv1(x)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        

        fmap = []
        crossFusionKnowledge = []
        net1Knowledge = []
        net2Knowledge = []
        for stage in range(4):
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
            for to in range(stage+1, 4):
                stageKnowledge1.append(self.net1CrossNet[stage][to-stage-1](x1))
                stageKnowledge2.append(self.net2CrossNet[stage][to-stage-1](x2))
            net1Knowledge.append(stageKnowledge1)
            net2Knowledge.append(stageKnowledge2)
        fmap.append(x1)
        fmap.append(x2)

        x1 = x1.view(x1.size(0), -1)
        x1 = self.classifier(x1)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.classifier(x2)
        
        return x1, x2, crossFusionKnowledge, fmap
    
def cross_vgg16(pretrained=False, path=None, **kwargs):
    """
    Constructs a CrossVGG16 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = CrossVGG(depth=16, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
    
def cross_vgg19(pretrained=False, path=None, **kwargs):
    """
    Constructs a CrossVGG19 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = CrossVGG(depth=19, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model