'''ResNetROI in PyTorch.

BasicBlock and Bottleneck module is from the original ResNetROI paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import math
import pdb

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNetROI(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetROI, self).__init__()
        self.in_planes = 64

        #ToDo: test
        self.expansion = block.expansion

        #self.conv1 = conv3x3(3,64)
        self.conv1 = conv3x3(1,64)  #For gray scale images
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.linear = nn.Linear(512*block.expansion, num_classes)

        self.fc1 = nn.Linear(512*block.expansion*6*12, 2048)
        self.bn6 = nn.BatchNorm1d(2048)
        
        self.fc2 = nn.Linear(2048, 2048)
        self.bn7 = nn.BatchNorm1d(2048)
        
        self.fc3 = nn.Linear(2048,num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, roi):
        #pdb.set_trace()
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        
        out = self.roi_pooling(out, roi, size=(6,12), spatial_scale=1.0/8)

        out = out.view(-1,512*6*12)
        out = F.relu(self.bn6(self.fc1(out)))
        
        outFeat = self.bn7(self.fc2(out))
        out = F.relu(outFeat)
        
        out = self.fc3(out)

        return out, outFeat

    
    #added for finetuning
    def resetLastLayer(self, num_classes):
        self.fc3 = nn.Linear(2048, num_classes)      
        n = self.fc3.in_features * self.fc3.out_features
        self.fc3.weight.data.normal_(0, math.sqrt(2. / n))
    
    def appendLastLayer(self, num_classes):
        tempWeights = self.fc3.weight
        self.fc3 = nn.Linear(2048, num_classes)      
        n = self.fc3.in_features * self.fc3.out_features
        self.fc3.weight.data.normal_(0, math.sqrt(2. / n))
        
        #Copying previous network weights
        self.fc3.weight.data[:tempWeights.size()[0],:]=tempWeights.data

    def roi_pooling(self, input, rois, size=(7,7), spatial_scale=1.0):

        assert(rois.dim() == 2)
        assert(rois.size(1) == 5)
        output = []
        rois = rois.data.float()
        num_rois = rois.size(0)

        rois[:,1:].mul_(spatial_scale)
        rois = rois.long()
        for i in range(num_rois):
            roi = rois[i]
            im_idx = roi[0]
            im = input.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
            output.append(F.adaptive_max_pool2d(im, size))
        
        return torch.cat(output, 0)


def ResNetROI18(num_classes=10):
    return ResNetROI(PreActBlock, [2,2,2,2], num_classes)

def ResNetROI34(num_classes=10):
    return ResNetROI(BasicBlock, [3,4,6,3], num_classes)

def ResNetROI50():
    return ResNetROI(Bottleneck, [3,4,6,3])

def ResNetROI101():
    return ResNetROI(Bottleneck, [3,4,23,3])

def ResNetROI152():
    return ResNetROI(Bottleneck, [3,8,36,3])


def test():
    net = ResNetROI18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
