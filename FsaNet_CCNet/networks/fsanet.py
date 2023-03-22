#use normal bn layer
import torch.nn as nn
import torch

from dctnl_attention.Linsoftmax import DCTNLAttention11
from dctnl_attention.Dotproduct import DCTNLAttention21
from utils.pyt_utils import load_model

affine_par = True
from Synchronized.sync_batchnorm import SynchronizedBatchNorm2d as SyncBN
BatchNorm2d = SyncBN #functools.partial(InPlaceABNSync, activation='identity')

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, multi_grid=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        #self.bn1 = nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        # self.bn2 = nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu_inplace(out)

        return out

class RNLModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, att_mode, k):
        super(RNLModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
        self.fsa = eval(att_mode)(inter_channels,k)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels),nn.ReLU(inplace=False))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(out_channels),nn.ReLU(inplace=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.fsa(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, criterion, recurrence, train_mode, att_mode, k):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),nn.ReLU(inplace=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        
        if train_mode == 'back_only':
            for p in self.parameters():
                p.requires_grad = False     
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1,1,1))
        self.head = RNLModule(2048, 512, num_classes, att_mode, k)

        self.criterion = criterion
        self.recurrence = recurrence
        
        #zfy, only optimize head
        if train_mode == 'att_only':
            for p in self.parameters():
                p.requires_grad = False              
            for p in self.head.parameters():
                p.requires_grad = True   

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x, labels=None):
        #print(111)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_dsn = self.dsn(x)
        x = self.layer4(x)
        x = self.head(x, self.recurrence)
        outs = [x, x_dsn]
        #print(outs)

        if self.criterion is not None and labels is not None:
            return self.criterion(outs, labels)
        else:
            return outs


def Seg_Model(num_classes, criterion=None, pretrained_model=None, recurrence=0, train_mode=None, att_mode='DCTNLAttention13', k=[0,5,0,5], **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, criterion, recurrence, train_mode, att_mode, k)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)

    return model
