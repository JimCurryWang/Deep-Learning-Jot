import torch
import torch.nn as nn


class Block(nn.Module):
    '''A "bottleneck" building block for ResNet-50/101/152
    '''
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        
        # ResNet use 4 times expansion in the entire design
        self.expansion = 4
        
        self.identity_downsample = identity_downsample
        self.stride = stride
        
        # 1x1 Conv 
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, 
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        
        # 3x3 Conv (bottleneck)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        
        # 1x1 Conv
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # When the start point and end point dimension is different...
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    '''
        torch.nn.Conv2d(in_channels, out_channels, 
                    kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
    
        torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
        torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        torch.nn.AdaptiveAvgPool2d(output_size = (n,m))
        torch.nn.Linear(in_features, out_features, bias=True)
    '''
    def __init__(self, Block, layers, image_channels=3, num_classes=1000):
        super(ResNet, self).__init__()
        
        # Conv1 (7,7,64), s=2, p=3
        self.in_channels = 64
        self.Conv1 = nn.Sequential(
            nn.Conv2d(image_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) 
        
        # 3x3 Max Pooling, s=2, p=1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers (Residual block layers staking)
        self.layer1 = self._make_layer(Block, layers[0], intermediate_channels=64,  stride=1)
        self.layer2 = self._make_layer(Block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(Block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(Block, layers[3], intermediate_channels=512, stride=2)

        # Adaptive Average Pooling (1x1 outputsize)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # FC (2048 -> 1000)
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        
        x = self.Conv1(x)        
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        # AVG Pool & FC
        # count how many batch and reshape it to one-dimension
        # i.g. (batch_size,2048,1,1) -> (batch_size,2048)
        x = self.avgpool(x) 
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, Block, num_residual_blocks, intermediate_channels, stride):
        '''Creating the numbers of blocks stacked
        i.g.
            ResNet-50:  [3,4,6,3]
            ResNet-101: [3,4,23,3]
            ResNet-152: [3,8,36,3]
        '''
        identity_downsample = None
        layers = []

        
        # Adjust the channel size for the first block in each layer (Do identiry downsample)
        # we need to adapt the Identity (skip connection) 
        # so it can be able to be added to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, intermediate_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            Block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # Update in_channels size for the following block
        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4 

        # For example for first ResNet layer, sencond ~ last block: 
        # 256 will be mapped to 64 as intermediate layer, then finally back to 256(64*4),
        # Hence no identity downsample is needed
        #                     (256,56,56)
        # -> conv(1x1,64)  -> (64,56,56)
        # -> conv(3x3,64)  -> (64,56,56)
        # -> conv(1x1,256) -> (256,56,56)

        for i in range(num_residual_blocks - 1):
            layers.append(Block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet_test(img_channel=3, num_classes=1000):
    '''fake structure just for test
    '''
    return ResNet(Block, [2, 1, 1, 1], img_channel, num_classes)

def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(Block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(Block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(Block, [3, 8, 36, 3], img_channel, num_classes)