import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
            ::Using same convolution
            ::Add BatchNorm2d which is not used in original papers
        '''
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels, out_channels,
                kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            # Conv2 
            nn.Conv2d(out_channels, out_channels,
                kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    '''
        ::in_channels=3
            image input with 3 channel(R-G-B)
        ::out_channels=1
            Output would be binary, so set out_channels= 1 as default
        ::features=[64, 128, 256, 512]
            the number of feature map in each block
    '''
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part (2*conv)
        for feature in features:
            self.downs.append(
                DoubleConv(in_channels=in_channels, out_channels=feature)
            )

            # update the next in_channels size
            in_channels = feature

        # Up part (Up + 2*conv)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=feature*2, out_channels=feature, 
                    kernel_size=2, stride=2,
                )
            )
            self.ups.append(
                DoubleConv(in_channels=feature*2, out_channels=feature)
            )

        # Bottleneck
        # Refer the last features size as bottleneck in and out
        # (i.g. 512->1024)
        self.bottleneck = DoubleConv(in_channels=features[-1], out_channels=features[-1]*2)
        
        # Final output part
        self.final_conv = nn.Conv2d(
            in_channels=features[0], out_channels=out_channels, 
            kernel_size=1
        )

    def forward(self, x):
        # forward through the downsample part and save each output for skip_connections
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # reversed the skip_connections list, list(reversed(skip_connections))
        skip_connections = skip_connections[::-1]

        # for i in range(0,8,2): -> [0,1,2,3,4,5,6,7] -> 0,2,4,6
        for idx in range(0, len(self.ups), 2):

            # fetch ConvTranspose2d layers
            x = self.ups[idx](x)
            # Do floot division to get the corresponding skip_connection 
            # 0,2,4,6 -> 0,1,2,3
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # channel-wise dimension feature concat
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape , (preds.shape, x.shape) 

if __name__ == "__main__":
    test()

    # double_conv = DoubleConv(in_channels=512, out_channels=338)
    unet = UNET()
    # print(unet.downs)





