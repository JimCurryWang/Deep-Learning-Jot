import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
            ::Using "same" convolution
            ::Add BatchNorm2d which is not used in original papers
                (the BatchNorm2d concept is launched after Unet...)
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

class UNet(nn.Module):
    '''U-Net architecture usually uses 3 channel input size for RGB image processing 
    and 1 channel output size

        ::in_channels=3
            image input with 3 channel(R-G-B)
        ::out_channels=1
            Output would be binary, so set out_channels= 1 as default
        ::features=[64, 128, 256, 512]
            the number of feature map in each block

        ::torch.cat((x1, x2), dim=1)
            https://pytorch.org/docs/stable/generated/torch.cat.html
            
        ::TORCHVISION.TRANSFORMS.FUNCTIONAL.resize(img, size, interpolation)
            https://pytorch.org/vision/master/_modules/torchvision/transforms/functional.html
    '''
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
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
        # --- downsampling ---
        # forward through the downsample part and save each output for skip_connections
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # --- bottleneck ---
        x = self.bottleneck(x)

        # reversed the skip_connections list, x[::-1] == list(reversed(x))
        skip_connections = skip_connections[::-1]

        # --- upsampling ---
        # for i in range(0,8,2): -> [0,1,2,3,4,5,6,7] -> 0,2,4,6
        for idx in range(0, len(self.ups), 2):

            # --- ConvTranspose2d ---
            # fetch ConvTranspose2d layers
            x = self.ups[idx](x)
            # do floot division to get the corresponding skip_connection 
            # 0,2,4,6 -> 0,1,2,3
            skip_connection = skip_connections[idx//2]

            # In the original paper, the authors use cropping to solve the size issues
            # But resize, add padding, or consider the input be even is all suitable in here
            if x.shape != skip_connection.shape:
                # resize by transpolation
                # torch.Size([3, 512, 20, 20]) -> torch.Size([20, 20])
                x = TF.resize(x, size=skip_connection.shape[2:])


            # --- Concatenate --- 
            # channel-wise dimension feature concat
            # dim=1 -> along the channel dimension, not on dim=0 which will increase dimension
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # print(skip_connection.shape)
            # print(x.shape)
            # print(concat_skip.shape)



            # --- DoubleConv --- 
            # throw the concat layer into DoubleConv
            x = self.ups[idx+1](concat_skip)

        # --- final output part ---
        x = self.final_conv(x)
        
        return x


def unittest():
    '''
        batch_size = 3 
        channel_size = 1 
        kernel_size = 572x572
    '''
    x = torch.randn((1, 3, 572, 572))
    model = UNet(in_channels=3, out_channels=1)
    # print(model.ups)
    preds = model(x)
    # print(preds.shape)
    
    assert preds.shape[2:] == x.shape[2:] , (preds.shape, x.shape) 

if __name__ == "__main__":
    unittest()

