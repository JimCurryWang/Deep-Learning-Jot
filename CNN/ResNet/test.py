import torch
from ResNet import Block
from ResNet import ResNet_test
from ResNet import ResNet50, ResNet101, ResNet152

# test for residual Block 
block_src = Block(in_channels=256, intermediate_channels=64)
print(block_src)

# test for mock ResNet
net = ResNet_test(img_channel=3, num_classes=1000)
print(net)


# test for ResNet-101
def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net = ResNet101(img_channel=3, num_classes=1000)
    y = net(torch.randn(4, 3, 224, 224)).to(device)
    print(y.size())
    
test()