from ResNet import Block
from ResNet import ResNet_test
from ResNet import ResNet50, ResNet101, ResNet152

import torch
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms 

# Check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# load the data 
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )


#Load train and test set
data = torchvision.datasets.CIFAR10(
    root='../CIFAR10',
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(data,batch_size=128,shuffle=True)
test_loader = torch.utils.data.DataLoader(data,batch_size=128,shuffle=False)

# Optimizer and loss function
model = ResNet101(img_channel=3, num_classes=1000)
optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
loss_function = nn.CrossEntropyLoss()


# training process
epochs = 2
for epoch in range(epochs):
    closs = 0
    
    for i,batch in enumerate(train_loader):
        
        inputs, output = batch
        inputs = inputs.to(device)
        output = output.to(device)
        
        # Forward
        prediction = model(inputs)

        # Backward
        optimizer.zero_grad()
        loss = loss_function(prediction, output)
        closs = loss.item()
        loss.backward()
        optimizer.step()
        
        # Show progress for every 100th times
        if i%100 == 0:
            print('[{}/{}] Loss: {}'.format(epoch+1,epochs,closs/100))
            closs = 0
            
            
    correctHits=0
    total=0     
    for i,batch in enumerate(test_loader):
        inputs, output = batch
        inputs = inputs.to(device)
        output = output.to(device)
        
        # Forward
        prediction = model(inputs)
        # returns max as well as its index
        _,prediction = torch.max(prediction.data,1)  
        total += output.size(0)
        correctHits += (prediction==output).sum().item()
    print('Accuracy on epoch ',epoch+1,'= ',str((correctHits/total)*100))