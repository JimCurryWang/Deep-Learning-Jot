import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.distributed as dist
import torch.utils.data as data_utils


class Net(nn.Module):
    '''Simple demo for multi structure
    '''
    def __init__(self, n_input, n_hidden, n_output):

        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)

        self.predict1 = nn.Linear(n_hidden*2, n_output)
        self.predict2 = nn.Linear(n_hidden*2, n_output)

    def forward(self, input1, input2):
        '''Multiple input
        '''
        out01 = self.hidden1(input1)
        out02 = torch.relu(out01)
        out03 = self.hidden2(out02)
        out04 = torch.sigmoid(out03)

        out11 = self.hidden1(input2)
        out12 = torch.relu(out11)
        out13 = self.hidden2(out12)
        out14 = torch.sigmoid(out13)

        # Concatenate the result if needed
        out = torch.cat((out04, out14), dim=1) 
 
        out1 = self.predict1(out)
        out2 = self.predict2(out)

        # Multiple output
        return out1, out2 

class DoubleNet(nn.Module):
    '''Same as Net but write in different way (Sequential)
    '''
    def __init__(self, n_input, n_hidden, n_output):

        super(DoubleNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid()
            )

        self.block2 = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid()
            )

        self.predict1 = nn.Linear(n_hidden*2, n_output)
        self.predict2 = nn.Linear(n_hidden*2, n_output)

    def forward(self, input1, input2):
        '''Multiple input

        input1 -> [100, 1]
        out1 -> [100, 20]
        out -> [100, 40]
        '''
        out1 = self.block1(input1)
        out2 = self.block2(input2)

        # Concatenate the result if needed
        out = torch.cat((out1, out2), dim=1) 
 
        pre1 = self.predict1(out)
        pre2 = self.predict2(out)

        # Multiple output
        return pre1, pre2 

def train(net, data):
    x1,y1,x2,y2 = data

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_func = torch.nn.MSELoss()

    for t in range(1000):
        prediction1, prediction2 = net(x1, x2)

        # Computing the loss seperately but need to sum up in the end
        # !Important
        loss1 = loss_func(prediction1, y1)
        loss2 = loss_func(prediction2, y2)
        loss = loss1 + loss2 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 100 == 0:
           print('Loss1 = %.4f' % loss1.data,'Loss2 = %.4f' % loss2.data,)

def SimpleDataLoader():
    '''Generating Fake Data (Testing data)
    '''
    x1 = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) 
    y1 = x1.pow(3)+0.1*torch.randn(x1.size())

    x2 = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y2 = x2.pow(3)+0.1*torch.randn(x2.size())

    x1, y1 = Variable(x1), Variable(y1)
    x2, y2 = Variable(x2), Variable(y2)

    return (x1,y1,x2,y2)


if __name__ == '__main__':
    data = SimpleDataLoader()
    
    # Net
    net = Net(n_input=1, n_hidden=20, n_output=1)
    train(net, data)

    # Sequential version Net
    doublenet = DoubleNet(n_input=1, n_hidden=20, n_output=1)
    train(doublenet, data)




