import torch
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from torch import optim  
from torch import nn  
from torch.utils.data import DataLoader
from tqdm import tqdm

"""
Example code of a simple RNN, GRU, LSTM on the MNIST dataset.

https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

    input_size – The number of expected features in the input x
    
    hidden_size – The number of features in the hidden state h
    
    num_layers – Number of recurrent layers. 
                E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, 
                with the second RNN taking in outputs of the first RNN and computing the final results. 
                Default: 1
    
    batch_first – If True, then the input and output tensors are provided as (batch, seq_len, feature). 
                Default: False

                input of shape (seq_len, batch, input_size):
                    -> (batch, seq_len, input_size)
                output of shape (seq_len, batch, num_directions * hidden_size)
                    -> (batch, seq_len, num_directions * hidden_size)

    bidirectional – If True, becomes a bidirectional RNN. 
                Default: False
"""

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
sequence_length = 28
input_size = 28
hidden_size = 256
num_layers = 2
num_classes = 10

learning_rate = 0.005
batch_size = 64
num_epochs = 3


class RNN(nn.Module):
    '''Recurrent neural network (many-to-one)
    '''
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        '''
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, h0 = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        '''
        out, _ = self.rnn(x)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


class GRU(nn.Module):
    '''Recurrent neural network with GRU (many-to-one)
    '''
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        '''
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, h0 = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        '''
        out, _ = self.gru(x)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


class LSTM(nn.Module):
    '''Recurrent neural network with LSTM (many-to-one)
    '''
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Using the last rnn output with fc to obtain the final classificaiton result
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        '''
        # Set initial hidden and cell states, 
        # h0, c0 -> (num_layers * num_direction, batch, hidden_size)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, (h, c) = self.lstm( x, (h0, c0)) 
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)
        '''
        out, _ = self.lstm(x)
        out = out.reshape(out.shape[0], -1) 

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


def check_accuracy(loader, model):
    '''Check accuracy on training & test to see how good our model
    '''
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    # Toggle model back to train
    model.train()
    return num_correct / num_samples


# Load Data
train_dataset = datasets.MNIST(root="mnist/MNIST", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="mnist/MNIST", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # (torch.Size([64, 1, 28, 28]), torch.Size([64]))
        # Get data to cuda if possible
        data = data.to(device=device).squeeze(1) # [64, 1, 28, 28] -> [64, 28, 28]
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent update step/adam step
        optimizer.step()


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")

