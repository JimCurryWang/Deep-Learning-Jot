import torch
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from torch import optim  
from torch import nn  
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
# ---- embedding part ---
vocab_size = 100
embedding_dim = 50 

# --- LSTM part ----
sequence_length = 28
# input_size = 28
hidden_size = 256
num_layers = 2
num_classes = 10

learning_rate = 0.005
batch_size = 64
num_epochs = 3


class EmbeddingLSTM(nn.Module):
    '''Embedding vector with LSTM (many-to-one)
    '''
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, pretrained_weights=None):
        super(EmbeddingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pretrained_weights = pretrained_weights

        if self.pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(self.pretrained_weights) 
            self.embedding.weight.requires_grad = False
        else:
            # Embedding(vocab_size=100, embedding_dim=50)
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=embedding_dim
            )    
            self.embedding.weight.requires_grad = True

        
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True
        )

        # Using the last rnn output with fc to obtain the final classificaiton result
        self.fc = nn.Linear(
            in_features=hidden_size, out_features=num_classes
        )

    def forward(self, x):
        '''
        '''
        embed = self.embedding(x) # in=[64, 28],  out=[64, 28, 50]

        out, _ = self.lstm(embed) # x=[64, 28, 50], out=[64, 28, 256]=(batch, seq_len, 1 * hidden_size)

        # Decode the hidden state of the last time step
        # only take the last hidden state and send it into fc
        out = out[:, -1, :] # out=[64, 256]
        out = self.fc(out) # out=[64, 10]

        return out

if __name__ == '__main__':

    # Initialize network
    
    pre = False
    if pre:
        # torch.FloatTensor()
        # "FloatTensor" containing pretrained weights
        pretrained_weights = torch.rand(size=(100,50)) 
        model = EmbeddingLSTM(
            vocab_size, embedding_dim, hidden_size, num_layers, num_classes, 
            pretrained_weights=pretrained_weights
        ).to(device)
    else:
        model = EmbeddingLSTM(
            vocab_size, embedding_dim, hidden_size, num_layers, num_classes, 
            pretrained_weights=None
        ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Generate test data 
    FakeData = torch.randint(low=0, high=10, size=[batch_size, sequence_length+1]) # [64, 29]
    data = FakeData[:,1:] # [64, 28]
    targets =  FakeData[:,0] # [64], torch.LongTensor([0,1,2,4...7,8,9...]])

    # Send data to device
    data = data.to(device=device)
    targets = targets.to(device=device)

    # Forward
    scores = model(data) # [64, 10]
    loss = criterion(scores, targets)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Gradient descent update step/adam step
    optimizer.step()

