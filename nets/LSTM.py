import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, device):
        super(LSTMModel, self).__init__()

        # Defining some parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # Defining the layers
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)   
        # fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        (hidden, cell) = self.init_hidden_cell(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden_cell(self, batch_size):
        # Initialize the first hidden state as zeros
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return hidden, cell