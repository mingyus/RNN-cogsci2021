import torch
from torch import nn
from info_task_exp import *


class LSTM_embed_Model(nn.Module):
    def __init__(self, input_type, input_size, input_index, embed_size, output_size, hidden_size, num_layers, device):
        super(LSTM_embed_Model, self).__init__()

        # Defining some parameters
        self.input_type = input_type
        self.input_index = input_index
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # Defining the layers
        # input to embedding
        if embed_size is None: # no embedding layer
            embed_layer_size = input_size
        else:
            embed_layer_size = 0
            embedlayer_key_list, embedlayer_list = [], []
            for embed_key, embed_size_value in embed_size.items():
                if embed_key == 'stimulus_dim':
                    if isinstance(embed_size_value, list): # seperate embeddings for different dimensions
                        for iDim in range(numDimensions):
                            embedlayer_key_list.append(embed_key + str(iDim))
                            embedlayer_list.append(nn.Linear(in_features=int(len(input_index['stimulus'])/numDimensions), out_features=embed_size_value[iDim]))
                            embed_layer_size += embed_size_value[iDim]
                    else: # the same embedding layer for all dimensions
                        embedlayer_key_list.append(embed_key)
                        embedlayer_list.append(nn.Linear(in_features=int(len(input_index['stimulus'])/numDimensions), out_features=embed_size_value))
                        embed_layer_size += embed_size_value * numDimensions
                else:
                    embedlayer_key_list.append(embed_key)
                    embedlayer_list.append(nn.Linear(in_features=len(input_index[embed_key]), out_features=embed_size_value))
                    embed_layer_size += embed_size_value
            self.embed = nn.ModuleDict(dict(zip(embedlayer_key_list, embedlayer_list)))
            for input_key in input_index.keys():
                if all([input_key not in embed_key for embed_key in embed_size.keys()]):
                    embed_layer_size += len(input_index[input_key])
        # LSTM layer
        self.lstm = nn.LSTM(embed_layer_size, hidden_size, num_layers, batch_first=True)
        # fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    
    def forward(self, x):
        
        batch_size = x.shape[0]

        # Initializing hidden state for first input using method defined below
        (hidden, cell) = self.init_hidden_cell(batch_size)
        
        # Input to embedding
        if self.embed_size is not None:
            x_embed = torch.tensor([]).to(self.device)
            for input_key, input_index_value in self.input_index.items():
                x_this = x[:, :, input_index_value]
                if any([input_key in embed_key for embed_key in self.embed_size.keys()]):
                    embed_key = input_key
                    if (input_key == 'stimulus') and ('stimulus_dim' in self.embed_size.keys()):
                        x_this = torch.cat([self.embed[embed_key + (str(iDim) if isinstance(embed_size['stimulus_dim'], list) else '')](x_this[iDim*numFeaturesPerDimension:(iDim+1)*numFeaturesPerDimension]) for iDim in range(numDimensions)], dim=2)
                    else:
                        x_this = self.embed[embed_key](x_this)
                x_embed = torch.cat((x_embed, x_this), dim=2)
            x = x_embed

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