import pickle
import torch
from torch import nn

from funcs_data_preprocessing import *
from info_model import *
from nets.LSTM import *
from nets.LSTM_embed import *


def get_device():
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
        print('GPU is available', flush=True)
    else:
        device = torch.device('cpu')
        print('GPU not available, CPU used', flush=True)
    return device


def get_dataInfo(modelName):
    dataInfo = dict()
    dataInfo['input_type'] = input_type[modelName]
    dataInfo['output_type'] = output_type[modelName]
    if 'iWorkerOnehot' in dataInfo['input_type']:
        _, workerIds = import_data(getWorkerIds=True)
    dataInfo['input_size'] = 28 + (len(workerIds) if 'iWorkerOnehot' in dataInfo['input_type'] else 0)
    dataInfo['output_size'] = 64
    dataInfo['num_output'] = 1
    return dataInfo


def get_model_filename(modelName, hyper_parameters, epoch=None):
    model_filename = modelName + '_' + '_'.join([k + str(v) for (k,v) in hyper_parameters.items()])
    if epoch is not None:
        model_filename = model_filename + '_epoch' + str(epoch) + '.p'
    return model_filename


def initialize_model(modelName, dataInfo, hyper_parameters):
    # Hyper-parameters
    hidden_size, n_epochs, lr_init, lr_adjust_type, batch_size = hyper_parameters['hidden_size'], hyper_parameters['n_epochs'], hyper_parameters['lr_init'], hyper_parameters['lr_adjust_type'], hyper_parameters['batch_size']
    # for embedding models
    if 'embed' in modelName:
        embed_size = hyper_parameters['embed_size']
        input_index = dict()
        input_index['gameStart'] = torch.arange(0,1)
        input_index['gameType'] = torch.arange(1,5)
        input_index['stimulus'] = torch.arange(5,26)
        input_index['reward'] = torch.arange(26,28)
        if 'iWorkerOnehot' in dataInfo['input_type']:
            _, workerIds = import_data(getWorkerIds=True)
            input_index['subjID'] = torch.arange(28,28+len(workerIds))
        
    # Instantiate the model with hyperparameters
    device = get_device()
    if modelName == 'LSTM':
        model = LSTMModel(input_size=dataInfo['input_size'], output_size=dataInfo['output_size'], hidden_size=hidden_size, num_layers=1, device=device)
    elif modelName == 'LSTM_embed':
        model = LSTM_embed_Model(input_type=dataInfo['input_type'], input_size=dataInfo['input_size'], input_index=input_index, embed_size=embed_size, output_size=dataInfo['output_size'], hidden_size=hidden_size, num_layers=1, device=device)
        
    return model