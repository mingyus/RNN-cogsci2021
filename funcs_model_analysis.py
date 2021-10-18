import numpy as np
import torch

from info_task_exp import *
from funcs_data_preprocessing import *
from funcs_model import *


def load_winning_model(datasetType, modelName, returnLlh=False):
    hyper_parameters, epoch = winning_model[modelName].values()
    print(hyper_parameters, epoch)
    device = get_device()
    dataInfo = get_dataInfo(modelName)
    fileName = get_model_filename(modelName, hyper_parameters, epoch)
    model = initialize_model(modelName, dataInfo, hyper_parameters)
    model.load_state_dict(torch.load('models/' + 'model_' + fileName, map_location=device))
    model.eval()
    res = pickle.load(open('models/' + 'test_' + fileName, 'rb'))
    if not returnLlh:
        return model
    else:
        return model, res['llh_test'], res['p_allChoices']


def get_test_games(data):
    # add game index to data
    iGame = -1
    for i in range(data.shape[0]):
        if data.loc[i,'trial'] == 1:
            iGame += 1
        data.loc[i,'gameIndex'] = iGame
    # get test game index
    games_index = pickle.load(open('data/split_info.p', 'rb'))
    # get test games
    return pd.concat([data[data['gameIndex']==iGame] for iGame in games_index['test']]).reset_index(drop=True)