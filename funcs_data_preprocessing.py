import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle
import os
from scipy.stats import binom
from collections import defaultdict

from utilities import *
from info_task_exp import *

########## import data ##########
def import_data(acc=0.46, getWorkerIds=False):
    # load data
    data = pd.read_csv('data/data.csv')

    # exclude participants based on accuracy criterion
    numRewardingFeaturesSelected = pd.concat([(~data['rewardingFeature_'+dim].isnull() & \
        (data['selectedFeature_'+dim] == data['rewardingFeature_'+dim])).astype(int) for dim in DIMENSIONS], axis = 1, keys = DIMENSIONS).sum(axis = 1).values
    numRelevantDimensionsUnselected = pd.concat([(~data['rewardingFeature_'+dim].isnull() & data['selectedFeature_'+dim].isnull()).astype(int) for dim in DIMENSIONS], axis = 1, keys = DIMENSIONS).sum(axis = 1).values
    rt = data['rt'].values
    numRelevantDimensions = data['numRelevantDimensions'].values
    p = np.zeros((numDimensions+1,numDimensions+1)) * np.nan
    for n in range(numDimensions+1):
        for k in range(n+1):
            p[n,k] = binom.pmf(k, n, p=1/3)
    data['add_expectedReward'] = [0.4 if np.isnan(rt[iRow]) else np.sum([p[int(numRelevantDimensionsUnselected[iRow]), i] *
                            rewardSetting[int(numRelevantDimensions[iRow]) - 1][numRewardingFeaturesSelected[iRow] + i]
                            for i in range(int(numRelevantDimensionsUnselected[iRow])+1)]) for iRow in range(data.shape[0])]
    if acc is not None:
        avgReward = data.groupby(['workerId']).mean()['add_expectedReward']
        workerIds = avgReward.index[avgReward >= acc].tolist()
        data = data[data['workerId'].isin(workerIds)]
    else:
        workerIds = data['workerId'].unique().tolist()
    data = data.reset_index(drop=True)

    if getWorkerIds:
        return data, workerIds
    else:
        return data


########## train test split ##########
def train_validation_test_split(data, rand_seed):
    
    np.random.seed(rand_seed)
    
    # create game index
    data_copy = data.copy()
    iGame = -1
    for i in range(data.shape[0]):
        if data_copy.loc[i,'trial'] == 1:
            iGame += 1
        data_copy.loc[i,'gameIndex'] = iGame
    
    # get the game indices for each set
    """
    Training set: 102 * 16 games
    Validation set: 102 * 1 game
    Test set: 102 * 1 game
    validation and test sets: each has 102/6=17 games per type, relatively evenly distributed across experiment
    """
    workerIds = data_copy['workerId'].unique()
    games_validation, games_test = [], []

    i = -1
    np.random.shuffle(workerIds)
    for informed in [True, False]:
        for numD in np.arange(numDimensions) + 1:
            i += 1
            for workerId in workerIds[i*17:(i+1)*17]:
                game = np.random.choice(data_copy.loc[(data_copy['informed']==informed) & (data_copy['numRelevantDimensions']==numD) & (data_copy['workerId']==workerId), 'gameIndex'].unique())
                games_validation.append(int(game))

    i = -1
    np.random.shuffle(workerIds)
    for informed in [True, False]:
        for numD in np.arange(numDimensions) + 1:
            i += 1
            for workerId in workerIds[i*17:(i+1)*17]:
                two_games = np.random.choice(data_copy.loc[(data_copy['informed']==informed) & (data_copy['numRelevantDimensions']==numD) & (data_copy['workerId']==workerId), 'gameIndex'].unique(), size=2, replace=False)
                if two_games[0] in games_validation:
                    game = two_games[1]
                else:
                    game = two_games[0]
                games_test.append(int(game))

    games_train = [game for game in range(int(data_copy['gameIndex'].max())+1) if game not in games_validation and game not in games_test]

    # split
    data_split = dict()
    data_split['train'] = pd.concat([data[data_copy['gameIndex'] == iGame] for iGame in games_train], ignore_index=True).reset_index(drop=True)
    data_split['test'] = pd.concat([data[data_copy['gameIndex'] == iGame] for iGame in games_test], ignore_index=True).reset_index(drop=True)
    data_split['validation'] = pd.concat([data[data_copy['gameIndex'] == iGame] for iGame in games_validation], ignore_index=True).reset_index(drop=True)
    
    return data_split, {'train':games_train, 'validation':games_validation, 'test':games_test}


########## functions: data to RNN variables ##########

# all possible values for the variables
aAll = [[a0, a1, a2] for a0 in range(numFeaturesPerDimension+1) for a1 in range(numFeaturesPerDimension+1) for a2 in range(numFeaturesPerDimension+1)]
sAll = [[s0, s1, s2] for s0 in range(numFeaturesPerDimension) for s1 in range(numFeaturesPerDimension) for s2 in range(numFeaturesPerDimension)]
sFullAll = [[sFull0, sFull1, sFull2] for sFull0 in range(numFeaturesPerDimension*2) for sFull1 in range(numFeaturesPerDimension*2) for sFull2 in range(numFeaturesPerDimension*2)]

# real action (w/ feature on each dimension) to a: each dimension coded as 0,1,2,3 (3: no-selection)
def action2a(action):
    actions = [DIMENSIONS_TO_FEATURES[DIMENSIONS[iDim]] + [np.nan] for iDim in range(numDimensions)]
    a = [actions[iDim].index(action[iDim]) for iDim in range(numDimensions)]
    return a

# real stimulus (w/ feature on each dimension) to s: each dimension coded as 0,1,2
def stimulus2s(stimulus):
    s = [DIMENSIONS_TO_FEATURES[DIMENSIONS[iDim]].index(stimulus[iDim]) for iDim in range(numDimensions)]
    return s

# human+computer choices -> sFull: each dimension coded as 0-5
def hc2sFull(human, computer):
    sFull = [[] for _ in range(numDimensions)]
    for iDim in range(numDimensions):
        if not pd.isnull(human[iDim]):
            sFull[iDim] = DIMENSIONS_TO_FEATURES[DIMENSIONS[iDim]].index(human[iDim])
        else:
            sFull[iDim] = DIMENSIONS_TO_FEATURES[DIMENSIONS[iDim]].index(computer[iDim]) + numFeaturesPerDimension
    return sFull

# function to convert data to RNN variables
def data2RNNVariables(data):
    # note: 
    # invalid information is coded as np.nan (including invalid trials, and no information about last trial on the first trial of a game)
    # no-selection (in a valid trial) is coded as 3
    
    # variable initialization
    data = data.copy()
    _, workerIds = import_data(getWorkerIds=True)
    numGames = int(np.sum(data['trial'] == 1))
    iWorker, gameStart, gameType, iaOld, icOld, isOld, isFullOld, rOld, iaNew = [np.zeros((numGames, gameLength)) for _ in range(9)]
    gameStart[:, 0] = 1
    
    iGame = -1
    i = 0
    iLast = None
    
    for iThis in range(data.shape[0]):
        
        if data.loc[iThis, 'trial'] == 1:
            # pad last game (if less than 30 trials) with np.nan
            if (iGame >= 0) & (i < gameLength):
                iaOld[iGame, i:], icOld[iGame, i:], isOld[iGame, i:], isFullOld[iGame, i:], rOld[iGame, i:], iaNew[iGame, i:] = [np.nan] * 6
            ## new game
            i = 0
            iGame += 1
            # which worker
            iWorker[iGame, :] = workerIds.index(data.loc[iThis, 'workerId'])
            # game type
            if data.loc[iThis, 'informed'] == False:
                gameType[iGame, :] = 0
            else:
                gameType[iGame, :] = data.loc[iThis, 'numRelevantDimensions']
        
        if not pd.isnull(data.loc[iThis, 'rt']):

            # stimulus and reward of last trial
            if i == 0:  # the first valid trial of a game
                iaOld[iGame, i], icOld[iGame, i], isOld[iGame, i], isFullOld[iGame, i], rOld[iGame, i] = [np.nan] * 5
            else:
                human = [data.loc[iLast, 'selectedFeature_'+dim] for dim in DIMENSIONS]
                iaOld[iGame, i] = aAll.index(action2a(human))
                computer = [data.loc[iLast, 'randomlySelectedFeature_'+dim] for dim in DIMENSIONS]
                icOld[iGame, i] = aAll.index(action2a(computer))
                stimulus = [data.loc[iLast, 'builtFeature_'+dim] for dim in DIMENSIONS]
                isOld[iGame, i] = sAll.index(stimulus2s(stimulus))
                isFullOld[iGame, i] = sFullAll.index(hc2sFull(human, computer))
                rOld[iGame, i] = data.loc[iLast, 'reward']

            # action of the current trial
            aNew = action2a([data.loc[iThis, 'selectedFeature_'+dim] for dim in DIMENSIONS])
            iaNew[iGame, i] = aAll.index(aNew)

            i += 1
            iLast = iThis

    return iWorker, gameStart, gameType, iaOld, icOld, isOld, isFullOld, rOld, iaNew


########## shuffle matrices for RNN variables ##########
from itertools import permutations
from math import factorial
from copy import deepcopy

allPermsD = list(permutations(range(numDimensions)))
allPermsF = [list(permutations(range(numFeaturesPerDimension))) for _ in range(numDimensions)]
numPermsD = factorial(numDimensions)
numPermsF = factorial(numFeaturesPerDimension)

def get_permutation_info(ifShuffleD=True, ifShuffleF=True):
    if ifShuffleD and ifShuffleF:  # all permutations
        allPerms = [[iD, iF0, iF1, iF2] for iD in range(numPermsD) for iF0 in range(numPermsF) for iF1 in range(numPermsF) for iF2 in range(numPermsF)]
    elif ifShuffleD:
        allPerms = [[iD, 0, 0, 0] for iD in range(numPermsD)]
    elif ifShuffleF:
        allPerms = [[0, iF0, iF1, iF2] for iF0 in range(numPermsF) for iF1 in range(numPermsF) for iF2 in range(numPermsF)]
    numPermsTotal = len(allPerms)
    return allPerms, numPermsTotal

# define S_sFull: (isFull, iPerm) -> isFull_shuffled, used for stimulus_full, with both human and computer choice information
def get_S_sFull(sFullAll, allPerms):
    S_sFull = np.empty((len(sFullAll), len(allPerms))) * np.nan
    for isFull, sFull in enumerate(sFullAll):
        for iPerm in range(len(allPerms)):
            [iD, iF0, iF1, iF2] = allPerms[iPerm]
            sFull_shuffled = deepcopy(sFull)
            # shuffle features
            iFs = [iF0, iF1, iF2]
            for iDim in range(numDimensions):
                f0s = allPermsF[iDim][0]
                fs = allPermsF[iDim][iFs[iDim]]
                ifHuman = np.floor(sFull_shuffled[iDim]/numFeaturesPerDimension) == 0
                whichFeat = sFull_shuffled[iDim] % numFeaturesPerDimension
                sFull_shuffled[iDim] = fs[f0s.index(whichFeat)] if ifHuman else fs[f0s.index(whichFeat)] + numFeaturesPerDimension
            # shuffle dimensions
            sFull_shuffled_copy = deepcopy(sFull_shuffled)
            d0s = allPermsD[0]
            ds = allPermsD[iD]
            for iDim in range(numDimensions):
                sFull_shuffled[iDim] = sFull_shuffled_copy[d0s.index(ds[iDim])]
            S_sFull[isFull, iPerm] = sFullAll.index(sFull_shuffled)
    return S_sFull

# define S_aors: (i, iPerm) -> i_shuffled, used for action (human or computer) and stimulus
def get_S_aors(aorsAll, allPerms):
    S_aors = np.empty((len(aorsAll), len(allPerms))) * np.nan
    for iaors, aors in enumerate(aorsAll):
        for iPerm in range(len(allPerms)):
            [iD, iF0, iF1, iF2] = allPerms[iPerm]
            aors_shuffled = deepcopy(aors)
            # shuffle features
            iFs = [iF0, iF1, iF2]
            for iDim in range(numDimensions):
                f0s = allPermsF[iDim][0]
                fs = allPermsF[iDim][iFs[iDim]]
                if aors_shuffled[iDim] != 3:
                    aors_shuffled[iDim] = fs[f0s.index(aors_shuffled[iDim])]
            # shuffle dimensions
            aors_shuffled_copy = deepcopy(aors_shuffled)
            d0s = allPermsD[0]
            ds = allPermsD[iD]
            for iDim in range(numDimensions):
                aors_shuffled[iDim] = aors_shuffled_copy[d0s.index(ds[iDim])]
            S_aors[iaors, iPerm] = aorsAll.index(aors_shuffled)
    return S_aors



########## indices of other games ##########
def add_game_index(data):
    iGame = -1
    for i in range(data.shape[0]):
        if data.loc[i,'trial'] == 1:
            iGame += 1
        data.loc[i,'gameIndex'] = iGame
    return data


########## preprocessing function ##########
def preprocessing(data, ifAugment=False, ifShuffleD=True, ifShuffleF=True):
    
    numGames = int(np.sum(data['trial'] == 1))
    
    # data to RNN variables
    iWorker, gameStart, gameType, iaOld, icOld, isOld, isFullOld, rOld, iaNew = data2RNNVariables(data)
    
    if ifAugment:
        
        allPerms, numPermsTotal = get_permutation_info(ifShuffleD=ifShuffleD, ifShuffleF=ifShuffleF)

        # get the shuffling matrices
        S_sFull = get_S_sFull(sFullAll, allPerms)
        S_s = get_S_aors(sAll, allPerms)
        S_a = get_S_aors(aAll, allPerms)

        # augmenting the data
        iWorker_aug, gameStart_aug, gameType_aug, iaOld_aug, icOld_aug, isOld_aug, isFullOld_aug, rOld_aug, iaNew_aug = [np.zeros((numGames*numPermsTotal, gameLength)) for _ in range(9)]
        for iPerm in range(numPermsTotal):
            indices = list(range(iPerm*numGames, (iPerm+1)*numGames))
            iWorker_aug[indices, :], gameStart_aug[indices, :], gameType_aug[indices, :], rOld_aug[indices, :] = iWorker, gameStart, gameType, rOld
            for iGame in range(numGames):
                for i in range(gameLength):
                    iRow = iPerm * numGames + iGame
                    if np.isnan(iaNew[iGame, i]):  # invalid trials (padding at the end of game)
                        iaOld_aug[iRow, i], icOld_aug[iRow, i], isOld_aug[iRow, i], isFullOld_aug[iRow, i], iaNew_aug[iRow, i] = [np.nan] * 5
                    else:
                        iaOld_aug[iRow, i] = np.nan if np.isnan(iaOld[iGame, i]) else S_a[int(iaOld[iGame, i]), iPerm]
                        icOld_aug[iRow, i] = np.nan if np.isnan(icOld[iGame, i]) else S_a[int(icOld[iGame, i]), iPerm]
                        isOld_aug[iRow, i] = np.nan if np.isnan(isOld[iGame, i]) else S_s[int(isOld[iGame, i]), iPerm]
                        isFullOld_aug[iRow, i] = np.nan if np.isnan(isFullOld[iGame, i]) else S_sFull[int(isFullOld[iGame, i]), iPerm]
                        iaNew_aug[iRow, i] = S_a[int(iaNew[iGame, i]), iPerm]

        # save data into a dictionary
        data_dict = {'iWorker': iWorker_aug, 'gameStart': gameStart_aug, 'gameType': gameType_aug, 'iaOld': iaOld_aug, 'icOld': icOld_aug, 'isOld': isOld_aug, 'isFullOld': isFullOld_aug, 'rOld': rOld_aug, 'iaNew': iaNew_aug}
        
    else:
        
        # save data into a dictionary
        data_dict = {'iWorker': iWorker, 'gameStart': gameStart, 'gameType': gameType, 'iaOld': iaOld, 'icOld': icOld, 'isOld': isOld, 'isFullOld': isFullOld, 'rOld': rOld, 'iaNew': iaNew}
        
    return data_dict



########## Get input and output variables ##########
def ia2onehot(ia):  # ia -> a^i: 0,1,2,3 -> [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]
    a0_onehot, a1_onehot, a2_onehot = [np.zeros(4) for _ in range(3)]
    [a0, a1, a2] = aAll[int(ia)]
    a0_onehot[a0] = 1
    a1_onehot[a1] = 1
    a2_onehot[a2] = 1
    return a0_onehot, a1_onehot, a2_onehot
    
def is2onehot(is_):  # is -> s^i: 0,1,2 -> [1,0,0], [0,1,0], [0,0,1]
    s0_onehot, s1_onehot, s2_onehot = [np.zeros(3) for _ in range(3)]
    [s0, s1, s2] = sAll[int(is_)]
    s0_onehot[s0] = 1
    s1_onehot[s1] = 1
    s2_onehot[s2] = 1
    return s0_onehot, s1_onehot, s2_onehot

def isFull2onehot(isFull):  # isFull -> sFull^i: 0,1,2,3,4,5 -> [1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]
    sFull0_onehot, sFull1_onehot, sFull2_onehot = [np.zeros(6) for _ in range(3)]
    [sFull0, sFull1, sFull2] = sFullAll[int(isFull)]
    sFull0_onehot[sFull0] = 1
    sFull1_onehot[sFull1] = 1
    sFull2_onehot[sFull2] = 1
    return sFull0_onehot, sFull1_onehot, sFull2_onehot

def get_inputs_output(data_dict, whichset, input_type, output_type):
    
    _, workerIds = import_data(getWorkerIds=True)
    
    numGamesTotal = len(data_dict['gameStart'])
    
    ## inputs
    input_dict = dict()
    
    # keys for inputs
    keys = ['gameStart', 'gameType'] + ['aOld' + str(iDim) for iDim in range(numDimensions)] + ['sOld' + str(iDim) for iDim in range(numDimensions)] + ['rOld']
    if 'iWorkerOnehot' in input_type:
        keys = keys + ['iWorkerOnehot']
    
    # input variable initialization
    if 'iWorkerOnehot' in keys:
        input_dict['iWorkerOnehot'] = np.zeros((numGamesTotal, gameLength, len(workerIds)))
    input_dict['gameStart'] = deepcopy(data_dict['gameStart'])[:, :, np.newaxis]  # game start: 0 or 1
    input_dict['gameType'] = np.zeros((numGamesTotal, gameLength, 4))
    for iDim in range(numDimensions):
        input_dict['aOld' + str(iDim)] = np.empty((numGamesTotal, gameLength, 4)) * np.nan
        input_dict['sOld' + str(iDim)] = np.empty((numGamesTotal, gameLength, 3)) * np.nan
    input_dict['rOld'] = np.zeros((numGamesTotal, gameLength, 2))
    
    # get inputs
    for i1 in range(numGamesTotal):
        
        for i2 in range(gameLength):
            
            # record iWorker and gameType regardless of valid or invalid trials
            if 'iWorkerOnehot' in keys:
                input_dict['iWorkerOnehot'][i1, i2, int(data_dict['iWorker'][i1, i2])] = 1
            # game type: 0,1,2,3 -> [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]
            input_dict['gameType'][i1, i2, int(data_dict['gameType'][i1, i2])] = 1
        
            # invalid trials (padding)
            if np.isnan(data_dict['iaNew'][i1, i2]):
                for key in keys:
                    if key not in ['iWorkerOnehot', 'iWorkerID', 'gameStart', 'gameType']:
                        input_dict[key][i1, i2, :] = 0

            # valid trials
            else:
                # first trial of a game, only knows iWorker and game type
                if data_dict['gameStart'][i1, i2] == 1:
                    for key in keys:
                        if key not in ['iWorkerOnehot', 'iWorkerID', 'gameStart', 'gameType']:
                            input_dict[key][i1, i2, :] = 0

                # 2nd to last valid trials
                else:
                    # action and stimulus information of last trial
                    input_dict['aOld0'][i1, i2, :], input_dict['aOld1'][i1, i2, :], input_dict['aOld2'][i1, i2, :] = ia2onehot(data_dict['iaOld'][i1, i2])
                    input_dict['sOld0'][i1, i2, :], input_dict['sOld1'][i1, i2, :], input_dict['sOld2'][i1, i2, :] = is2onehot(data_dict['isOld'][i1, i2])

                    # rOld: 0,1 -> [1,0], [0,1]
                    input_dict['rOld'][i1, i2, int(data_dict['rOld'][i1, i2])] = 1

    # concatenate variables in input_dict to get inputs
    inputs = np.concatenate([input_dict[key] for key in keys], axis=2)

    ## target output
    target_output = np.zeros((numGamesTotal, gameLength))
    for i1 in range(numGamesTotal):
        for i2 in range(gameLength):
            if np.isnan(data_dict['iaNew'][i1, i2]):  # invalid trials
                target_output[i1, i2] = np.nan
            else:
                target_output[i1, i2] = data_dict['iaNew'][i1, i2]
    
    pickle.dump({'inputs':inputs, 'target_output':target_output}, open('data/data_RNN_' + whichset + '_input_' + input_type + '_output_' + output_type + '.p', 'wb'), protocol = 4)
    
    return inputs, target_output