import numpy as np
import pandas
from scipy.special import softmax
import torch

from nets import *

from info_task_exp import *
from funcs_data_preprocessing import *


def simulation(dataInfo, model, numGamePerType, x_other=None, nTrialsRealData=0, data_real=None):
    _, workerIds = import_data(getWorkerIds=True)

    # create the variables
    subID, game, trial, informedList, numRelevantDimensionsList, reward, numSelectedFeatures, rt = \
        zerosLists(numList=8, lengthList=len(workerIds)*gameLength*numGamePerType*6)
    ifRelevantDimension, rewardingFeature, selectedFeature, randomlySelectedFeature, builtFeature = \
        emptyDicts(numDict=5, keys=DIMENSIONS, lengthList=len(workerIds)*gameLength*numGamePerType*6)

    # simulation
    iRow = 0
    iGame = 0
    for subjID, workerId in enumerate(workerIds):
        for informed in [True, False]:
            for numRelevantDimensions in np.arange(3) + 1:

                # save game info
                subID[iRow:(iRow + gameLength * numGamePerType)] = [subjID] * gameLength * numGamePerType
                informedList[iRow:(iRow + gameLength * numGamePerType)] = [informed] * gameLength * numGamePerType
                numRelevantDimensionsList[iRow:(iRow + gameLength * numGamePerType)] = [numRelevantDimensions] * gameLength * numGamePerType

                for iRepeat in range(numGamePerType):
                    # generate and save game-specific (reward) setting
                    game[iRow:(iRow + gameLength)] = [iGame + 1] * gameLength
                    relevantDimensions = np.random.choice(DIMENSIONS, size=numRelevantDimensions, replace=False)
                    for dim in DIMENSIONS:
                        if dim in relevantDimensions:
                            ifRelevantDimension[dim][iRow:(iRow + gameLength)] = [True] * gameLength
                            rewardingFeature[dim][iRow:(iRow + gameLength)] = [np.random.choice(DIMENSIONS_TO_FEATURES[dim], size=1)[0]] * gameLength
                        else:
                            ifRelevantDimension[dim][iRow:(iRow + gameLength)] = [False] * gameLength
                            rewardingFeature[dim][iRow:(iRow + gameLength)] = [np.nan] * gameLength

                    input_net = torch.zeros((1, gameLength, dataInfo['input_size']))
                    
                    if nTrialsRealData > 0:  # use real data
                        # find the corresponding game in real data
                        gameIndices = data_real.loc[(data_real['workerId']==workerId)&(data_real['informed']==informed)&(data_real['numRelevantDimensions']==numRelevantDimensions), 'gameIndex'].unique()
                        gameIndex_this = gameIndices[iRepeat % len(gameIndices)]
                        iRow_real = data_real.index[(data_real['gameIndex']==gameIndex_this)&(data_real['trial']==1)].values[0]

                    # simulation
                    for iTrial in range(gameLength):

                        trial[iRow] = iTrial + 1
                        
                        if iTrial < nTrialsRealData:
                            while pd.isnull(data_real.loc[iRow_real, 'rt']):
                                iRow_real += 1

                        # create input for the network
                        g = torch.zeros(4)
                        sFull = torch.zeros(18)
                        a = torch.zeros(12)
                        c = torch.zeros(12)
                        s = torch.zeros(9)
                        r = torch.zeros(2)
                        ID = torch.zeros(len(workerIds))
                        
                        ID[subjID] = 1
                        if iTrial == 0:
                            gs = torch.tensor([1.])
                            g[numRelevantDimensions if informed == True else 0] = 1
                        else:
                            gs = torch.tensor([0.])
                            g[numRelevantDimensions if informed == True else 0] = 1
                            for iDim, dim in enumerate(DIMENSIONS):
                                if not pd.isnull(selectedFeature[dim][iRow-1]):
                                    sFull[iDim * 6 + DIMENSIONS_TO_FEATURES[dim].index(selectedFeature[dim][iRow-1])] = 1
                                    a[iDim * 4 + DIMENSIONS_TO_FEATURES[dim].index(selectedFeature[dim][iRow-1])] = 1
                                    c[iDim * 4 + 3] = 1
                                    s[iDim * 3 + DIMENSIONS_TO_FEATURES[dim].index(selectedFeature[dim][iRow-1])] = 1
                                else:
                                    sFull[iDim * 6 + 3 + DIMENSIONS_TO_FEATURES[dim].index(randomlySelectedFeature[dim][iRow-1])] = 1
                                    a[iDim * 4 + 3] = 1
                                    c[iDim * 4 + DIMENSIONS_TO_FEATURES[dim].index(randomlySelectedFeature[dim][iRow-1])] = 1
                                    s[iDim * 3 + DIMENSIONS_TO_FEATURES[dim].index(randomlySelectedFeature[dim][iRow-1])] = 1
                            r[int(reward[iRow-1])] = 1
                            
                        if 'sFull' in dataInfo['input_type']:
                            inputs_list = [gs, g, sFull, r]
                        elif 'ac' in dataInfo['input_type']:
                            inputs_list = [gs, g, a, c, r]
                        elif 'as' in dataInfo['input_type']:
                            inputs_list = [gs, g, a, s, r]
                        if 'iWorkerOnehot' in dataInfo['input_type']:
                            inputs_list.append(ID)
                        input_net[:, iTrial, :] = torch.cat(inputs_list, dim=0)
                        if not x_other:
                            x = input_net
                        else:
                            x = (input_net, x_other[subjID])
                        
                        if iTrial < nTrialsRealData:  # get choice, stimulus and reward (from realdata)
                            
                            for iDim, dim in enumerate(DIMENSIONS):
                                selectedFeature[dim][iRow] = data_real.loc[iRow_real, 'selectedFeature_'+dim]
                                randomlySelectedFeature[dim][iRow] = data_real.loc[iRow_real, 'randomlySelectedFeature_'+dim]
                                builtFeature[dim][iRow] = data_real.loc[iRow_real, 'builtFeature_'+dim]
                            numSelectedFeatures[iRow] = data_real.loc[iRow_real, 'numSelectedFeatures']
                            reward[iRow] = data_real.loc[iRow_real, 'reward']
                            iRow_real += 1
                        
                        else:  # generate choice (from model), stimulus and reward

                            # choice phase
                            output, _ = model(x)
                            if dataInfo['output_type'] == 'separate':
                                choice = []
                                for iDim in range(numDimensions):
                                    sampleP = softmax(output[iTrial, iDim*4:(iDim+1)*4].detach().numpy())
                                    indChoice = np.random.choice(np.array(list(range(numFeaturesPerDimension))+[3]), size=1, p=sampleP)[0]
                                    if indChoice < 3:
                                        choice.append(indChoice)
                                    else:
                                        choice.append(np.nan)
                            elif dataInfo['output_type'] == 'joint':
                                sampleP = softmax(output[iTrial, :].detach().numpy())
                                indChoice = np.random.choice(np.arange(len(aAll)), size=1, p=sampleP)[0]
                                choice = [ai if ai != 3 else np.nan for ai in aAll[indChoice]]

                            # generate stimulus and reward outcome
                            stimulus = deepcopy(choice)
                            for iDim, dim in enumerate(DIMENSIONS):
                                if ~np.isnan(choice[iDim]):
                                    selectedFeature[dim][iRow] = DIMENSIONS_TO_FEATURES[dim][choice[iDim]]
                                    randomlySelectedFeature[dim][iRow] = np.nan
                                    builtFeature[dim][iRow] = selectedFeature[dim][iRow]
                                else:
                                    selectedFeature[dim][iRow] = np.nan
                                    stimulus[iDim] = np.random.choice(np.arange(len(DIMENSIONS_TO_FEATURES[dim])), 1)[0]
                                    randomlySelectedFeature[dim][iRow] = DIMENSIONS_TO_FEATURES[dim][stimulus[iDim]]
                                    builtFeature[dim][iRow] = randomlySelectedFeature[dim][iRow]
                            numSelectedFeatures[iRow] = np.array(
                                [(not pandas.isnull(selectedFeature[dim][iRow])) for dim in DIMENSIONS]).sum()
                            numRewardingFeatureBuilt = np.array([((not pandas.isnull(rewardingFeature[dim][iRow])) &
                                                                  (builtFeature[dim][iRow] == rewardingFeature[dim][iRow]))
                                                                 for dim in DIMENSIONS]).sum()
                            reward[iRow] = int(
                                        np.random.random() < rewardSetting[numRelevantDimensions - 1][numRewardingFeatureBuilt])

                        iRow += 1

                    iGame += 1

    # save variables into dataframe
    simudata = pandas.DataFrame(
        {'ID': subID, 'game': game, 'trial': trial, 'informed': informedList, 'numRelevantDimensions': numRelevantDimensionsList,
         'reward': reward, 'numSelectedFeatures': numSelectedFeatures, 'rt': rt,
         'ifRelevantDimension_color': ifRelevantDimension['color'], 'rewardingFeature_color': rewardingFeature['color'],
         'selectedFeature_color': selectedFeature['color'],
         'randomlySelectedFeature_color': randomlySelectedFeature['color'], 'builtFeature_color': builtFeature['color'],
         'ifRelevantDimension_shape': ifRelevantDimension['shape'], 'rewardingFeature_shape': rewardingFeature['shape'],
         'selectedFeature_shape': selectedFeature['shape'],
         'randomlySelectedFeature_shape': randomlySelectedFeature['shape'], 'builtFeature_shape': builtFeature['shape'],
         'ifRelevantDimension_pattern': ifRelevantDimension['pattern'],
         'rewardingFeature_pattern': rewardingFeature['pattern'], 'selectedFeature_pattern': selectedFeature['pattern'],
         'randomlySelectedFeature_pattern': randomlySelectedFeature['pattern'],
         'builtFeature_pattern': builtFeature['pattern']})

    return simudata