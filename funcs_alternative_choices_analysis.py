import numpy as np
import pandas as pd
from info_task_exp import *
from funcs_data_analysis import *


numDimDiff_dict, typeDiff_dict = dict(), dict()
for modelType in ['cog', 'RNN']:
    allChoices = allChoices_dict[modelType]
    NChoices = len(allChoices)
    numDimDiff, typeDiff = np.zeros((NChoices, NChoices)), np.zeros((NChoices, NChoices))
    for i1, choice1 in enumerate(allChoices):
        for i2, choice2 in enumerate(allChoices):
            # number of dimensions different
            numDimDiff[i1, i2] = numDimensions - np.sum([choice1[iDim] == choice2[iDim] or (np.isnan(choice1[iDim]) and np.isnan(choice2[iDim])) for iDim in range(numDimensions)])
            # type of different (0:same, 1: more dim, 2: less dim, 3: wrong feature, 4: wrong dim, 5: mix)
            typediff = [0] * numDimensions
            for iDim in range(numDimensions):
                if np.isnan(choice1[iDim]) and not np.isnan(choice2[iDim]):
                    typediff[iDim] = 1
                if not np.isnan(choice1[iDim]) and np.isnan(choice2[iDim]):
                    typediff[iDim] = 2
                if not np.isnan(choice1[iDim]) and not np.isnan(choice2[iDim]):
                    if choice1[iDim] != choice2[iDim]:
                        typediff[iDim] = 3
            if all([typediff[iDim] == 0 for iDim in range(numDimensions)]):
                typeDiff[i1, i2] = 0
            elif all([typediff[iDim] in [0,1] for iDim in range(numDimensions)]):
                typeDiff[i1, i2] = 1
            elif all([typediff[iDim] in [0,2] for iDim in range(numDimensions)]):
                typeDiff[i1, i2] = 2
            elif all([typediff[iDim] in [0,3] for iDim in range(numDimensions)]):
                typeDiff[i1, i2] = 3
            elif all([typediff[iDim] in [0,1,2] for iDim in range(numDimensions)]):
                typeDiff[i1, i2] = 4
            else:
                typeDiff[i1, i2] = 5
    numDimDiff_dict[modelType] = numDimDiff
    typeDiff_dict[modelType] = typeDiff
typediff_list = ['same', 'more-dim', 'less-dim', 'wrong-feature', 'wrong-dim', 'mix']


def getPConfusion(data, modelType):
    allChoices = allChoices_dict[modelType]
    # iChoice
    data['iChoice'] = [allChoices.index([DIMENSIONS_TO_FEATURES[dim].index(data.loc[i, 'selectedFeature_' + dim]) if not pd.isnull(data.loc[i, 'selectedFeature_' + dim]) else np.nan for dim in DIMENSIONS]) for i in data.index]
    # lik of real choice
    data['lik'] = [data.loc[i, 'p_allChoices' + str(data.loc[i, 'iChoice'])] for i in data.index]
    # whether the true choice has the highest likelihood
    data['isHighestLik'] = [all(data.loc[i, 'p_allChoices' + str(data.loc[i, 'iChoice'])] >= [data.loc[i, 'p_allChoices' + str(iChoice)] for iChoice in range(len(allChoices))]) for i in data.index]
    # alternative choice: # features selected
    for numFeat in range(numFeaturesPerDimension+1):
        data['ptotal_numFeatureSelected' + str(numFeat)] = [np.sum([data.loc[i, 'p_allChoices' + str(iChoice)] for iChoice in np.where([3-np.sum(np.isnan(choice))==numFeat for choice in allChoices])[0]]) for i in data.index]
    # num different dimensions
    for numdimdiff in range(numDimensions+1):
        data['ptotal_numdimdiff' + str(numdimdiff)] = [np.sum([data.loc[i, 'p_allChoices' + str(iChoice)] for iChoice in np.where(numDimDiff_dict[modelType][data.loc[i, 'iChoice'], :] == numdimdiff)[0]]) for i in data.index]
    # type of difference
    for iType in range(len(typediff_list)):
        data['ptotal_typediff_' + typediff_list[iType]] = [np.sum([data.loc[i, 'p_allChoices' + str(iChoice)] for iChoice in np.where(typeDiff_dict[modelType][data.loc[i, 'iChoice'], :] == iType)[0]]) for i in data.index]
    # difference in number of features selected 
    for diff in range(-numDimensions, numDimensions+1):
        data['ptotal_numFeaturesSelectedDiff' + str(diff)] = [np.sum([data.loc[i, 'p_allChoices' + str(iChoice)] for iChoice in np.where([(3-np.sum(np.isnan(allChoice)))==(data.loc[i, 'numSelectedFeatures']+diff) for allChoice in allChoices])[0]]) if 0<=(data.loc[i, 'numSelectedFeatures']+diff)<=3 else np.nan for i in data.index]
    # last choice
    data_valid = data[~pd.isnull(data['rt'])].copy().reset_index(drop=True)
    choiceChange = (np.sum([(data_valid['selectedFeature_' + dim].iloc[1:].values == data_valid['selectedFeature_' + dim].iloc[:-1].values) | (pd.isnull(data_valid['selectedFeature_' + dim].iloc[1:]).values & pd.isnull(data_valid['selectedFeature_' + dim].iloc[:-1]).values) for dim in DIMENSIONS],axis=0)<3).astype(np.float)
    data_valid['add_choiceChange'] = mark_1st_trials_nan(choiceChange, data_valid)
    iLastChoice = [allChoices.index([np.nan if pd.isnull(data_valid.loc[iLast, 'selectedFeature_' + dim]) else DIMENSIONS_TO_FEATURES[dim].index(data_valid.loc[iLast, 'selectedFeature_' + dim]) for dim in DIMENSIONS]) for iLast in data_valid.index[:-1]]
    data_valid['add_iLastChoice'] = mark_1st_trials_nan(iLastChoice, data_valid)
    for col in ['add_choiceChange', 'add_iLastChoice']:
        data.loc[~pd.isnull(data['rt']), col] = data_valid[col].values
        data.loc[pd.isnull(data['rt']), col] = np.nan
    # num different dimensions (compared to previous choice)
    for numdimdiff in range(numDimensions+1):
        data['ptotal_complast_numdimdiff' + str(numdimdiff)] = [np.sum([data.loc[i, 'p_allChoices' + str(iChoice)] for iChoice in np.where(numDimDiff_dict[modelType][int(data.loc[i, 'add_iLastChoice']), :] == numdimdiff)[0]]) if not pd.isnull(data.loc[i, 'add_iLastChoice']) else np.nan for i in data.index]
    # alternative choice as switch
    data['ptotal_switch_numFeaturesSelectedMore'] = [np.nansum([np.sum([data.loc[i, 'p_allChoices' + str(iChoice)] for iChoice in np.where([(3-np.sum(np.isnan(allChoices[iChoice])))==(data.loc[i, 'numSelectedFeatures']+diff) and iChoice != data.loc[i, 'add_iLastChoice'] for iChoice in range(len(allChoices))])[0]]) if 0<=(data.loc[i, 'numSelectedFeatures']+diff)<=3 else np.nan for diff in range(1,numDimensions+1)]) for i in data.index]
    data['ptotal_switch_numFeaturesSelectedLess'] = [np.nansum([np.sum([data.loc[i, 'p_allChoices' + str(iChoice)] for iChoice in np.where([(3-np.sum(np.isnan(allChoices[iChoice])))==(data.loc[i, 'numSelectedFeatures']+diff) and iChoice != data.loc[i, 'add_iLastChoice'] for iChoice in range(len(allChoices))])[0]]) if 0<=(data.loc[i, 'numSelectedFeatures']+diff)<=3 else np.nan for diff in range(-numDimensions,0)]) for i in data.index]
    # mark invalid trials as nan
    cols = ['iChoice', 'lik', 'isHighestLik'] + ['ptotal_numFeatureSelected' + str(numFeat) for numFeat in range(numFeaturesPerDimension+1)] + ['ptotal_numdimdiff' + str(numdimdiff) for numdimdiff in range(numDimensions+1)] + ['ptotal_typediff_' + typediff for typediff in typediff_list] + ['ptotal_numFeaturesSelectedDiff' + str(diff) for diff in range(-numDimensions, numDimensions+1)] + ['ptotal_switch_numFeaturesSelectedMore', 'ptotal_switch_numFeaturesSelectedLess'] + ['ptotal_complast_numdimdiff' + str(numdimdiff) for numdimdiff in range(numDimensions+1)]
    for col in cols:
        data.loc[pd.isnull(data['rt']), col] = np.nan
    return data