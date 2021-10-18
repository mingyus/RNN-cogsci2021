import numpy as np
import pandas as pd
from info_task_exp import *


def mark_1st_trials_nan(var_list, data_valid): # the first trial of a game (not including the first game)
    nan_index = np.where((data_valid['trial'].iloc[1:].values-data_valid['trial'].iloc[:-1].values)<0)[0]
    for i in nan_index:
        var_list[i] = np.nan
    return np.concatenate((np.nan, var_list), axis = None)


def get_choiceChange_info(data):
    allChoices = allChoices_dict['cog']
    
    ## exclude no-response trials
    data_valid = data[~pd.isnull(data['rt'])].copy().reset_index(drop=True)
    
    ## last reward
    lastReward = [data_valid.loc[iLast, 'reward'] for iLast in range(len(data_valid)-1)]
    data_valid['add_lastReward'] = mark_1st_trials_nan(lastReward, data_valid)
    
    ## change points of choices
    choiceChange = (np.sum([(data_valid['selectedFeature_' + dim].iloc[1:].values == data_valid['selectedFeature_' + dim].iloc[:-1].values) | (pd.isnull(data_valid['selectedFeature_' + dim].iloc[1:]).values & pd.isnull(data_valid['selectedFeature_' + dim].iloc[:-1]).values) for dim in DIMENSIONS],axis=0)<3).astype(np.float)
    data_valid['add_choiceChange'] = mark_1st_trials_nan(choiceChange, data_valid)
    
    ## last choice
    iLastChoice = [allChoices.index([np.nan if pd.isnull(data_valid.loc[iLast, 'selectedFeature_' + dim]) else DIMENSIONS_TO_FEATURES[dim].index(data_valid.loc[iLast, 'selectedFeature_' + dim]) for dim in DIMENSIONS]) for iLast in range(len(data_valid)-1)]
    data_valid['add_iLastChoice'] = mark_1st_trials_nan(iLastChoice, data_valid)
    
    ## count the number of features changed in each choice
    numDimensionsChange = 3.0 - np.sum([(data_valid['selectedFeature_' + dim].iloc[1:].values == data_valid['selectedFeature_' + dim].iloc[:-1].values) | (pd.isnull(data_valid['selectedFeature_' + dim].iloc[1:]).values & pd.isnull(data_valid['selectedFeature_' + dim].iloc[:-1]).values) for dim in DIMENSIONS], axis=0)
    data_valid['add_numDimensionsChange'] = mark_1st_trials_nan(numDimensionsChange, data_valid)
    
    ## count the change of total number of features selected
    numFeaturesSelectedChange = data_valid['numSelectedFeatures'].iloc[1:].values - data_valid['numSelectedFeatures'].iloc[:-1].values
    data_valid['add_numFeaturesSelectedChange'] = mark_1st_trials_nan(numFeaturesSelectedChange, data_valid)

    ## label the type of choice change
    # if there is no choice change, mark as nan
    for dim in DIMENSIONS:
        data_valid.loc[:,'add_choiceChangeType_'+dim] = np.nan
    data_valid.loc[:,'add_choiceChangeType_overall'] = np.nan
    data_valid.loc[:,'add_choiceChangeTypeCount'] = np.nan
    data_valid.loc[:,'add_choiceChangeTypeList'] = np.nan
    data_valid.loc[:,'add_choiceChangewSame'] = 0
    data_valid.loc[data_valid['add_choiceChange']==False,'add_choiceChangewSame'] = np.nan
    data_valid['add_choiceChangeTypeList'] = data_valid['add_choiceChangeTypeList'].astype(object)

    for irow in np.where(data_valid['add_choiceChange']==True)[0]:
        choiceChangeTypeList = []
        for dim in DIMENSIONS:
            if (pd.isnull(data_valid.loc[irow-1,'selectedFeature_'+dim])) & (pd.isnull(data_valid.loc[irow,'selectedFeature_'+dim])):
                data_valid.loc[irow,'add_choiceChangeType_'+dim] = 'no_choice'
            elif data_valid.loc[irow-1,'selectedFeature_'+dim] == data_valid.loc[irow,'selectedFeature_'+dim]:
                data_valid.loc[irow,'add_choiceChangeType_'+dim] = 'same'
                data_valid.loc[irow,'add_choiceChangewSame'] = 1
            elif (pd.isnull(data_valid.loc[irow-1,'selectedFeature_'+dim])) & (~pd.isnull(data_valid.loc[irow,'selectedFeature_'+dim])):
                data_valid.loc[irow,'add_choiceChangeType_'+dim] = 'add'
                choiceChangeTypeList.append('add')
            elif (~pd.isnull(data_valid.loc[irow-1,'selectedFeature_'+dim])) & (pd.isnull(data_valid.loc[irow,'selectedFeature_'+dim])):
                data_valid.loc[irow,'add_choiceChangeType_'+dim] = 'drop'
                choiceChangeTypeList.append('drop')
            elif data_valid.loc[irow-1,'selectedFeature_'+dim] != data_valid.loc[irow,'selectedFeature_'+dim]:
                data_valid.loc[irow,'add_choiceChangeType_'+dim] = 'switch_within'
                choiceChangeTypeList.append('switch_within')
        data_valid.loc[irow,'add_choiceChangeTypeCount'] = len(choiceChangeTypeList)

        if len(set(choiceChangeTypeList)) == 1: # only one type of changes
            data_valid.loc[irow,'add_choiceChangeType_overall'] = choiceChangeTypeList[0]
        elif (data_valid.loc[irow,'add_choiceChangeTypeCount'] == 2) & np.isin('add',choiceChangeTypeList) & np.isin('drop',choiceChangeTypeList):
            data_valid.loc[irow,'add_choiceChangeType_overall'] = 'switch_across'
        else:
            data_valid.loc[irow,'add_choiceChangeType_overall'] = 'mixed'

        choiceChangeTypeList.sort()
        data_valid.at[irow,'add_choiceChangeTypeList'] = '-'.join(choiceChangeTypeList) 
    
    ## assign values to original dataframe
    colNames = ['add_lastReward', 'add_choiceChange', 'add_iLastChoice', 'add_numDimensionsChange', 'add_numFeaturesSelectedChange'] + ['add_choiceChangeType_'+dim for dim in DIMENSIONS] + ['add_choiceChangeType_overall', 'add_choiceChangeTypeCount', 'add_choiceChangeTypeList', 'add_choiceChangewSame']
    for col in colNames:
        data.loc[~pd.isnull(data['rt']), col] = data_valid[col].values
        data.loc[pd.isnull(data['rt']), col] = np.nan
        
    return data


def calculate_metrics(data, dataType):
    if dataType == 'real':
        colID = 'workerId'
    elif dataType == 'simu':
        colID = 'ID'
    
    ## follow instructions
    data['ifFollowInstruction'] = data['numSelectedFeatures'] <= data['numRelevantDimensions']
    data.loc[data['informed'] == False, 'ifFollowInstruction'] = None
    data.loc[data['rt'].isnull(), 'ifFollowInstruction'] = None

    ## choice change
    if 'add_choiceChange' not in data.keys() or 'add_numDimensionsChange' not in data.keys():
        data_valid = data[~pd.isnull(data['rt'])].copy().reset_index(drop=True)
        # mark choice change point
        choiceChange = (np.sum([(data_valid['selectedFeature_' + dim].iloc[1:].values == data_valid['selectedFeature_' + dim].iloc[:-1].values) | (pd.isnull(data_valid['selectedFeature_' + dim].iloc[1:]).values & pd.isnull(data_valid['selectedFeature_' + dim].iloc[:-1]).values) for dim in DIMENSIONS],axis=0)<3).astype(np.float)
        data.loc[~pd.isnull(data['rt']),'add_choiceChange'] = mark_1st_trials_nan(choiceChange, data_valid)
        data.loc[pd.isnull(data['rt']), 'add_choiceChange'] = np.nan
        # count the number of features changed in each choice
        numDimensionsChange = 3.0 - np.sum([(data_valid['selectedFeature_' + dim].iloc[1:].values == data_valid['selectedFeature_' + dim].iloc[:-1].values) | (pd.isnull(data_valid['selectedFeature_' + dim].iloc[1:]).values & pd.isnull(data_valid['selectedFeature_' + dim].iloc[:-1]).values) for dim in DIMENSIONS], axis=0)
        data.loc[~pd.isnull(data['rt']),'add_numDimensionsChange'] = mark_1st_trials_nan(numDimensionsChange, data_valid)
        data.loc[pd.isnull(data['rt']), 'add_numDimensionsChange'] = np.nan
        data.loc[data['add_choiceChange']==0, 'add_numDimensionsChange'] = np.nan
        # change in number of feature selected
        numFeatureSelectedChange = (data_valid['numSelectedFeatures'].iloc[1:].values - data_valid['numSelectedFeatures'].iloc[:-1].values) * 1.0
        data.loc[~pd.isnull(data['rt']),'add_numFeatureSelectedChange'] = mark_1st_trials_nan(numFeatureSelectedChange, data_valid)
        data.loc[pd.isnull(data['rt']), 'add_numFeatureSelectedChange'] = np.nan
        data.loc[data['add_choiceChange']==0, 'add_numFeatureSelectedChange'] = np.nan
        data['add_numFeatureSelectedChangeAbs'] = np.abs(data['add_numFeatureSelectedChange'])

    ## metrics
    IDs = data[colID].unique()
    metrics = pd.DataFrame(index=IDs)
    metrics['numSelectedFeatures'] = data.groupby(colID)['numSelectedFeatures'].agg(np.nanmean)[IDs]
    metrics['reward'] = data.groupby(colID)['reward'].agg(np.nanmean)[IDs]
    metrics['pChoiceChange'] = data.groupby(colID).mean()['add_choiceChange'][IDs]
    metrics['numDimensionsChange'] = data.groupby(colID).mean()['add_numDimensionsChange'][IDs]
    metrics['numFeatureSelectedChange'] = data.groupby(colID).mean()['add_numFeatureSelectedChange'][IDs]
    metrics['numFeatureSelectedChangeAbs'] = data.groupby(colID).mean()['add_numFeatureSelectedChangeAbs'][IDs]
    
    data['add_numDimensionsChange_with0'] = data['add_numDimensionsChange']
    data.loc[data['add_choiceChange']==0, 'add_numDimensionsChange_with0'] = 0
    metrics['numDimensionsChange_with0'] = data.groupby(colID).mean()['add_numDimensionsChange_with0'][IDs]
    
    return data, metrics