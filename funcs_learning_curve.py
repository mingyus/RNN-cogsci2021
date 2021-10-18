import numpy as np
import pandas as pd
from pandas import DataFrame, concat, read_csv
from statsmodels.stats.anova import AnovaRM
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import comb
from scipy.stats import binom
from info_task_exp import *


def plot_learningCurve(data, varName, singleWorker=None):
    
    plotType = 'separate'  # 'separate' or 'collapsed'; whether to plot informed and uninformed game together or separately
    empiricalChance = True
    wLegend = False
    
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(8, 2.5))
    fontsize = 12.5
    plt.rcParams.update({'font.size': fontsize})
    linewidth = 2
    axes_linewidth = 1.3
    
    titles = {
        'NumSelected': '# features selected',
        'ExpectedReward': 'Expected reward rate',
    }

    gameLength = pd.DataFrame.max(data['trial'])
    trial_index = np.arange(gameLength)+1

    for numRelevantDimensions in np.arange(3) + 1:
        for idx, informed in enumerate([True, False]):
            learning_curves, chance_curves = [], []
            if singleWorker is None:
                current_df = data[(data['numRelevantDimensions'] == numRelevantDimensions) & (data['informed'] == informed)].copy()
            else:
                if 'workerId' in data.keys():
                    current_df = data[(data['numRelevantDimensions'] == numRelevantDimensions) & (data['informed'] == informed) & (data['workerId'] == singleWorker)].copy()
                elif 'ID' in data.keys():
                    current_df = data[(data['numRelevantDimensions'] == numRelevantDimensions) & (data['informed'] == informed) & (data['ID'] == singleWorker)].copy()
            learning_curve, _ = get_learning_curve(current_df, varName, numRelevantDimensions)
            learning_curves.append(learning_curve)
            chance_curves.append(get_chance_curve(empiricalChance, current_df, varName, numRelevantDimensions, gameLength))
            ax = axes[numRelevantDimensions - 1]
            average_values = np.squeeze(np.nanmean(np.stack(learning_curves, axis=1),axis=1))
            sem_values = np.squeeze(np.nanstd(np.stack(learning_curves, axis=1),axis=1)/np.sqrt(np.stack(learning_curves, axis=1).shape[1]))
            ax.plot(trial_index, average_values, color='red' if informed else 'blue', lw=linewidth)
            ax.fill_between(trial_index, average_values - sem_values, average_values + sem_values, lw=0, alpha=0.3, color='red' if informed else 'blue')
            ax.plot(trial_index, np.squeeze(np.nanmean(np.stack(chance_curves, axis=1),axis=1)), color='k', lw=linewidth, ls='--')
            ylimValues = {
                'Built': [0, numRelevantDimensions],
                'NumSelected': [0, 3],
                'ExpectedReward': [0, 1],
            }
            ax.set_ylim(ylimValues[varName])
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', length=3.5, labelsize=fontsize, pad=4.5, width=axes_linewidth)
        ax.set_xlim([0, 30])
        xticklabels = [0, 15, 30]
        ax.set_xticks(xticklabels)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('Trial', labelpad=8, fontsize=fontsize)
        ax.spines['bottom'].set_linewidth(axes_linewidth)
        ax.spines['left'].set_linewidth(axes_linewidth)
    fig.suptitle(titles[varName], fontsize=fontsize+0.5, y=0.9)
    plt.subplots_adjust(wspace=0.4, top=0.75, bottom=0.25)


def get_chance_curve(empiricalChance, current_df, varName, numRelevantDimension, game_length, ifdata = True):
    if varName == 'NumSelected':
        return [np.nan]*game_length
    elif varName == 'ExpectedReward':
        return [0.4] * game_length


def get_learning_curve(current_df, varName, numRelevantDimension, ifdata = True):
    if varName == 'ExpectedReward':
        current_df['add_numFeatureSelected'] = concat([(~current_df['selectedFeature_'+dim].isnull()).astype(int) for dim in DIMENSIONS], axis = 1, keys = DIMENSIONS).sum(axis = 1)
        current_df['add_numRewardingFeaturesSelected'] = concat([(~current_df['rewardingFeature_'+dim].isnull() & \
            (current_df['selectedFeature_'+dim] == current_df['rewardingFeature_'+dim])).astype(int) \
            for dim in DIMENSIONS], axis = 1, keys = DIMENSIONS).sum(axis = 1)
        current_df['add_expectedNumRewardingFeaturesBuilt'] = current_df['add_numRewardingFeaturesSelected'] + concat([(~current_df['rewardingFeature_'+dim].isnull() & \
            current_df['selectedFeature_'+dim].isnull()).astype(int) for dim in DIMENSIONS], axis = 1, keys = DIMENSIONS).sum(axis = 1)/3
        if varName == 'ExpectedReward':
            tmp_df = current_df.reset_index(drop=True)
            tmp_df['numRelevantDimensionsUnselected'] = concat([(~tmp_df['rewardingFeature_'+dim].isnull() & tmp_df['selectedFeature_'+dim].isnull()).astype(int) for dim in DIMENSIONS], axis = 1, keys = DIMENSIONS).sum(axis = 1)
            current_df['add_expectedReward'] = [np.sum([binom.pmf(k=i, n=tmp_df.loc[iRow, 'numRelevantDimensionsUnselected'], p=1/3) *
                                                        rewardSetting[int(tmp_df.loc[iRow, 'numRelevantDimensions']-1)][int(tmp_df.loc[iRow, 'add_numRewardingFeaturesSelected']) + i] for i in range(int(tmp_df.loc[iRow, 'numRelevantDimensionsUnselected']+1))]) for iRow in range(tmp_df.shape[0])]
    returnVarName = {
        'NumSelected': 'numSelectedFeatures',
        'ExpectedReward': 'add_expectedReward',
    }
    if ifdata:
        current_df.loc[current_df['rt'].isnull(),returnVarName[varName]] = np.nan
    return current_df.groupby('trial').agg({returnVarName[varName]:np.nanmean})[returnVarName[varName]].values, current_df[returnVarName[varName]]

