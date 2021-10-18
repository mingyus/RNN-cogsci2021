import numpy as np

# task setting
DIMENSIONS = ('color', 'shape', 'pattern')
DIMENSIONS_TO_FEATURES = {
    'color': ['red','blue','green'],
    'shape': ['square','circle','triangle'],
    'pattern': ['plaid','dots','waves']
}
numDimensions = len(DIMENSIONS)
numFeaturesPerDimension = len(DIMENSIONS_TO_FEATURES[DIMENSIONS[0]])
rewardSetting = [[0.2,0.8],[0.2,0.5,0.8],[0.2,0.4,0.6,0.8]]

# exp info
gameLength = 30
NGames = 18

# choices
allChoices_dict = dict()
allChoices_dict['cog'] = [
    # doesn't select anything
    [np.nan,np.nan,np.nan],
    # select on 1 dimension
    [0,np.nan,np.nan],[1,np.nan,np.nan],[2,np.nan,np.nan],[np.nan,0,np.nan],[np.nan,1,np.nan],[np.nan,2,np.nan],[np.nan,np.nan,0],[np.nan,np.nan,1],[np.nan,np.nan,2],
    # select on 2 dimensions
    [0,0,np.nan],[0,1,np.nan],[0,2,np.nan],[0,np.nan,0],[0,np.nan,1],[0,np.nan,2],
    [1,0,np.nan],[1,1,np.nan],[1,2,np.nan],[1,np.nan,0],[1,np.nan,1],[1,np.nan,2],
    [2,0,np.nan],[2,1,np.nan],[2,2,np.nan],[2,np.nan,0],[2,np.nan,1],[2,np.nan,2],
    [np.nan,0,0],[np.nan,0,1],[np.nan,0,2],[np.nan,1,0],[np.nan,1,1],[np.nan,1,2],[np.nan,2,0],[np.nan,2,1],[np.nan,2,2],
    # select on 3 dimensions
    [0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,1,2],[0,2,0],[0,2,1],[0,2,2],
    [1,0,0],[1,0,1],[1,0,2],[1,1,0],[1,1,1],[1,1,2],[1,2,0],[1,2,1],[1,2,2],
    [2,0,0],[2,0,1],[2,0,2],[2,1,0],[2,1,1],[2,1,2],[2,2,0],[2,2,1],[2,2,2],
]
allChoices_dict['RNN'] = [[a0, a1, a2] for a0 in [0,1,2,np.nan] for a1 in [0,1,2,np.nan] for a2 in [0,1,2,np.nan]]