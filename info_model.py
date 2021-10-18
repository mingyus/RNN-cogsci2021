# cog model
cog_model = 'inferSerialHypoTesting_CountingValueBasedSwitchNoResetDecayThresonTestSelectMoreNoCost_FlexibleHypoAvg'

# RNN
net_label = {'LSTM': 'RNN', 'LSTM_embed': 'RNN w/ embedding'}
input_type = {'LSTM': 'as', 'LSTM_embed': 'as_iWorkerOnehot'}
output_type = {'LSTM': 'joint', 'LSTM_embed': 'joint'}

winning_model = {
    'LSTM': {
        'hyper_parameters': {'hidden_size': 50, 'n_epochs': 1000, 'lr_init': 0.001, 'lr_adjust_type': 'Adam', 'batch_size': 10000},
        'epoch': 28
    },
    'LSTM_embed': {
        'hyper_parameters': {'hidden_size': 50, 'n_epochs': 200, 'lr_init': 0.001, 'lr_adjust_type': 'Adam', 'batch_size': 10000, 'embed_size': {'subjID': 3}},
        'epoch': 23
    }
}