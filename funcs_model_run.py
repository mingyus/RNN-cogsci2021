import pickle
import torch
from torch import nn
from nets import *
from funcs_data_preprocessing import *
from funcs_model import *


def load_net_data(input_type, output_type):
    str_input_output = '_input_' + input_type + '_output_' + output_type
    fileName = {
        'train': 'data_RNN_train_shuffleDF' + str_input_output + '.p',
        'validation': 'data_RNN_validation' + str_input_output + '.p',
        'test': 'data_RNN_test' + str_input_output + '.p'
    }
    data = dict()
    for setName in ['train', 'validation', 'test']:
        data_set = pickle.load(open('data/' + fileName[setName], 'rb'))
        data['inputs_'+setName] = torch.from_numpy(data_set['inputs']).float()
        data['target_output_'+setName] = torch.from_numpy(data_set['target_output'])
    return data


def training(modelName, data, dataInfo, hyper_parameters, device, rand_seed=0, ifSaveResults=True, epoch_stop=None):
    # set random seed
    torch.manual_seed(rand_seed)
    
    # initialize the model with dataInfo and hyper-parameters
    model = initialize_model(modelName, dataInfo, hyper_parameters)
    
    print(model)
    
    # model filename
    model_filename = get_model_filename(modelName, hyper_parameters)
    
    # extract hyper-parameter values for training
    n_epochs, lr_init, lr_adjust_type, batch_size = hyper_parameters['n_epochs'], hyper_parameters['lr_init'], hyper_parameters['lr_adjust_type'], hyper_parameters['batch_size'] 
    
    # define loss and optimizer
    criterion_train = nn.CrossEntropyLoss(ignore_index = -1, reduction='sum')
    criterion_test = nn.CrossEntropyLoss(ignore_index = -1, reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
            
    # training information
    trainingInfo = dict()
    trainingInfo['batch_size'] = batch_size
    trainingInfo['criterion'] = criterion_train, criterion_test
    trainingInfo['optimizer'] = optimizer
        
    # move variables to device
    model = model.to(device)
    
    # prepare data
    inputs = dict()
    target_output = dict()
    for setName in ['train', 'validation', 'test']:
        inputs[setName] = data['inputs_'+setName]
        target_output[setName] = data['target_output_'+setName].clone()
        target_output[setName][torch.isnan(target_output[setName])] = -1
        target_output[setName] = target_output[setName].long()
    
    # run training (and validation)
    loss_train, loss_validation = torch.zeros(n_epochs), torch.zeros(n_epochs)
    for i_epoch in range(n_epochs):
        # training
        loss_train[i_epoch] = evaluate(modelName, model, device, inputs['train'], target_output['train'], dataInfo, trainingInfo, 'train')
        
        # calculate validation loss
        loss_validation[i_epoch], neg_llh, _ = evaluate(modelName, model, device, inputs['validation'], target_output['validation'], dataInfo, trainingInfo, 'validation')
        
        # print training and validation loss
        if (i_epoch+1) % 10 == 0:
            print('Epoch: {}/{}.............'.format(i_epoch+1, n_epochs), end=' ', flush=True)
            print("Loss: {:.4f}".format(loss_train[i_epoch].item()), end=', ', flush=True)
            print("Loss validation: {:.4f}".format(loss_validation[i_epoch].item()), end=', ', flush=True)
        
        # early stopping, save model and test results at this stop point
        if (epoch_stop is not None) and (i_epoch == epoch_stop):
            torch.save(model.state_dict(), 'models/model_' + model_filename + '_epoch' + str(epoch_stop) + '.p')
            print('Model saved.')
            # run test
            loss_test, neg_llh_test, prob = evaluate(modelName, model, device, inputs['test'], target_output['test'], dataInfo, trainingInfo, 'test')
            pickle.dump({'llh_test': -neg_llh_test.numpy(), 'p_allChoices': prob.numpy()}, open('models/test_' + model_filename + '_epoch' + str(epoch_stop) + '.p', 'wb'))
            return
    
    # save results
    results = {'loss_train':loss_train, 'loss_validation':loss_validation, 'hyper_parameters':hyper_parameters}
    if ifSaveResults:
        torch.save(results, 'model_results/' + model_filename + '.p')
    
    # return the trained model and results
    return model, model_filename, results


def getBatchIndices(setName, datasize, batch_size):
    if setName == 'train': # shuffle
        indices_all = torch.randperm(datasize)
    elif setName in ['validation', 'test']:  # in order
        indices_all = torch.arange(datasize)
    indices = []
    NBatches = int(torch.ceil(torch.tensor(datasize/batch_size)))
    for iBatch in range(NBatches):
        if iBatch < NBatches - 1:
            indices.append(indices_all[iBatch*batch_size:(iBatch+1)*batch_size])
        else:
            indices.append(indices_all[iBatch*batch_size:])
    return NBatches, indices
        

def evaluate(modelName, model, device, inputs, target_output, dataInfo, trainingInfo, setName):
    # get training info
    batch_size, optimizer = trainingInfo['batch_size'], trainingInfo['optimizer']
    criterion_train, criterion_test = trainingInfo['criterion']
    
    # get mini-batches
    NBatches, indices = getBatchIndices(setName, inputs.shape[0], batch_size)
    
    # initialization for variables storing llh and loss
    loss_total = 0
    if setName != 'train':
        neg_llh = torch.zeros(target_output.shape)
        prob = torch.zeros(list(target_output.shape)+[dataInfo['output_size']])

    for iBatch in range(NBatches):
        indices_this = indices[iBatch]
        
        if setName == 'train':
            optimizer.zero_grad() # clears existing gradients from previous epoch
        
        x = inputs[indices_this, :, :].to(device)
        
        # evaluate model
        output, hidden = model(x)
        if setName == 'train':
            loss = criterion_train(output, target_output[indices_this, :].view(-1).to(device))
        else:
            prob[indices_this, :, :] = nn.functional.softmax(output, dim=1).detach().cpu().view(prob[indices_this,:,:].shape)
            neg_llh[indices_this, :] = criterion_test(output, target_output[indices_this, :].view(-1).to(device)).detach().cpu().view(neg_llh[indices_this, :].shape)
        
        if setName == 'train':
            loss.backward() # backpropagate and calculate gradients
            optimizer.step() # update the weights accordingly
            loss_total += float(loss)
        
    if setName == 'train':
        return loss_total/(torch.sum(target_output != -1)/dataInfo['num_output'])
    else:
        neg_llh[target_output == -1] = np.nan
        return torch.mean(neg_llh[~torch.isnan(neg_llh)]), neg_llh, prob