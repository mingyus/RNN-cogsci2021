This repository contains data and code that accompany the following paper:

Song, M., Niv, Y., & Cai, M. (2021). [Using Recurrent Neural Networks to Understand Human Reward Learning](https://escholarship.org/uc/item/3wj7w4x7). *In Proceedings of the Annual Meeting of the Cognitive Science Society* (Vol. 43, No. 43).

## Table of Contents

* `data/`: contains the original human data, and the game indices for training, validation and test set split.
* `cog_model_pAllChoices.csv`: trial-by-trial human data (as above) with the fits of the best cognitive model (model log likelihood, and the predicted probability for all possible choices)
* `nets/`: code for RNN models
* `models/`: RNN training results
* `model_simulation/`: simulation results of the trained RNNs
* `*.py` and `*.ipynb`: analysis code


## Data preprocessing

**Original dataset:**
In the original dataset, 102 participants each completed 18 games in total (3 repetitions of 6 game types each, in random order). All data are stored in `data/data.csv`.

**Data split:**
For network training, we split data into training, validation and test sets. The training set consisted of 16 games from each participant, and the validation and testsets each  consisted of 1 game per participant. The game type and game index were balanced (to the extent possible) in each set to reduce potential bias or order effect.

**Training data augmentation:**
The training set was augmented to 1296 times of the original size (note there was a typo in the paper which wrongly said "1024 times"). Data augmentation was done through shuffling the dimensions and features in the data (permutation of 3 dimensions and 3 features within each dimension results in 3!x3!x3!x3! = 1296). No shuffling was done for validation or test sets.

**Data preprocessing:**
As shown in Figure 3, the input of the network consisted of the following variables: a game-start indicator (1 if it was the first trial, 0 otherwise), the game type, the participant's choice, the configured stimulus, and the outcome. The last three variables were taken from the previous trial and were zero on the first trial of a game. During data preprocessing, all input variables were one-hot encoded, and concatenated into a binary input vector.

For RNN w/ embedding model, the participant's ID was also encoded as a one-hot vector.

**Code:** 
To run data preprocessing, execute `data_preprocessing.ipynb` (implementation code in `funcs_data_preprocessing.py`). Please expect it to take some time (i.e., hours) to complete, and the preprocessed training data to take up a large disk space (i.e., tens of GB).

## Network training

**Training setup:**
We used the Adam optimizer in PyTorch, and the cross-entropy loss (i.e., log likelihood of participant's choice) as the cost function.

**Network selection and evaluation:**
The weights of the networks were trained on the training set, the values of the hyper-parameters (learning rate, batch size, recurrent layer size, embedding layer size, and early stopping epoch) were selected based on the validation set, and all results reported were evaluated on the test set using the best fit network. 

**Code:**
To train the model, provide hyper-parameter values in `run_model.ipynb` and execute the code (see `funcs_model.py` and `funcs_model_run.py` for implementation code).

**Results:**
The best training results are saved in `models/` (file names contain the model name and the selected hyper-parameter values).

* `models/model_*.p`: the trained models (at the early stopping epochs);
* `models/test_*.p`: the prediction on the test set using the trained models.

## Analyses and results

### Analysis code
Code for data and model analyses (including helper functions) can be found in the following files: `info_task_exp.py`, `info_model.py`, `funcs_data_analysis.py`, `funcs_learning_curve.py`, and `funcs_model_analysis.py`.

### To reproduce figures in the paper

* Figure 2C,F: `model_simulation.ipynb`
* Figure 4: `model_analysis_RNN_vs_cog.ipynb`
* Figure 5: `model_analysis_RNN_vs_cog.ipynb`
* Figure 6: `model_analysis_embedding.ipynb` and `model_simulation.ipynb`

### Supplementary data size analysis

In the paper, we conducted an analysis on how data size affected the fitting of RNN models  (Figure 7). This analysis involved the following steps:

1. Generate simulated data using the best cognitive model with various sizes (1 to 1296 times of the original dataset);
2. Augment the simulated data of size 1 by shuffling dimensions (shuffleD), features (shuffleF) and both (shuffleDF);
3. Fit RNN w/ embedding model on the above "fake" data sets, and compare the predicted likelihood with the ground-truth likelihood (based on the generative model).

The reproduction of this analysis requires the fake data (huge in size) and corresponding best-fit RNN models, which are hard to share due to space limit. However, we note that steps 2 and 3 above can be achieved using the same code for analyses on human data (as described in [Data preprocessing](#data-preprocessing) and [Network training](#network-training)).