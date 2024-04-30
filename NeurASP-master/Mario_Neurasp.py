import os
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
import sys
sys.path.append('../../')
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms

# from network import Net, testNN
from RL.DQN_network_asp import DQNSolver_asp
from neurasp import NeurASP

start_time = time.time()

#############################
# Load the training and testing data
#############################
with open('/Users/maximvandecasteele/PycharmProjects/NeurASP/Object_detector/data_neurasp/tensors_train.pkl', 'rb') as f:
    # Load the data from the pickle file
    dataList = pickle.load(f)

with open('/Users/maximvandecasteele/PycharmProjects/NeurASP/Object_detector/data_neurasp/symbols_train.pkl.pkl',
          'rb') as f:
    # Load the data from the pickle file
    obsList = pickle.load(f)


#############################
# NeurASP program
#############################

dprogram = '''
nn(dqn(1,state),[0,1,2,3,4]). 

% you cannot perform NOOP, right, right B when enemy is just right of you.
:- dqn(1,state,1), cell(R1, C1, 1), cell(R2, C2, 3), R1 = R2, C2 <= C1 + 2, C2 > C1.
:- dqn(1,state,3), cell(R1, C1, 1), cell(R2, C2, 3), R1 = R2, C2 <= C1 + 2, C2 > C1.
:- dqn(1,state,0), cell(R1, C1, 1), cell(R2, C2, 3), R1 = R2, C2 <= C1 + 2, C2 > C1.

% You cannot run into a hole.
:- dqn(1,state,1), cell(R1, C1, 1), cell(13, C1 + 1, 4), not protected.
:- dqn(1,state,3), cell(R1, C1, 1), cell(13, C1 + 1, 4), not protected.
protected :- cell(R1, C1, 1), cell(R2, C1 + 1, 2), R2 > R1.

% you cannot walk through a platform on the right of you.
:- dqn(1,state,1), cell(R1, C1, 1), cell(R2, C2, 2), R1 = R2, C1 + 1 = C2.
:- dqn(1,state,3), cell(R1, C1, 1), cell(R2, C2, 2), R1 = R2, C1 + 1 = C2.

% you cannot walk through a pipe to the right of you
:- dqn(1,state,1), cell(R1, C1, 1), cell(R2, C2, 5), R1 = R2, C1 + 1 = C2.
:- dqn(1,state,3), cell(R1, C1, 1), cell(R2, C2, 5), R1 = R2, C1 + 1 = C2.

% you cannot walk through a wall to the right of you
:- dqn(1,state,1), cell(R1, C1, 1), cell(R2, C2, 6), R1 = R2, C1 + 1 = C2.
:- dqn(1,state,3), cell(R1, C1, 1), cell(R2, C2, 6), R1 = R2, C1 + 1 = C2.

% standing still against a platform / pipe / wall achieves nothing.
:- dqn(1,state,0), cell(R1, C1, 1), cell(R2, C2, 2), R1 = R2, C1 + 1 = C2.
:- dqn(1,state,0), cell(R1, C1, 1), cell(R2, C2, 5), R1 = R2, C1 + 1 = C2.
:- dqn(1,state,0), cell(R1, C1, 1), cell(R2, C2, 6), R1 = R2, C1 + 1 = C2.
'''

########
# Define nnMapping and optimizers, initialze NeurASP object
########

m = DQNSolver_asp((6, 15, 16), 5)
nnMapping = {'dqn': m}
optimizers = {'dqn': torch.optim.Adam(m.parameters(), lr=0.001)}
NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

########
# Start training and testing
########

print('Start training for 1 epoch...')
NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=1, smPickle=None, bar=True)

# device = torch.device('cpu')
# # check testing accuracy
# accuracy, singleAccuracy = testNN(model=m, testLoader=testLoader, device=device)
# # check training accuracy
# accuracyTrain, singleAccuracyTrain = testNN(model=m, testLoader=trainLoader, device=device)
# print(f'{accuracyTrain:0.2f}\t{accuracy:0.2f}')
# print('--- total time from beginning: %s seconds ---' % int(time.time() - start_time) )