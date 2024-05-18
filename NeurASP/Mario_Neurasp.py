import os
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
import sys
sys.path.append('../../')
import time
import pickle
import torch

from RL.DQN_network_asp import DQNSolver_asp
from neurasp import NeurASP
from RL.DQN_network_asp import testNN

start_time = time.time()

if torch.backends.mps.is_available():
    # mps_device = torch.device(self.device)
    print("Using mps device.")
    device = 'cpu'
elif torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(1)
    print("Using CUDA device:", device_name)
    device = 'cuda:1'
else:
    print("CUDA is not available")
    device = 'cpu'


#############################
# Load the training and testing data
#############################
with open('/Object_detector/data_neurasp/balanced_dataset/tensors_train.pkl', 'rb') as f:
    # Load the data from the pickle file
    dataList = pickle.load(f)

with open('/Object_detector/data_neurasp/balanced_dataset/symbols_train.pkl',
          'rb') as f:
    # Load the data from the pickle file
    obsList = pickle.load(f)

with open('/Object_detector/data_neurasp/balanced_dataset/tensors_test.pkl', 'rb') as f:
    # Load the data from the pickle file
    tensors_test = pickle.load(f)

with open('/Object_detector/data_neurasp/balanced_dataset/symbols_test_SM.pkl',
          'rb') as f:
    # Load the data from the pickle file
    symbols_test = pickle.load(f)


#############################
# NeurASP program
#############################

# dprogram1 = '''
# nn(dqn(1,state),[0,1,2,3,4]).
#
# :- dqn(0,state,1), cell(R1, C1, 1), cell(R2, C2, 3), R1 = R2, C2 <= C1 + 2, C2 > C1.
# :- dqn(0,state,3), cell(R1, C1, 1), cell(R2, C2, 3), R1 = R2, C2 <= C1 + 2, C2 > C1.
# :- dqn(0,state,0), cell(R1, C1, 1), cell(R2, C2, 3), R1 = R2, C2 <= C1 + 2, C2 > C1.
#
# :- dqn(0,state,1), cell(R1, C1, 1), cell(13, C1 + 1, 4), not protected.
# :- dqn(0,state,3), cell(R1, C1, 1), cell(13, C1 + 1, 4), not protected.
# protected :- cell(R1, C1, 1), cell(R2, C1 + 1, 2), R2 > R1.
#
# :- dqn(0,state,1), cell(R1, C1, 1), cell(R2, C2, 2), R1 = R2, C1 + 1 = C2.
# :- dqn(0,state,3), cell(R1, C1, 1), cell(R2, C2, 2), R1 = R2, C1 + 1 = C2.
#
# :- dqn(0,state,1), cell(R1, C1, 1), cell(R2, C2, 5), R1 = R2, C1 + 1 = C2.
# :- dqn(0,state,3), cell(R1, C1, 1), cell(R2, C2, 5), R1 = R2, C1 + 1 = C2.
#
# :- dqn(0,state,1), cell(R1, C1, 1), cell(R2, C2, 6), R1 = R2, C1 + 1 = C2.
# :- dqn(0,state,3), cell(R1, C1, 1), cell(R2, C2, 6), R1 = R2, C1 + 1 = C2.
#
# :- dqn(0,state,0), cell(R1, C1, 1), cell(R2, C2, 2), R1 = R2, C1 + 1 = C2.
# :- dqn(0,state,0), cell(R1, C1, 1), cell(R2, C2, 5), R1 = R2, C1 + 1 = C2.
# :- dqn(0,state,0), cell(R1, C1, 1), cell(R2, C2, 6), R1 = R2, C1 + 1 = C2.
#
# '''


dprogram = '''
nn(dqn(1,state),[0,1,2,3,4]). 

protected :- cell(R1, C1, 1), cell(R2, C1 + 1, 2), R2 > R1.
jump :- cell(R1, C1, 1), cell(R2, C2, 3), R1 = R2, C2 <= C1 + 2, C2 > C1.
jump :- cell(R1, C1, 1), cell(13, C1 + 1, 4), not protected.
jump :- cell(R1, C1, 1), cell(R2, C2, 2), R1 = R2, C1 + 1 = C2.
jump :- cell(R1, C1, 1), cell(R2, C2, 5), R1 = R2, C2 >= C1 + 1, C2 <= C1 + 2. 
jump :- cell(R1, C1, 1), cell(R2, C2, 6), R1 = R2, C1 + 1 = C2.
jump :- cell(R1, C1, 1), not cell(R1 + 1, C1, _). 

:- dqn(0,state,1), jump. 

:- dqn(0,state,2), not jump.

:- dqn(0,state,0).
:- dqn(0,state,3). 
:- dqn(0,state,4).
'''

########
# Define nnMapping and optimizers, initialze NeurASP object
########

# m = DQNSolver_asp((6, 15, 16), 5).to(device)
# nnMapping = {'dqn': m}
# optimizers = {'dqn': torch.optim.Adam(m.parameters(), lr=0.00001)}
# NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

########
# Start training and testing
########

print('Start training for 1 epoch...')

for run in range(2, 6):
    start_time = time.time()
    m = DQNSolver_asp((6, 15, 16), 5).to(device)
    nnMapping = {'dqn': m}
    optimizers = {'dqn': torch.optim.Adam(m.parameters(), lr=0.00001)}
    NeurASPobj = NeurASP(dprogram, nnMapping, optimizers, run=run)
    NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=30, smPickle=None, bar=True)
    print('--- total time from beginning: %s seconds ---' % int(time.time() - start_time))

# # check testing accuracy
# accuracy = testNN(model=m, tensors_test=tensors_test, symbols_test=symbols_test, device=device)
# print(f'Accuracy on test set: {accuracy}')
# # check training accuracy
# print('--- total time from beginning: %s seconds ---' % int(time.time() - start_time))