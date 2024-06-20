### How to train the reinforcement learning agent
Tested with python 3.9
First install requirements.txt

Download the semantic segmentation model from [here](https://drive.google.com/file/d/1JRdPggs5jTWAXKRXk6hVxzmP-KnOr8Hw/view?usp=sharing) and place it inside Segmentation_model

Then, you can run the mario_ddqn.py file, it has multiple options. The command to train on level 1-1 using semantic segmentation, and saving weights on 1_1_ssweights is:

    python .\mario_ddqn.py -wd 1_1_ssweights -it ss -t

If you want to visualize it, use -vis, we recommend only visualizing on inference.

Once it trained, if you want to see how the model behaves and plays, run:

    python .\mario_ddqn.py -wd 1_1_ssweights -it ss -pt -vis

You can also change the level with --level

    python .\mario_ddqn.py -wd 1_1_ssweights -it ss -t --level 2-1

For a full list of commands do:

    python .\mario_ddqn.py -h

## Table of Contents
### Configuration

Contains the various configuration dictionaries:   

Config contains the absolute paths to the necessary clingo files.  
config_rgb contains the settings for RGB training.   
config_asp contains the settings for symbolic tensor input training (ASP, Neurasp, Neurasp-Advisor)  
Config_player is used for running and playing trained models.  
Config_evaluation used purely for evaluation purposes.  

### Environment

Contains the necessary gym wrappers for mario gym environment creation.   

### Evaluation

Contains three files:

mario_ddqn_evaluation is used for evaluating ddqn performance (everything without neurasp)   
mario_neurasp_evaluation is used for neurasp evaluation.

results_baseline: jupyter notebook used to generate graphs, plots... (data analysis)

### NeurASP

Contains the NeurASP code provided by Yang (https://github.com/azreasoners/NeurASP)

In addition, Mario_Neurasp_post and Mario_Neurasp_pre were used for training the various models using NeurASP.    

### Object_detector

Contains the models, yaml and necessary components for YOLOv8 training. 
frame_generator_human can be used to play the game as a human and store frames while doing so. 
dataset_creator was used to create the training and test sets for NeurASP training. 

### RL (Most important)

This folder contains two subfolders:  

#### asp 
Contains the various clingo files, created for the purposes of this thesis. 

#### symbolic_components 
Contains the detector (responsible for detecting objects in a frame), positioner (responsible for converting the YOLO 
data into symbolic tensor format) and advisor (advises agent in certain situations).   

#### RL code
The other files all encompass the RL training aspect. DQNAgent, DQN_network_asp, DQN_network_vanilla, 
DQN_network_neurasp all contain the basic Q learning logic and network architectures for DQN learning. 

ASP_frame_checker was used to evaluate the ASP code. 

mario_ddqn and mario_ddqn_advisor contain the running scripts for DQN training. 

mario_ddqn_player and mario_ddqn_advisor_player can be used for running trained models. 