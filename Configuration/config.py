import torch

config = {
        "detector_model_path": '/Users/maximvandecasteele/PycharmProjects/NeurASP/Object_detector/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
        "detector_label_path": '/Users/maximvandecasteele/PycharmProjects/NeurASP/Object_detector/models/data.yaml',
        "positions_asp": '/Users/maximvandecasteele/PycharmProjects/NeurASP/RL/asp/positions.lp',
        "show_asp": '/Users/maximvandecasteele/PycharmProjects/NeurASP/RL/asp/show.lp',
        # "constraints": '/Users/maximvandecasteele/PycharmProjects/NeurASP/RL/asp/game_rules.lp',
        "constraints": '/Users/maximvandecasteele/PycharmProjects/NeurASP/RL/asp/game_rules_balanced.lp',
        "show_constraints": '/Users/maximvandecasteele/PycharmProjects/NeurASP/RL/asp/show_constraints.lp',
        'game_rules': '/Users/maximvandecasteele/PycharmProjects/NeurASP/RL/asp/game_rules_balanced_for_advisor.lp',
        'show_airborne': '/Users/maximvandecasteele/PycharmProjects/NeurASP/RL/asp/show_airborne.lp',
    }

config_rgb = {
        'vis': False,
        'level': '1-1',
        'tensorboard': True,
        # Run name, used in Tensorboard
        'run_name': 'Test',
        'input_type': 'rgb',
        'inference_type': 'pure',

        'train': True,
        'max_exp_r': 1.0,
        'min_exp_r': 0.02,
        'epochs': 5000,
        'backup_epochs': 100,
        'save_good_model': -1,

        # where files will be stored to and loaded from
        'working_dir': 'Models_rgb_mac',
        'pretrained_weights': False,
        'model_name': "",
        'load_experience_replay': False,
        'save_experience_replay': False,
}

config_asp = {
        'vis': False,
        'level': '1-1',
        'tensorboard': True,
        # Run name, used in Tensorboard
        'run_name': 'Test',
        'input_type': 'asp',
        'inference_type': 'pure',

        'train': True,
        'max_exp_r': 1.0,
        'min_exp_r': 0.02,
        'epochs': 5000,
        'backup_epochs': 100,
        'save_good_model': -1,

        # where files will be stored to and loaded from
        'working_dir': 'Models_asp_mac',
        'pretrained_weights': False,
        'model_name': "",
        'load_experience_replay': False,
        'save_experience_replay': False,
}

config_player = {
    'vis': False,
    'level': '2-1',
    # asp or rgb
    'input_type': 'asp',
    'inference_type': 'pure',
    'train': True,
    'exp_r': 0.1,
    'num_runs': 5,
    'epochs': 100,
    'working_dir': '/Users/maximvandecasteele/PycharmProjects/NeurASP/Evaluation/asp_ddqn_advisor_final_slow_expl/run4/',
    # 4 werkte heel goed
    'model': '4900best_performer_dq2.pt',
    'pretrained_weights': True,
    'load_experience_replay': False,
    'save_experience_replay': False,
}

config_evaluation = {
    'vis': False,
    'level': '1-1',
    'tensorboard' : True,
    # asp or rgb
    'input_type': 'asp',
    'inference_type': 'pure',
    'train': False,
    'max_exp_r': 1.0,
    'min_exp_r': 0.02,
    'num_runs': 5,
    'epochs': 100,
    'working_dir': '/Users/maximvandecasteele/PycharmProjects/NeurASP/NeurASP/models/test/',
    'pretrained_weights': True,
    'load_experience_replay': False,
    'save_experience_replay': False,
}