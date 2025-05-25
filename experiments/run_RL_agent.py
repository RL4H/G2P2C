import os
import sys
import torch
import json
import shutil
import random
import numpy as np
import argparse
from pprint import pprint
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)
from utils.core import set_logger, get_patient_env
from utils.options import Options

import warnings
warnings.simplefilter('ignore', Warning)


def setup_folders(args):
    # create the folder which will save experiment data.
    LOG_DIR = MAIN_PATH + "/results/" + args.folder_id
    CHECK_FOLDER = os.path.isdir(LOG_DIR)
    if CHECK_FOLDER:
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR + '/checkpoints')
    os.makedirs(LOG_DIR + '/training/data')
    os.makedirs(LOG_DIR + '/training/plots')
    os.makedirs(LOG_DIR + '/testing/data')
    os.makedirs(LOG_DIR + '/testing/plots')
    os.makedirs(LOG_DIR + '/code')
    args.experiment_dir = LOG_DIR
    set_logger(LOG_DIR)
    return args


def copy_folder(src, dst):
    for folders, subfolders, filenames in os.walk(src):
        for filename in filenames:
            shutil.copy(os.path.join(folders, filename), dst)


def set_agent_parameters(args):
    agent = None
    device = args.device
    if args.agent == 'ppo':
        from agents.ppo.ppo import PPO
        from agents.ppo.parameters import set_args
        args.aux_mode = 'off'
        args.use_planning = 'no'
        args = set_args(args)
        args = setup_folders(args)
        weights = MAIN_PATH+'/results/ppo_lstm_12_testing/checkpoints/'
        #agent = PPO(args, device, True, weights+'episode_379_Actor.pth', weights+'episode_379_Critic.pth')
        agent = PPO(args, device, False, '', '')
        dst = MAIN_PATH + '/results/' + args.folder_id + '/code'
        copy_folder(MAIN_PATH + '/agents/ppo', dst)

    elif args.agent == 'a2c':
        from agents.a2c.a2c import A2C
        from agents.a2c.parameters import set_args
        args.aux_mode = 'off'
        args.use_planning = 'no'
        args = set_args(args)
        args = setup_folders(args)
        weights = MAIN_PATH+'/results/ppo_lstm_12_testing/checkpoints/'
        #agent = PPO(args, device, True, weights+'episode_379_Actor.pth', weights+'episode_379_Critic.pth')
        agent = A2C(args, device, False, '', '')
        dst = MAIN_PATH + '/results/' + args.folder_id + '/code'
        copy_folder(MAIN_PATH + '/agents/a2c', dst)

    elif args.agent == 'sac':
        from agents.sac.sac import SAC
        from agents.sac.parameters import set_args
        args.aux_mode = 'off'
        args.use_planning = 'no'
        args = set_args(args)
        args = setup_folders(args)
        weights = MAIN_PATH+'/results/ppo_lstm_12_testing/checkpoints/'
        #agent = PPO(args, device, True, weights+'episode_379_Actor.pth', weights+'episode_379_Critic.pth')
        agent = SAC(args, device, False, '', '')
        dst = MAIN_PATH + '/results/' + args.folder_id + '/code'
        copy_folder(MAIN_PATH + '/agents/sac', dst)

    elif args.agent == 'g2p2c':
        from agents.g2p2c.g2p2c import G2P2C
        from agents.g2p2c.parameters import set_args
        args.aux_mode = 'dual'
        args.use_planning = 'yes'
        args = set_args(args)
        args = setup_folders(args)
        weights = MAIN_PATH+'/results/ppo_lstm_12_testing/checkpoints/'
        #agent = PPO(args, device, True, weights+'episode_379_Actor.pth', weights+'episode_379_Critic.pth')
        agent = G2P2C(args, device, False, '', '')
        dst = MAIN_PATH + '/results/' + args.folder_id + '/code'
        copy_folder(MAIN_PATH + '/agents/g2p2c', dst)

    elif args.agent == 'ppo_aux':
        from agents.ppo_aux_plan.ppo import PPO
        from agents.ppo_aux_plan.parameters import set_args
        args.aux_mode = 'dual'
        args.use_planning = 'no'
        args = set_args(args)
        args = setup_folders(args)
        weights = MAIN_PATH+'/results/ppo_lstm_12_testing/checkpoints/'
        #agent = PPO(args, device, True, weights+'episode_379_Actor.pth', weights+'episode_379_Critic.pth')
        agent = PPO(args, device, False, '', '')
        dst = MAIN_PATH + '/results/' + args.folder_id + '/code'
        copy_folder(MAIN_PATH + '/agents/ppo_aux_plan', dst)

    elif args.agent == 'ddpg':
        from agents.ddpg.ddpg import DDPG
        from agents.ddpg.parameters import set_args
        args.aux_mode = 'off'
        args.use_planning = 'no'
        args = set_args(args)
        args = setup_folders(args)
        weights = MAIN_PATH+'/results/ppo_lstm_12_testing/checkpoints/'
        #agent = PPO(args, device, True, weights+'episode_379_Actor.pth', weights+'episode_379_Critic.pth')
        agent = DDPG(args, device, False, '', '')
        dst = MAIN_PATH + '/results/' + args.folder_id + '/code'
        copy_folder(MAIN_PATH + '/agents/ddpg', dst)

    elif args.agent == 'dpg':
        from agents.dpg.dpg import DPG
        from agents.dpg.parameters import set_args
        args.aux_mode = 'off'
        args.use_planning = 'no'
        args = set_args(args)
        args = setup_folders(args)
        weights = MAIN_PATH+'/results/ppo_lstm_12_testing/checkpoints/'
        #agent = PPO(args, device, True, weights+'episode_379_Actor.pth', weights+'episode_379_Critic.pth')
        agent = DPG(args, device, False, '', '')
        dst = MAIN_PATH + '/results/' + args.folder_id + '/code'
        copy_folder(MAIN_PATH + '/agents/dpg', dst)

    elif args.agent == 'td3':
        from agents.td3.td3 import TD3
        from agents.td3.parameters import set_args
        args.aux_mode = 'off'
        args.use_planning = 'no'
        args = set_args(args)
        args = setup_folders(args)
        weights = MAIN_PATH + '/results/ppo_lstm_12_testing/checkpoints/'
        # agent = PPO(args, device, True, weights+'episode_379_Actor.pth', weights+'episode_379_Critic.pth')
        agent = TD3(args, device, False, '', '')
        dst = MAIN_PATH + '/results/' + args.folder_id + '/code'
        copy_folder(MAIN_PATH + '/agents/td3', dst)
        print('Please select an agent for the experiment. Hint: a2c, a3c, sac, ppo, ppo_v2')
    return args, agent, device


def main():
    args = Options().parse()
    args, agent, device = set_agent_parameters(args)

    with open(args.experiment_dir + '/args.json', 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    if args.verbose:
        print('\nExperiment Starting...')
        print("\nOptions =================>")
        print(vars(args))  # pprint(vars(args), indent=4)
        print('\nDevice which the program run on:', device)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    patients, env_ids = get_patient_env()  # note: left here so that type of subject can be selected.
    agent.run(args, patients, env_ids, args.seed)


if __name__ == '__main__':
    main()
