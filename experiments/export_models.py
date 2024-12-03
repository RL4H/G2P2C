# todo => resume a stopped experiment from last saved checkpoint.
import os
import sys
import torch
import json
import shutil
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
    args.experiment_dir = LOG_DIR
    set_logger(LOG_DIR)
    return args

def set_agent_parameters(args):
    agent = None
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = args.device

    from agents.td3.td3 import TD3
    from agents.td3.parameters import set_args
    args = set_args(args)
    args = setup_folders(args)
    weights = MAIN_PATH+'/results/'+args.restart+'/checkpoints/'
    agent = TD3(args, device, True, weights+'episode_195_policy_net.pth', weights+'episode_195_value_net.pth')
    LOG_DIR = MAIN_PATH + "/results/" + args.folder_id

    # torch.save(agent.policy.Actor.state_dict(), weights+'Trained_Actor.pth')
    # torch.save(agent.policy.Critic.state_dict(), weights+'Trained_Critic.pth')

    torch.save(agent.td3.policy_net.state_dict(), LOG_DIR + '/checkpoints/'+'Trained_Actor.pth')
    torch.save(agent.td3.value_net1.state_dict(), LOG_DIR + '/checkpoints/'+'Trained_Critic.pth')

    # Actor_scripted = torch.jit.script(agent.policy.Actor)  # Export to TorchScript
    # Critic_scripted = torch.jit.script(agent.policy.Actor)  # Export to TorchScript
    # Actor_scripted.save(weights+'actor_scripted.pt')
    # Critic_scripted.save(weights+'critic_scripted.pt')
    # model = torch.jit.load(weights+'actor_scripted.pt')
    # model.eval()
    # print(next(model.parameters()).device)

    return args, agent, device, weights

def main():
    print('\nExperiment Starting...')

    args = Options().parse()
    args, agent, device, weights = set_agent_parameters(args)
    print('successs')

if __name__ == '__main__':
    main()
