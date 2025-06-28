import os
from pathlib import Path
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


def setup_dmms_dirs(args):
    """Create directories for DMMS-based training runs."""
    dmms_root = Path(args.main_dir) / "results" / "dmms_runs"
    dmms_root.mkdir(parents=True, exist_ok=True)
    args.dmms_io_root = str(dmms_root)
    return args


def set_agent_parameters(args):
    agent = None
    device = args.device

    if args.agent == 'g2p2c':
        from agents.g2p2c.g2p2c import G2P2C
        from agents.g2p2c.parameters import set_args # G2P2C 에이전트의 파라미터 설정 함수

        # G2P2C 에이전트 기본 인자 설정 (필요시 기존 run_RL_agent.py 참조)
        args.aux_mode = getattr(args, 'aux_mode', 'dual') # 기본값 설정 가능
        args.use_planning = getattr(args, 'use_planning', 'yes') # 기본값 설정 가능

        args = set_args(args) # 에이전트 특화 인자 설정 (중요: 이 함수가 args를 많이 채움)

        # === 미세 조정을 위한 수정 시작 ===
        if args.fine_tune_from_checkpoint: # fine_tune_from_checkpoint 와 같은 인자를 추가하여 제어
            print(f"INFO: Attempting to load G2P2C agent from checkpoint for fine-tuning.")
            # 사용자가 명시한 사전 훈련된 모델 경로
            # MAIN_PATH는 decouple 또는 다른 방식으로 정의되어 있다고 가정
            actor_checkpoint_path = os.path.join(MAIN_PATH, 'results', 'test', 'checkpoints', f'episode_{args.fine_tune_from_checkpoint}_Actor.pth')
            critic_checkpoint_path = os.path.join(MAIN_PATH, 'results', 'test', 'checkpoints', f'episode_{args.fine_tune_from_checkpoint}_Critic.pth')

            if not os.path.exists(actor_checkpoint_path):
                raise FileNotFoundError(f"Actor checkpoint not found: {actor_checkpoint_path}")
            if not os.path.exists(critic_checkpoint_path):
                raise FileNotFoundError(f"Critic checkpoint not found: {critic_checkpoint_path}")

            print(f"INFO: Loading Actor from: {actor_checkpoint_path}")
            print(f"INFO: Loading Critic from: {critic_checkpoint_path}")

            # G2P2C 생성자를 호출하여 모델 로드
            # load=True, path1=actor_path, path2=critic_path
            agent = G2P2C(args, device, load=True, path1=actor_checkpoint_path, path2=critic_checkpoint_path)
            print(f"INFO: Successfully loaded G2P2C agent episode {args.fine_tune_from_checkpoint} for fine-tuning.")
        else:
            # 기존 방식: 새 에이전트 훈련 (load=False)
            print(f"INFO: Initializing new G2P2C agent for training from scratch.")
            agent = G2P2C(args, device, load=False, path1='', path2='')
        # === 미세 조정을 위한 수정 끝 ===

        # 실험 결과 폴더 설정 (setup_folders는 agent 생성 *후* 또는 args.experiment_dir이 먼저 설정되어야 함)
        # setup_folders가 args.experiment_dir을 설정하므로, agent 생성 전에 호출하거나,
        # agent가 이 경로를 사용하지 않는다면 순서는 크게 상관없음.
        # 단, args가 set_args에 의해 변경된 후 setup_folders가 호출되는 것이 안전할 수 있음.
        args = setup_folders(args) # setup_folders는 args.experiment_dir을 설정

        # 코드 복사 (이 부분은 그대로 유지하거나 필요에 따라 조정)
        dst = os.path.join(args.experiment_dir, 'code') # MAIN_PATH 대신 args.experiment_dir 사용 권장
        copy_folder(os.path.join(MAIN_PATH, 'agents', 'g2p2c'), dst)
        # parameters.py도 해당 위치에 복사되었는지 확인 (g2p2c_agent_api.py 로직 참고)
        # shutil.copy(os.path.join(MAIN_PATH, 'agents', 'g2p2c', 'parameters.py'), dst) # 명시적 복사

    return args, agent, device


def main():
    args = Options().parse()
    args, agent, device = set_agent_parameters(args)

    if args.sim == 'dmms':
        args = setup_dmms_dirs(args)

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
