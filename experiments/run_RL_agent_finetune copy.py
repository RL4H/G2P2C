# experiments/run_RL_agent_finetune.py

import os
from pathlib import Path
import sys
import torch
import json
import shutil
import random
import numpy as np
import traceback
import argparse # argparse는 Options() 내부에서 사용될 것이므로 직접 임포트 필요 없을 수 있음
from pprint import pprint
from decouple import config

# --- FastAPI 및 Uvicorn 통합을 위한 추가 임포트 ---
import uvicorn
import threading
import time
# ----------------------------------------------------

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from utils.core import set_logger, get_patient_env
from utils.options import Options # Options 클래스에 fine_tune_from_checkpoint 인자 추가 확인

# --- Sim_CLI.main_finetune 모듈에서 필요한 구성요소 임포트 ---
from Sim_CLI.main_finetune import app as fastapi_app_finetune # FastAPI 앱 인스턴스
from Sim_CLI.main_finetune import setup_app_state_for_finetuning # 앱 상태 설정 함수
# ---------------------------------------------------------

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

# set_agent_parameters 함수는 이전 검토에서 수정한 버전 사용
def set_agent_parameters(args):
    agent = None
    device = args.device

    if args.agent == 'g2p2c':
        from agents.g2p2c.g2p2c import G2P2C
        from agents.g2p2c.parameters import set_args

        args.aux_mode = getattr(args, 'aux_mode', 'dual')
        args.use_planning = getattr(args, 'use_planning', 'yes') 
        args = set_args(args)
        
        # setup_folders를 에이전트 생성 전에 호출하여 args.experiment_dir을 먼저 설정
        args = setup_folders(args)

        if args.fine_tune_from_checkpoint and args.fine_tune_from_checkpoint > 0: # 0보다 큰 값일 때만 로드
            print(f"INFO: Attempting to load G2P2C agent from checkpoint for fine-tuning (Episode {args.fine_tune_from_checkpoint}).")
            actor_checkpoint_path = os.path.join(MAIN_PATH, 'results', 'test', 'checkpoints', f'episode_{args.fine_tune_from_checkpoint}_Actor.pth')
            critic_checkpoint_path = os.path.join(MAIN_PATH, 'results', 'test', 'checkpoints', f'episode_{args.fine_tune_from_checkpoint}_Critic.pth')

            if not os.path.exists(actor_checkpoint_path):
                raise FileNotFoundError(f"Actor checkpoint not found: {actor_checkpoint_path}")
            if not os.path.exists(critic_checkpoint_path):
                raise FileNotFoundError(f"Critic checkpoint not found: {critic_checkpoint_path}")

            print(f"INFO: Loading Actor from: {actor_checkpoint_path}")
            print(f"INFO: Loading Critic from: {critic_checkpoint_path}")
            
            agent = G2P2C(args, device, load=True, path1=actor_checkpoint_path, path2=critic_checkpoint_path)
            print(f"INFO: Successfully loaded G2P2C agent episode {args.fine_tune_from_checkpoint} for fine-tuning.")
        else:
            print(f"INFO: Initializing new G2P2C agent for training from scratch.")
            agent = G2P2C(args, device, load=False, path1='', path2='')
        
        dst = os.path.join(args.experiment_dir, 'code', 'g2p2c_agent_code') # 에이전트 코드 복사 위치 명확화
        os.makedirs(dst, exist_ok=True)
        copy_folder(os.path.join(MAIN_PATH, 'agents', 'g2p2c'), dst)
        # parameters.py 등도 명시적으로 복사
        try:
            shutil.copy2(os.path.join(MAIN_PATH, 'agents', 'g2p2c', 'parameters.py'), os.path.join(dst, 'parameters.py'))
        except FileNotFoundError:
            print(f"WARNING: agents/g2p2c/parameters.py not found for copying.")
            
    else:
        raise ValueError(f"Unsupported agent type: {args.agent}")

    return args, agent, device

# --- Uvicorn 서버를 백그라운드 스레드에서 실행하기 위한 클래스/함수 ---
class UvicornServer(threading.Thread):
    def __init__(self, app, host="127.0.0.1", port=5000, log_level="info"):
        super().__init__(daemon=True) # 데몬 스레드로 설정하여 메인 스레드 종료 시 함께 종료
        self.config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
        self.server = uvicorn.Server(self.config)
        self._should_stop_event = threading.Event() # 명시적 종료를 위한 이벤트 (현재는 사용 안함, daemon으로 처리)

    def run(self):
        print(f"INFO: [UvicornThread] Starting Uvicorn server on http://{self.config.host}:{self.config.port}")
        try:
            self.server.run() # uvicorn.Server.run()은 내부적으로 asyncio.run(self.serve()) 호출
        except Exception as e:
            print(f"ERROR: [UvicornThread] Uvicorn server crashed: {e}")


    def stop(self): # 명시적 종료 시도 (주의: uvicorn.Server.run()은 외부에서 직접 중단시키기 어려울 수 있음)
        if self.server.started:
            print("INFO: [UvicornThread] Attempting to stop Uvicorn server...")
            self.server.should_exit = True 
            # self.server.force_exit = True # 강제 종료 (필요시)
            # uvicorn 서버가 should_exit를 얼마나 자주 확인하는지에 따라 즉시 종료되지 않을 수 있음
            # 일반적으로 Ctrl+C (KeyboardInterrupt)가 uvicorn에 의해 더 잘 처리됨
            # 데몬 스레드이므로 메인 프로그램 종료 시 자동으로 종료되는 것에 의존하는 것이 더 간단할 수 있음


def main():
    args = Options().parse() # Options에 fine_tune_from_checkpoint 인자 등이 추가되어 있어야 함
                           # 예: self.parser.add_argument('--fine_tune_from_checkpoint', type=int, default=0, help='Episode number for fine-tuning. 0 to train from scratch.')
                           # 예: self.parser.add_argument('--start_finetune_session_at_episode', type=int, default=1, help='Episode number for this fine-tuning session logging.')


    # 1. 에이전트 로드 (또는 새로 생성)
    print("--- Initializing Agent ---")
    args, agent, device = set_agent_parameters(args)
    if agent is None:
        print("CRITICAL_ERROR: Agent could not be initialized. Exiting.")
        return
    print(f"--- Agent Initialized (Device: {device}) ---")
    print(f"--- Agent object ID in run_RL_agent_finetune: {id(agent)} ---")


    # 2. FastAPI 앱 상태 설정 (로드된 에이전트 주입)
    # experiment_dir은 setup_folders에서 args에 설정됨
    # initial_episode_number는 이 미세조정 '세션'의 에피소드 시작 번호를 의미 (예: 항상 1부터 시작)
    # 또는 args에 --start_finetune_session_at_episode 같은 인자를 추가하여 제어 가능
    current_finetune_session_start_episode = getattr(args, 'start_finetune_session_at_episode', 1)
    print(f"--- Setting up FastAPI app state for fine-tuning (Session Episode Start: {current_finetune_session_start_episode}) ---")
    setup_app_state_for_finetuning(
        agent_instance=agent, 
        agent_args_for_statespace=agent.args, # G2P2C 에이전트가 자신의 args를 가지고 있다고 가정
        initial_episode_number=current_finetune_session_start_episode,
        experiment_dir=Path(args.experiment_dir) # Path 객체로 전달
    )
    print("--- FastAPI app state setup complete ---")

    # 3. Uvicorn 서버 백그라운드 스레드에서 시작
    # Sim_CLI.main_finetune에서 정의한 fastapi_app_finetune 사용
    # DMMS.R JS 플러그인이 접속할 주소/포트. args 또는 기본값 사용.
    server_host = getattr(args, 'dmms_server_host', "127.0.0.1") 
    server_port = getattr(args, 'dmms_server_port', 5000) # utils.options에 추가 필요할 수 있음
    
    print(f"--- Starting FastAPI server in background thread on http://{server_host}:{server_port} ---")
    uvicorn_server_thread = UvicornServer(app=fastapi_app_finetune, host=server_host, port=server_port)
    uvicorn_server_thread.start()

    # 서버가 시작될 시간을 잠시 줍니다 (필수적이지는 않으나, 바로 다음 DMMS.R 실행 시 안정적일 수 있음)
    print("INFO: Waiting for Uvicorn server to start (approx 2-5 seconds)...")
    time.sleep(5) # 실제로는 상태 체크 루프가 더 좋지만, 간단하게 sleep 사용
    print("--- FastAPI server should be running ---")


    # 4. DMMS.R 연동을 위한 디렉토리 설정 (args.sim == 'dmms' 일 때)
    if args.sim == 'dmms':
        print("--- Setting up DMMS directories ---")
        args = setup_dmms_dirs(args) # dmms_io_root를 세션별로 설정
        print("--- DMMS directories setup complete ---")


    # 5. args.json 저장 (모든 인자가 확정된 후)
    # args.experiment_dir은 setup_folders에서 설정됨
    args_json_path = os.path.join(args.experiment_dir, 'args_finetune.json') # 파일명 구분
    print(f"--- Saving arguments to {args_json_path} ---")
    try:
        with open(args_json_path, 'w') as fp:
            # args 객체를 vars()로 풀 때, Path 객체 등 직렬화 불가능한 타입이 있다면 str으로 변환 필요
            serializable_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            json.dump(serializable_args, fp, indent=4)
        print("--- Arguments saved ---")
    except Exception as e:
        print(f"ERROR: Could not save args.json: {e}")


    if args.verbose:
        print('\n--- Experiment Starting (Fine-tuning) ---')
        print("\nFinal Options (before agent.run) =================>")
        # Path 객체 때문에 직접 print(vars(args))가 길게 나올 수 있음. pprint가 더 나을 수 있음.
        try : 
            {
            pprint(vars(args), indent=2, width=120)
        } 
        except Exception:
            print(vars(args))

        print(f'\nDevice which the program run on: {device}')

    # 6. 랜덤 시드 설정
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"--- Random seeds set (Seed: {args.seed}) ---")

    # 7. 환자 환경 가져오기 (이 부분은 args.sim 타입에 따라 내부 동작이 달라질 수 있음)
    # DMMS.R의 경우, 실제 환자 환경은 DMMS.R exe와 cfg 파일로 정의됨.
    # get_patient_env()는 Simglucose에서는 환자 객체 리스트를 반환했을 수 있으나,
    # DMMS.R 모드에서는 다른 용도(예: 반복 실행할 환자 cfg 파일 목록 등)로 사용되거나 단순히 빈 리스트를 반환할 수 있음.
    # agent.run() 내부에서 args.dmms_exe, args.dmms_cfg 등을 사용하여 DmmsEnv를 초기화할 것임.
    print("--- Getting patient environment (Note: For DMMS, this might be a placeholder or list of configs) ---")
    patients, env_ids = get_patient_env() 
    print(f"--- Patient env info: patients={patients}, env_ids={env_ids} ---")

    # 8. 에이전트 미세 조정 훈련 루프 실행
    print(f"--- Starting agent fine-tuning run (agent.run method) ---")
    try:
        agent.run(args, patients, env_ids, args.seed) # 이 함수가 실제 훈련/테스트 루프를 포함
    except KeyboardInterrupt:
        print("\nINFO: KeyboardInterrupt received. Shutting down fine-tuning process...")
    except Exception as e:
        print(f"CRITICAL_ERROR: Exception during agent.run(): {e}")
        traceback.print_exc()
    finally:
        print("--- Agent run finished or interrupted ---")
        # 9. Uvicorn 서버 종료 시도 (메인 스레드 종료 시 데몬 스레드는 자동 종료되지만, 명시적 시도)
        # uvicorn_server_thread.stop() # UvicornServer 클래스에 구현된 stop 메소드 (주의사항 참고)
        # 현재 UvicornServer.stop()은 안정적인 종료를 보장하기 어려울 수 있음.
        # Ctrl+C로 메인 스크립트를 종료하면 데몬 스레드도 함께 종료됨.
        print("INFO: Main fine-tuning script finished. Uvicorn server (daemon thread) will exit with the main program.")

if __name__ == '__main__':
    # --- .env 파일 로드 (decouple 라이브러리가 자동으로 처리) ---
    # 예를 들어, 프로젝트 루트에 .env 파일이 있고, MAIN_PATH="C:/Users/user/Desktop/G2P2C" 와 같이 정의
    # 이 스크립트가 experiments 폴더에 있으므로, .env 파일은 프로젝트 루트에 위치해야 함.
    # (decouple은 실행 위치 기준으로 .env를 찾거나, settings.ini를 찾음)
    # MAIN_PATH = config('MAIN_PATH')가 스크립트 상단에 있으므로, main() 호출 전에 이미 설정됨.
    
    # --- utils.options.py에 필요한 인자 추가 ---
    # 예시: Options 클래스의 __init__ 내 self.parser.add_argument 에 다음 추가
    # self.parser.add_argument('--fine_tune_from_checkpoint', type=int, default=0, help='Episode number of checkpoint to load for fine-tuning. If 0 or not provided, trains from scratch.')
    # self.parser.add_argument('--start_finetune_session_at_episode', type=int, default=1, help='Episode number for this fine-tuning session logging, if server logs episodes.')
    # self.parser.add_argument('--dmms_server_host', type=str, default="127.0.0.1", help='Host for the integrated FastAPI server for DMMS.R')
    # self.parser.add_argument('--dmms_server_port', type=int, default=5000, help='Port for the integrated FastAPI server for DMMS.R')
    # self.parser.add_argument('--restart', type=str, default='0', help='If experiment_dir exists, remove it (1) or not (0).') # 폴더 덮어쓰기 방지용

    main()