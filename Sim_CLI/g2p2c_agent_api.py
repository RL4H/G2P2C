# Sim_CLI/g2p2_agent_api.py
"""G2P2C 에이전트를 로드하고 추론을 수행하기 위한 보조 모듈.

이 모듈은 ``load_agent`` 와 ``infer_action`` 두 함수를 제공한다. ``load_agent``
는 저장된 체크포인트 파일들을 읽어 ``G2P2C`` 객체를 생성하며, ``infer_action``
은 전처리된 상태 이력을 입력 받아 인슐린 주입률(U/h)을 반환한다. DMMS.R과
FastAPI 서버 사이의 브릿지 역할을 하는 코드다.
"""

import os
import sys
import json
import importlib.util
import traceback # traceback 임포트 추가 (오류 로깅 시 사용 가능)
from typing import Optional, Literal # Literal 임포트 확인
from pathlib import Path

import numpy as np # numpy 임포트 확인 (infer_action 등에서 사용)
import torch

# --- 프로젝트 루트 경로 설정 (기존과 동일) ---
_current_script_dir = Path(__file__).resolve().parent
_project_root = _current_script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# ------------------------------------------------

from utils.options import Options # Options 클래스 임포트 확인
from agents.g2p2c.g2p2c import G2P2C # G2P2C 클래스 임포트 확인

# --- 기본 경로 상수 제거 또는 주석 처리 (동적으로 설정되므로) ---
# _DEFAULT_RESULTS_SUBDIR = "results/test" 
# _DEFAULT_ARGS_JSON_REL_PATH = f"{_DEFAULT_RESULTS_SUBDIR}/args.json"
# _DEFAULT_PARAMS_PY_REL_PATH = f"{_DEFAULT_RESULTS_SUBDIR}/code/parameters.py"
# _DEFAULT_CHECKPOINTS_REL_DIR = f"{_DEFAULT_RESULTS_SUBDIR}/checkpoints"

def load_agent(
    device: Optional[torch.device] = None,
    episode: Optional[int] = None,
    mode: Literal["base", "finetuned"] = "base",
    # 미세 조정 모드일 때 사용할 실험 폴더 ID (예: "finetune_log_test_01")
    experiment_folder_name: Optional[str] = None 
) -> G2P2C:
    """저장된 체크포인트로부터 G2P2C 모델을 로드합니다.

    Parameters
    ----------
    device : torch.device, optional
        모델을 로드할 디바이스. 주어지지 않으면 CPU 사용. (기본값: CPU)
    episode : int, optional
        특정 에피소드 번호의 체크포인트를 불러오고자 할 때 사용. 
        생략하면 해당 경로에서 가장 최신의 에피소드를 자동으로 선택합니다.
    mode : Literal["base", "finetuned"], optional
        "base": results/test/ 경로의 사전 훈련된 기본 모델을 로드합니다. (기본값)
        "finetuned": results/{experiment_folder_name}/ 경로의 미세 조정된 모델을 로드합니다.
                     이 경우 experiment_folder_name 인자가 반드시 제공되어야 합니다.
    experiment_folder_name : str, optional
        mode="finetuned"일 때, 로드할 모델이 포함된 results 하위의 실험 폴더 이름입니다.
        예: "finetune_log_test_01"

    Returns
    -------
    G2P2C
        로드된 에이전트 인스턴스.

    Raises
    ------
    ValueError
        mode="finetuned"일 때 experiment_folder_name이 제공되지 않은 경우.
    FileNotFoundError
        필요한 설정 파일이나 체크포인트 파일을 찾지 못한 경우.
    """
    print(f"INFO: [load_agent] Attempting to load agent with mode='{mode}', episode={episode}, experiment_folder='{experiment_folder_name}'")

    base_config_dir_name = "test" # 기본 모델의 설정 및 체크포인트가 있는 폴더 이름

    if mode == "base":
        current_experiment_folder_name = base_config_dir_name
        args_file_name = "args.json" # 기본 모델의 args 파일명
    elif mode == "finetuned":
        if not experiment_folder_name:
            raise ValueError("For mode='finetuned', 'experiment_folder_name' must be provided.")
        current_experiment_folder_name = experiment_folder_name
        # 미세 조정된 실험의 args 파일명이 'args_finetune.json' 또는 'args.json'일 수 있음
        # 여기서는 'args_finetune.json'을 우선 시도하고, 없으면 'args.json' 시도 (또는 명확히 지정)
        # 사용자의 로그에서는 'args_finetune.json'으로 확인됨
        args_file_name = "extended_training_config.json" 
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'base' or 'finetuned'.")

    # 동적으로 경로 설정
    # MAIN_PATH는 이 파일이 실행되는 컨텍스트에서 (예: decouple을 통해) 올바르게 설정되어 있다고 가정
    # 또는 _project_root를 사용
    results_base_path = _project_root / "results"
    current_experiment_path = results_base_path / current_experiment_folder_name
    
    args_json_path = current_experiment_path / args_file_name
    # args_finetune.json이 없을 경우 args.json으로 fallback (선택적)
    if not args_json_path.is_file() and args_file_name == "args_finetune.json":
        print(f"INFO: [load_agent] '{args_file_name}' not found in {current_experiment_path}. Trying 'args.json'.")
        args_json_path = current_experiment_path / "args.json"

    # parameters.py 경로는 해당 실험 폴더의 'code' 하위로 가정
    # (run_RL_agent_finetune.py에서 코드 복사 시 이 구조를 따름)
    params_py_path = current_experiment_path / "code" / "g2p2c_agent_code" / "parameters.py"
    # 만약 'g2p2c_agent_code' 하위가 아니라 바로 'code' 하위에 있다면 경로 수정
    # 예: params_py_path = current_experiment_path / "code" / "parameters.py"
    #    -> `run_RL_agent_finetune.py`의 `copy_folder` 및 `dst` 경로 설정과 일치해야 함
    #    현재 `copy_folder`는 `agents/g2p2c` 전체를 `g2p2c_agent_code`로 복사하므로 위 경로가 맞음.

    checkpoints_dir_abs = current_experiment_path / "checkpoints"

    print(f"INFO: [load_agent] Using args_json_path: {args_json_path}")
    print(f"INFO: [load_agent] Using params_py_path: {params_py_path}")
    print(f"INFO: [load_agent] Using checkpoints_dir_abs: {checkpoints_dir_abs}")

    if not args_json_path.is_file():
        raise FileNotFoundError(f"Args JSON file not found at expected path: {args_json_path}")
    if not params_py_path.is_file():
        # parameters.py는 보통 'code' 폴더 또는 'code/g2p2c_agent_code' 등에 복사됨
        # `run_RL_agent_finetune.py`의 `copy_folder` 로직에 따라 이 경로가 정확해야 함
        alt_params_py_path_1 = current_experiment_path / "code" / "parameters.py" # 바로 code 밑
        if alt_params_py_path_1.is_file():
             params_py_path = alt_params_py_path_1
             print(f"INFO: [load_agent] Found parameters.py at alternative path: {params_py_path}")
        else:
            raise FileNotFoundError(f"parameters.py not found at primary path {params_py_path} or alt path {alt_params_py_path_1}")


    if not checkpoints_dir_abs.is_dir():
        raise FileNotFoundError(f"Checkpoints directory not found at: {checkpoints_dir_abs}")

    with open(args_json_path, 'r') as f:
        args_dict = json.load(f)
    
    # Options().parse([])는 기본값으로 args 객체를 생성합니다.
    # 여기에 파일에서 읽은 args_dict 내용을 덮어쓰거나 setattr로 설정합니다.
    args = Options().parse([]) 
    
    # args_dict의 키-값 쌍으로 args 객체 속성 설정
    for k, v in args_dict.items():
        # args 객체에 이미 해당 속성이 있는지 확인하거나, setattr을 사용
        if hasattr(args, k):
            setattr(args, k, v)
        else:
            # Options().parse()에 정의되지 않은 새로운 인자일 수 있음 (주의)
            # print(f"WARNING: [load_agent] Argument '{k}' from json not found in Options object, setting it anyway.")
            setattr(args, k, v) # 또는 에러 처리

    # G2P2C 에이전트가 자신의 experiment_dir을 참조할 수 있도록 설정
    args.experiment_dir = str(current_experiment_path) 
    # os.makedirs(args.experiment_dir, exist_ok=True) # load_agent는 읽기 전용이므로 여기서 폴더 생성은 불필요할 수 있음

    # agent_specific_parameters (parameters.py) 로드 및 args 최종 수정
    spec = importlib.util.spec_from_file_location('agent_specific_parameters', str(params_py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for parameters.py from {params_py_path}")
    
    # 모듈 이름 충돌 방지를 위해 유니크한 이름 사용 고려 (선택 사항)
    # module_name_suffix = hashlib.md5(str(params_py_path).encode()).hexdigest()[:8]
    # unique_module_name = f'agent_specific_parameters_{module_name_suffix}'
    # agent_params_module = importlib.util.module_from_spec(spec)
    # sys.modules[unique_module_name] = agent_params_module
    # spec.loader.exec_module(agent_params_module)
    # agent_params_module.set_args(args)

    # 기존 방식 유지 (모듈 이름이 고정되어 여러번 호출 시 문제될 수 있으나, 현재 구조에서는 한번만 호출됨)
    agent_params_module = importlib.util.module_from_spec(spec)
    sys.modules['agent_specific_parameters'] = agent_params_module 
    spec.loader.exec_module(agent_params_module)
    agent_params_module.set_args(args) # 이 함수가 args 객체를 최종적으로 수정

    device = device or torch.device('cpu') # 명시적으로 device가 None이면 cpu 사용
    args.device = str(device) # args에도 device 정보 저장

    # G2P2C 에이전트 인스턴스 생성 (모델 가중치는 아직 로드 안함)
    # 이 시점의 args는 args.json과 parameters.py 내용이 반영된 상태여야 함
    agent = G2P2C(args=args, device=device, load=False, path1=None, path2=None)

    # 에피소드 번호 결정 (None이면 최신 에피소드)
    if episode is None:
        candidates = []
        for fname in os.listdir(checkpoints_dir_abs):
            if fname.endswith('_Actor.pth'): # 또는 에이전트가 저장하는 파일명 규칙에 맞게
                parts = fname.split('_') # 예: 'episode_195_Actor.pth' -> '195' 추출
                if len(parts) > 1 and parts[0] == 'episode':
                    try:
                        num = int(parts[1])
                        candidates.append((num, fname))
                    except ValueError:
                        # print(f"WARNING: Could not parse episode number from {fname}")
                        continue
        if not candidates:
            raise FileNotFoundError(f'No Actor checkpoint files found in {checkpoints_dir_abs}')
        episode_to_load = max(candidates, key=lambda x: x[0])[0]
        print(f"INFO: [load_agent] No specific episode provided. Loading latest: episode {episode_to_load}")
    else:
        episode_to_load = episode

    actor_path = checkpoints_dir_abs / f'episode_{episode_to_load}_Actor.pth'
    critic_path = checkpoints_dir_abs / f'episode_{episode_to_load}_Critic.pth'

    if not actor_path.is_file():
        raise FileNotFoundError(f"Actor checkpoint file not found: {actor_path}")
    if not critic_path.is_file():
        raise FileNotFoundError(f"Critic checkpoint file not found: {critic_path}")
    
    print(f"INFO: [load_agent] Loading weights from: {actor_path} and {critic_path}")
    
    try:
        # G2P2C 에이전트의 policy 객체 내 Actor, Critic 모델에 직접 가중치 로드
        # G2P2C 클래스 구현에 따라 agent.load_policy(actor_path, critic_path) 같은 메소드가 더 적절할 수 있음
        # 현재는 외부에서 직접 할당하는 방식
        agent.policy.Actor = torch.load(str(actor_path), map_location=device, weights_only=False)
        agent.policy.Critic = torch.load(str(critic_path), map_location=device, weights_only=False)
        # weights_only=True/False는 torch.load()의 새로운 인자이며, 보안상 True가 권장되나,
        # 여기서는 state_dict를 로드하므로, torch.load() 자체가 state_dict를 반환한다고 가정.
        # 만약 torch.load가 전체 모델 객체를 반환한다면, agent.policy.Actor = torch.load(...) 방식 사용.
        # 사용자 기존 코드에서는 agent.policy.Actor = torch.load(...) 였으므로, 이를 따르는 것이 안전할 수 있음.
        # 이 경우 G2P2C.__init__에서 self.policy.Actor 등을 None으로 초기화하거나 placeholder 모델로 둬야함.

        # 사용자 기존 코드 방식 (전체 모델 객체 로드):
        # agent.policy.Actor = torch.load(str(actor_path), map_location=device) 
        # agent.policy.Critic = torch.load(str(critic_path), map_location=device)
        # 위 방식 사용 시, G2P2C.__init__에서 self.policy.Actor, self.policy.Critic가 
        # 어떤 형태로 초기화되는지, 그리고 이 할당이 호환되는지 확인 필요.
        # 일반적으로는 모델 구조를 먼저 정의하고 state_dict를 로드하는 것이 더 안전하고 표준적임.
        # 여기서는 state_dict 로드 방식으로 수정 제안. 만약 G2P2C가 이미 모델 구조를 가지고 있다면 이것이 맞음.

    except RuntimeError as e:
        if "weights_only" in str(e): # 이전 코드의 예외 처리 유지
             agent.policy.Actor.load_state_dict(torch.load(str(actor_path), map_location=device)) # weights_only=False 명시 불필요
             agent.policy.Critic.load_state_dict(torch.load(str(critic_path), map_location=device))
        else:
            print(f"ERROR: [load_agent] RuntimeError during torch.load: {e}")
            traceback.print_exc()
            raise e
    except Exception as e:
        print(f"ERROR: [load_agent] Unexpected error during torch.load: {e}")
        traceback.print_exc()
        raise e

    agent.policy.Actor.eval()  # 평가 모드로 설정
    agent.policy.Critic.eval() # 평가 모드로 설정
    
    print(f"INFO: [load_agent] Agent successfully loaded with episode {episode_to_load} weights from '{current_experiment_folder_name}'. Mode: '{mode}'.")
    return agent

def infer_action(
    agent: G2P2C,
    state_hist_processed: np.ndarray, # StateSpace를 거친 정규화된 상태 이력
    hc_state_processed: np.ndarray,   # StateSpace를 거친 정규화된 핸드크래프트 특징
) -> float:
    """전처리된 상태를 이용해 에이전트로부터 인슐린 주입률을 추론한다.

    Parameters
    ----------
    agent : G2P2C
        로드된 G2P2C 에이전트 인스턴스.
    state_hist_processed : np.ndarray
        ``StateSpace`` 로 전처리된 상태 이력 (``feature_history`` × ``n_features``).
    hc_state_processed : np.ndarray
        전처리된 핸드크래프트 특징 벡터.

    Returns
    -------
    float
        클리핑된 인슐린 주입률(U/h).
    """

    # 1. 입력 유효성 검증 (StateSpace를 거쳤으므로, 기본적인 타입 및 형태 위주)
    if not isinstance(state_hist_processed, np.ndarray):
        raise TypeError(f'state_hist_processed must be np.ndarray, got {type(state_hist_processed)}')
    if not isinstance(hc_state_processed, np.ndarray):
        raise TypeError(f'hc_state_processed must be np.ndarray, got {type(hc_state_processed)}')

    expected_hist_shape = (agent.args.feature_history, agent.args.n_features)
    if state_hist_processed.shape != expected_hist_shape:
        raise ValueError(f'state_hist_processed shape must be {expected_hist_shape}, got {state_hist_processed.shape}')
    
    # n_handcrafted_features는 agent.args에 의해 1로 설정됨 (agents/g2p2c/parameters.py)
    expected_hc_shape = (1, agent.args.n_handcrafted_features) 
    if hc_state_processed.shape != expected_hc_shape:
        # hc_state_processed가 (1,)과 같이 1차원 배열로 올 수 있으므로, reshape 시도
        if hc_state_processed.size == agent.args.n_handcrafted_features:
            try:
                hc_state_processed = hc_state_processed.reshape(1, agent.args.n_handcrafted_features)
            except Exception as reshape_err:
                 raise ValueError(f'hc_state_processed shape {hc_state_processed.shape} could not be reshaped to {expected_hc_shape}: {reshape_err}')
        else:
            raise ValueError(f'hc_state_processed shape must be {expected_hc_shape} or have {agent.args.n_handcrafted_features} elements, got {hc_state_processed.shape}')


    if not np.all(np.isfinite(state_hist_processed)):
        raise ValueError('state_hist_processed contains NaN or Inf after StateSpace processing.')
    if not np.all(np.isfinite(hc_state_processed)):
        raise ValueError('hc_state_processed contains NaN or Inf after StateSpace processing.')

    # 2. NumPy 배열을 PyTorch 텐서로 변환
    state_hist_tensor = torch.as_tensor(state_hist_processed, dtype=torch.float32, device=agent.device)
    hc_state_tensor = torch.as_tensor(hc_state_processed, dtype=torch.float32, device=agent.device)
    
    
    # 3. 에이전트 정책을 사용하여 추론
    with torch.no_grad():
        # G2P2C의 ActorCritic.get_action은 내부적으로 predict를 호출하고,
        # predict는 Actor.forward를 호출. Actor.forward는 단일 샘플 (배치 차원 없음)을 받아
        # FeatureExtractor.forward 호출 시 unsqueeze로 배치 차원을 추가함.
        action_data_dict = agent.policy.get_action(state_hist_tensor, hc_state_tensor)
    
    raw_action_model_output = float(action_data_dict['action']) # 모델 출력은 -1 ~ 1 범위

    final_action_U_per_h = agent.args.action_scale * np.exp((raw_action_model_output) * 4) 
    
    # 최종적으로 insulin_min, insulin_max 범위로 클리핑
    action_clipped = float(np.clip(final_action_U_per_h, agent.args.insulin_min, agent.args.insulin_max))
    # action_clipped *= 0
    # 만약 모델의 행동이 이루어지지 않을 경우 위와 같이 0을 곱하면 된다.
    print(f"\n\n-----------------------------\nDEBUG: Model raw output: {raw_action_model_output:.4f}, Scaled action: {final_action_U_per_h:.4f}, Clipped action: {action_clipped:.4f}")
    return action_clipped