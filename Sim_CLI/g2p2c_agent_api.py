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
from typing import Optional
from pathlib import Path

import numpy as np
import torch


# --- 프로젝트 루트 경로 설정 및 sys.path 추가 ---
_current_script_dir = Path(__file__).resolve().parent
_project_root = _current_script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# ------------------------------------------------

from utils.options import Options
from agents.g2p2c.g2p2c import G2P2C

_DEFAULT_RESULTS_SUBDIR = "results/test"
_DEFAULT_ARGS_JSON_REL_PATH = f"{_DEFAULT_RESULTS_SUBDIR}/args.json"
_DEFAULT_PARAMS_PY_REL_PATH = f"{_DEFAULT_RESULTS_SUBDIR}/code/parameters.py" # 수정: code/ 하위
_DEFAULT_CHECKPOINTS_REL_DIR = f"{_DEFAULT_RESULTS_SUBDIR}/checkpoints"


def load_agent(
    device: Optional[torch.device] = None,
    episode: Optional[int] = None,
) -> G2P2C:
    """저장된 체크포인트로부터 G2P2C 모델을 로드한다.

    Parameters
    ----------
    device : torch.device, optional
        모델을 로드할 디바이스. 주어지지 않으면 CPU 사용.
    episode : int, optional
        특정 에피소드 번호의 체크포인트를 불러오고자 할 때 사용. 생략하면
        가장 최신의 에피소드를 자동으로 선택한다.

    Returns
    -------
    G2P2C
        로드된 에이전트 인스턴스.
    """

    args_json_path = (_project_root / _DEFAULT_ARGS_JSON_REL_PATH).resolve()
    params_py_path = (_project_root / _DEFAULT_PARAMS_PY_REL_PATH).resolve()
    checkpoints_dir_abs = (_project_root / _DEFAULT_CHECKPOINTS_REL_DIR).resolve()

    if not args_json_path.is_file():
        raise FileNotFoundError(f"args.json not found at expected path: {args_json_path}")
    if not params_py_path.is_file():
        # run_RL_agent.py에서 parameters.py를 code 폴더로 복사하므로, 해당 경로를 확인
        alt_params_py_path = (_project_root / _DEFAULT_RESULTS_SUBDIR / "code" / "parameters.py").resolve()
        if alt_params_py_path.is_file():
            params_py_path = alt_params_py_path
        else:
            raise FileNotFoundError(f"parameters.py not found at expected paths: {params_py_path} or {alt_params_py_path}")

    if not checkpoints_dir_abs.is_dir():
        raise FileNotFoundError(f"Checkpoints directory not found at expected path: {checkpoints_dir_abs}")

    with open(args_json_path, 'r') as f:
        args_dict = json.load(f)
    
    args = Options().parse([]) 
    
    for k, v in args_dict.items():
        setattr(args, k, v)

    args.experiment_dir = str(args_json_path.parent.parent) 
    os.makedirs(args.experiment_dir, exist_ok=True)

    spec = importlib.util.spec_from_file_location('agent_specific_parameters', str(params_py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for parameters.py from {params_py_path}")
    agent_params_module = importlib.util.module_from_spec(spec)
    sys.modules['agent_specific_parameters'] = agent_params_module # 모듈 캐시에 추가 (중요)
    spec.loader.exec_module(agent_params_module)
    agent_params_module.set_args(args) # 이 함수가 args 객체를 최종적으로 수정

    device = device or torch.device('cpu')
    args.device = str(device)

    agent = G2P2C(args=args, device=device, load=False, path1=None, path2=None)

    if episode is None:
        candidates = []
        for fname in os.listdir(checkpoints_dir_abs):
            if fname.endswith('_Actor.pth'):
                parts = fname.split('_')
                try:
                    num = int(parts[1])
                    candidates.append((num, fname))
                except Exception:
                    continue
        if not candidates:
            raise FileNotFoundError(f'No Actor checkpoint found in {checkpoints_dir_abs}')
        episode_to_load = max(candidates, key=lambda x: x[0])[0]
    else:
        episode_to_load = episode

    actor_path = checkpoints_dir_abs / f'episode_{episode_to_load}_Actor.pth'
    critic_path = checkpoints_dir_abs / f'episode_{episode_to_load}_Critic.pth'

    if not actor_path.is_file():
        raise FileNotFoundError(f"Actor checkpoint file not found: {actor_path}")
    if not critic_path.is_file():
        raise FileNotFoundError(f"Critic checkpoint file not found: {critic_path}")
    
    # PyTorch < 1.6 호환성을 위해 pickle_module 사용 가능성 고려 (일반적으로는 불필요)
    # 또는 weights_only=True (PyTorch 1.13+) 사용 권장. 여기서는 원본 코드 방식 유지.
    try:
        agent.policy.Actor = torch.load(str(actor_path), map_location=device, weights_only=False)
        agent.policy.Critic = torch.load(str(critic_path), map_location=device, weights_only=False)
    except RuntimeError as e: # torch.load에서 pickle 관련 오류 발생 시
        if "weights_only" in str(e): # PyTorch 1.13+에서 weights_only=False가 기본값이 아닐 때 발생 가능
             agent.policy.Actor = torch.load(str(actor_path), map_location=device, weights_only=False)
             agent.policy.Critic = torch.load(str(critic_path), map_location=device, weights_only=False)
        else: # 다른 RuntimeError는 그대로 발생
            raise e


    agent.policy.Actor.eval()
    agent.policy.Critic.eval()
    
    print(f"INFO:     Agent loaded with episode {episode_to_load} weights.")
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